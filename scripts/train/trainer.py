import os
import logging
import torch

from ..model.sparse_encoders import SparseModel
from .bi_encoder_wrapper import BiEncoderWrapper
from ..utils import gather_rep
from ..data.dataset import CombinedRandomSampler, CombinedDataset

from transformers import Trainer
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader
import json

logger = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    def __init__(self, sparse_model, inf_free=True):
        super().__init__()
        self.sparse_model = sparse_model
        self.inf_free = inf_free

    def forward(self, inputs):
        d_rep = self.sparse_model(
            inf_free=False,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        q_rep = self.sparse_model(
            inf_free=self.inf_free,
            input_ids=inputs["q_input_ids"],
            attention_mask=inputs["q_attention_mask"],
        )
        return d_rep, q_rep

    def save(self, output_dir, **kwargs):
        self.sparse_model.backbone.save_pretrained(output_dir, **kwargs)
        self.sparse_model.tokenizer.save_pretrained(output_dir)

        if self.sparse_model.idf_requires_grad:
            idf_json = dict()
            idf_vector = self.sparse_model.idf_vector.detach().cpu()
            for idx in idf_vector.nonzero():
                idf_json[self.sparse_model.tokenizer._convert_id_to_token(idx)] = float(
                    idf_vector[idx]
                )
            with open(os.path.join(output_dir, "idf.json"), "w") as f:
                json.dump(idf_json, f)


class SparseModelTrainer(Trainer):

    def __init__(self, model_args, data_args, loss_functions, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.loss_functions = loss_functions
        self.ranking_loss_moving_avg = 0
        self._threshold_handlers = {
            "concurrent": self._concurrent_threshold,
            "nonzero_mean": self._nonzero_mean_threshold,
            "detach_zero": self._detach_zero_threshold,
        }
        kwargs["model"] = ModelWrapper(kwargs["model"], model_args.inf_free)
        super().__init__(**kwargs)

    def flops_value(
        self, representation: torch.Tensor, group_num: int = 1
    ) -> torch.Tensor:
        """Calculate FLOPS value based on representation and threshold settings.

        Args:
            representation: Input tensor of shape (ndevice * batch_size) * vocab_dim
            group_num: Number of semantic similar documents representations in one batch

        Returns:
            torch.Tensor: Calculated FLOPS value
        """
        representation = representation.reshape(-1, group_num, representation.shape[-1])

        if self.data_args.flops_threshold is None:
            return torch.sum(torch.mean(torch.abs(representation), dim=0) ** 2)

        handler = self._threshold_handlers.get(self.data_args.threshold_type)
        if not handler:
            raise ValueError(
                f"Invalid flops threshold type: {self.data_args.threshold_type}"
            )

        return handler(representation)

    def _get_doc_mask(self, representation: torch.Tensor) -> torch.Tensor:
        """Calculate document mask based on threshold."""
        w_j_per_doc = torch.abs(representation)
        doc_length = torch.norm(w_j_per_doc, p=0, dim=2)
        return (doc_length > self.data_args.flops_threshold).float()

    def _concurrent_threshold(self, representation: torch.Tensor) -> torch.Tensor:
        """Handle concurrent threshold type."""
        w_j_per_doc = torch.abs(representation)
        mask = self._get_doc_mask(representation)
        mask = mask.unsqueeze(2).repeat(1, 1, w_j_per_doc.shape[2])
        flops_per_average_token = torch.mean(mask * w_j_per_doc, dim=0) ** 2
        return torch.sum(flops_per_average_token)

    def _nonzero_mean_threshold(self, representation: torch.Tensor) -> torch.Tensor:
        """Handle nonzero_mean threshold type."""
        w_j_per_doc = torch.abs(representation)
        mask = self._get_doc_mask(representation)
        index = torch.nonzero(mask).squeeze(1)

        if index.numel() == 0:
            return torch.tensor(0.0)

        flops_per_average_token = torch.mean(w_j_per_doc[index], dim=0) ** 2
        return torch.sum(flops_per_average_token)

    def _detach_zero_threshold(self, representation: torch.Tensor) -> torch.Tensor:
        """Handle detach_zero threshold type."""
        w_j_per_doc = torch.abs(representation)
        mask = self._get_doc_mask(representation)
        index = torch.nonzero(mask).squeeze(1)

        if index.numel() == 0:
            return torch.tensor(0.0)

        w_j_per_doc[index] = w_j_per_doc[index].detach()
        flops_per_average_token = torch.mean(w_j_per_doc[index], dim=0) ** 2
        return torch.sum(flops_per_average_token)

    def get_lambda(self, lambda_value, lambda_T):
        if self.state.global_step >= lambda_T:
            return lambda_value
        step = self.state.global_step + 1
        return lambda_value * (step / lambda_T) ** 2

    def compute_loss(self, model: SparseModel, inputs, return_outputs=False):
        if hasattr(self, "bi_encoder_teacher"):
            scores = self.bi_encoder_teacher.get_scores_batch(
                q_features_list=inputs["query"][1:], d_features_list=inputs["docs"][1:]
            )
            inputs["scores"] = scores

        flops_loss = 0

        # we construct this input and model wrapper to make it work for both data parallel and DDP
        model_wrapper_input = {
            "q_input_ids": inputs["query"][0]["input_ids"],
            "q_attention_mask": inputs["query"][0]["attention_mask"],
            "input_ids": inputs["docs"][0]["input_ids"],
            "attention_mask": inputs["docs"][0]["attention_mask"],
        }

        d_rep, q_rep = model(model_wrapper_input)
        d_rep = gather_rep(d_rep, self.accelerator)
        q_rep = gather_rep(q_rep, self.accelerator)
        if "scores" in inputs:
            inputs["scores"] = gather_rep(inputs["scores"], self.accelerator)
        d_flops = self.flops_value(d_rep, d_rep.shape[0] // q_rep.shape[0])
        flops_loss += d_flops * self.get_lambda(
            self.data_args.flops_d_lambda, self.data_args.flops_d_T
        )

        if not self.model_args.inf_free:
            flops_loss += self.flops_value(q_rep) * self.get_lambda(
                self.data_args.flops_q_lambda, self.data_args.flops_q_T
            )

        ranking_loss = 0
        for loss_function in self.loss_functions:
            ranking_loss += loss_function.get_loss(
                q_rep=q_rep, d_rep=d_rep, inputs=inputs
            )
        self.ranking_loss_moving_avg = (
            0.01 * ranking_loss.item() + 0.99 * self.ranking_loss_moving_avg
        )

        loss = ranking_loss + flops_loss
        outputs = {
            "q_rep": q_rep,
            "d_rep": d_rep,
        }

        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"Step {self.state.global_step}. ranking loss moving avg:{self.ranking_loss_moving_avg}, d_flops: {d_flops}, flops_loss: {flops_loss} avg doc length: {(d_rep>0).sum()/d_rep.shape[0]}"
            )
        # DP reduce grad by sum, while DDP reduce grad by mean
        # scale the loss to fix the gap
        loss = loss * self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

    def set_bi_encoder_teacher(self):
        self.bi_encoder_teacher = BiEncoderWrapper(
            types=self.data_args.kd_ensemble_teacher_kwargs["types"],
            model_ids=self.data_args.kd_ensemble_teacher_kwargs["model_ids"],
            use_in_batch_negatives=self.data_args.use_in_batch_negatives,
            score_scale=self.data_args.kd_ensemble_teacher_kwargs.get(
                "score_scale", 30
            ),
        )
        self.bi_encoder_teacher.accelerator = self.accelerator
        for i, model in enumerate(self.bi_encoder_teacher.models):
            self._move_model_to_device(model, self.args.device)
            self.bi_encoder_teacher.models[i] = self._wrap_model(model, training=False)
            use_accelerator_prepare = (
                True if model is self.bi_encoder_teacher.models[i] else False
            )
            if use_accelerator_prepare:
                self.accelerator.prepare(self.bi_encoder_teacher.models[i])

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if isinstance(train_dataset, CombinedDataset):
            logger.info("Combined dataset. Set combined sampler.")
            sampler = CombinedRandomSampler(
                train_dataset.datasets, batch_size=self._train_batch_size
            )
            dataloader_params.pop("sampler")
            dataloader_params.pop("batch_size")
            dataloader_params.pop("drop_last")
            dataloader_params["batch_sampler"] = sampler

        if hasattr(train_dataset, "no_prepare") and self.accelerator.num_processes > 1:
            if train_dataset.no_prepare:
                logger.info("no accelerator prepare")
                return DataLoader(train_dataset, **dataloader_params)
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
