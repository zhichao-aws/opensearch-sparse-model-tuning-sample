import torch
import logging

logger = logging.getLogger(__name__)


class SparseTrainingLoss:
    def __init__(self, weight=1):
        self.weight = weight

    def __call__(self, q_rep, d_rep, inputs):
        raise NotImplementedError

    def get_loss(self, q_rep, d_rep, inputs):
        return self.weight * self.__call__(q_rep, d_rep, inputs)


class KLDivLoss(SparseTrainingLoss):
    def __init__(self, use_in_batch_negatives=False, weight=1):
        self.use_in_batch_negatives = use_in_batch_negatives
        self.loss = torch.nn.KLDivLoss(reduction="none")
        super().__init__(weight)

    def __call__(self, q_rep, d_rep, inputs):
        teacher_scores = inputs["scores"]

        if not self.use_in_batch_negatives:
            batch_size = q_rep.shape[0]
            d_rep = d_rep.reshape(
                batch_size, d_rep.shape[0] // batch_size, d_rep.shape[-1]
            )
            student_scores = torch.bmm(
                d_rep, q_rep.reshape(batch_size, -1, 1)
            ).squeeze()
        else:
            student_scores = torch.matmul(q_rep, d_rep.t())

        student_scores = torch.log_softmax(student_scores, dim=1)
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        loss = self.loss(student_scores, teacher_scores).sum(dim=1).mean(dim=0)
        return loss


class MarginMSELoss(SparseTrainingLoss):
    def __init__(self, use_in_batch_negatives=False, weight=1):
        self.use_in_batch_negatives = use_in_batch_negatives
        self.loss = torch.nn.MSELoss()
        self.margin_func = (
            lambda x: x[:, 0].reshape(-1, 1).expand(x.shape[0], x.shape[1] - 1)
            - x[:, 1:]
        )
        super().__init__(weight)

    def __call__(self, q_rep, d_rep, inputs):
        teacher_scores = inputs["scores"]
        if not self.use_in_batch_negatives:
            batch_size = q_rep.shape[0]
            d_rep = d_rep.reshape(
                batch_size, d_rep.shape[0] // batch_size, d_rep.shape[-1]
            )
            student_scores = torch.bmm(
                d_rep, q_rep.reshape(batch_size, -1, 1)
            ).squeeze()
        else:
            student_scores = torch.matmul(q_rep, d_rep.t())

        loss = self.loss(
            self.margin_func(student_scores), self.margin_func(teacher_scores)
        )
        return loss


class InfoNCELoss(SparseTrainingLoss):
    def __init__(self, weight=1, use_in_batch_negatives=False):
        self.loss = torch.nn.CrossEntropyLoss()
        self.use_in_batch_negatives = use_in_batch_negatives
        super().__init__(weight)

    def __call__(self, q_rep, d_rep, inputs):
        # q_rep: batch_size * dim
        # d_rep: (batch_size*(1+neg_num)) * dim
        bs = q_rep.shape[0]
        indices = torch.arange(0, d_rep.shape[0], step=d_rep.shape[0] // bs)
        pos_rep = d_rep[indices]
        scores_pos = torch.matmul(q_rep, pos_rep.t()).diag().unsqueeze(-1)

        mask = torch.ones(d_rep.shape[0], dtype=torch.bool)
        mask[indices] = False
        neg_rep = d_rep[mask]
        if self.use_in_batch_negatives:
            scores_neg = torch.matmul(q_rep, neg_rep.t())
        else:
            neg_rep = neg_rep.reshape(bs, -1, neg_rep.shape[-1])
            scores_neg = torch.bmm(neg_rep, q_rep.reshape(bs, -1, 1)).reshape(bs, -1)

        scores = torch.cat([scores_pos, scores_neg], dim=1)
        target = torch.zeros(scores.shape).to(scores.device)
        target[:, 0] = 1
        output = self.loss(scores, target)
        return output


LOSS_CLS_MAP = {"infonce": InfoNCELoss, "kldiv": KLDivLoss, "marginmse": MarginMSELoss}
