import torch
from typing import Optional


class MyModel:
    def __init__(self, flops_threshold: Optional[float] = None, threshold_type: str = "concurrent"):
        self.flops_threshold = flops_threshold
        self.threshold_type = threshold_type
        self._threshold_handlers = {
            "concurrent": self._concurrent_threshold,
            "nonzero_mean": self._nonzero_mean_threshold,
            "detach_zero": self._detach_zero_threshold
        }

    def flops_value(self, representation: torch.Tensor, group_num: int = 1) -> torch.Tensor:
        representation = representation.reshape(-1, group_num, representation.shape[-1])
        
        if self.flops_threshold is None:
            return torch.sum(torch.mean(torch.abs(representation), dim=0) ** 2)
            
        handler = self._threshold_handlers.get(self.threshold_type)
        if not handler:
            raise ValueError(f"Invalid flops threshold type: {self.threshold_type}")
            
        return handler(representation)

    def _get_doc_mask(self, representation: torch.Tensor) -> torch.Tensor:
        w_j_per_doc = torch.abs(representation)
        doc_length = torch.norm(w_j_per_doc, p=0, dim=2)
        return (doc_length > self.flops_threshold).float()

    def _concurrent_threshold(self, representation: torch.Tensor) -> torch.Tensor:
        w_j_per_doc = torch.abs(representation)
        mask = self._get_doc_mask(representation)
        mask = mask.unsqueeze(2).expand(-1, -1, w_j_per_doc.shape[2])
        flops = torch.mean(mask * w_j_per_doc, dim=0) ** 2
        return torch.sum(flops)

    def _nonzero_mean_threshold(self, representation: torch.Tensor) -> torch.Tensor:
        w_j_per_doc = torch.abs(representation)
        mask = self._get_doc_mask(representation).bool()
        
        if not torch.any(mask):
            return torch.tensor(0.0)
            
        selected_docs = w_j_per_doc[mask]
        flops = torch.mean(selected_docs, dim=0) ** 2
        return torch.sum(flops)

    def _detach_zero_threshold(self, representation: torch.Tensor) -> torch.Tensor:
        w_j_per_doc = torch.abs(representation)
        mask = self._get_doc_mask(representation).bool()
        
        if not torch.any(mask):
            return torch.tensor(0.0)
        
        # inverse mask
        mask = ~mask
        w_j_per_doc = w_j_per_doc.clone()
        w_j_per_doc[mask] = w_j_per_doc[mask].detach()
        flops = torch.mean(w_j_per_doc, dim=0) ** 2
        return torch.sum(flops)


def test_model(representation: torch.Tensor, threshold_type: Optional[str] = None):
    model = MyModel(flops_threshold=2, threshold_type=threshold_type) if threshold_type else MyModel()
    mode = threshold_type if threshold_type else "no threshold"
    
    print(f"\nTesting {mode} mode:")
    representation.grad = None
    
    loss = model.flops_value(representation, group_num=len(representation))
    print(f"Loss with {mode}:", loss.item())
    
    loss.backward()
    print(f"Gradients with {mode}:", representation.grad)


if __name__ == "__main__":
    representation = torch.tensor([
        [1.0, 1.2, 0.0, 0.0],
        [1.0, 1.2, 0.0, 0.0],
        [1.5, 1.3, 1.3, 1.4],
        [2.0, 1.6, 1.7, 1.8],
    ], requires_grad=True)

    # Test all modes
    test_model(representation)
    for threshold_type in ["concurrent", "nonzero_mean", "detach_zero"]:
        test_model(representation, threshold_type)
