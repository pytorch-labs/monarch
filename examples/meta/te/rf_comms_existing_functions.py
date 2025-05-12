import torch
from torch import autograd, distributed as dist, nn


class _MultiplyAllReduce(autograd.Function):
    "Existing user autograd.Function"

    @staticmethod
    def forward(ctx, x, y, pg):
        wa = torch.rand(x.shape, device=x.device)
        ctx.save_for_backward(x, y, wa)
        ctx.my_property = True
        ctx.pg = pg
        z = x * y
        dist.all_reduce(z, op=dist.ReduceOp.SUM, group=pg)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, y, a = ctx.saved_tensors
        assert ctx.my_property
        grad_x = grad_output * y
        grad_y = grad_output * x * a
        dist.all_reduce(grad_x, op=dist.ReduceOp.SUM, group=ctx.pg)
        dist.all_reduce(grad_y, op=dist.ReduceOp.SUM, group=ctx.pg)
        return grad_x, grad_y, None


class MultiplyAllReduce(nn.Module):
    """Existing user nn.Module which wraps the autograd.Function"""

    def __init__(self, pg: dist.ProcessGroup):
        super().__init__()
        self.fc = nn.Linear(3, 3, device="cuda")
        self.pg = pg

    def forward(self, x, y):
        return self.fc(_MultiplyAllReduce.apply(x, y, self.pg))
