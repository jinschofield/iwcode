import torch
import torch.nn.functional as F
import higher
from models.logic_bottleneck import LogicBottleneck

class MAMLPatch(torch.nn.Module):
    """
    Meta-learner for LogicBottleneck using higher library for inner-loop adaptation.
    """
    def __init__(self, inner_lr: float = 0.1, adapt_steps: int = 5):
        super().__init__()
        self.net = LogicBottleneck()
        self.inner_lr = inner_lr
        self.adapt_steps = adapt_steps

    def forward(self, support_x, support_y, query_x, query_y):
        # Use SGD optimizer for inner loop
        inner_opt = torch.optim.SGD(self.net.parameters(), lr=self.inner_lr)
        # Inner-loop adaptation context
        with higher.innerloop_ctx(self.net, inner_opt) as (fnet, diffopt):
            for _ in range(self.adapt_steps):
                preds_sup = fnet(support_x)
                loss_sup = F.mse_loss(preds_sup.view(-1), support_y.float())
                diffopt.step(loss_sup)
            # Query loss after adaptation
            preds_q = fnet(query_x)
            loss_q = F.mse_loss(preds_q.view(-1), query_y.float())
        return loss_q
