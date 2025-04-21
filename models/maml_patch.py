import torch
import torch.nn.functional as F
import learn2learn as l2l
from models.logic_bottleneck import LogicBottleneck

class MAMLPatch(torch.nn.Module):
    """
    MAML wrapper for LogicBottleneck over 3x3 patches.
    Performs inner-loop adaptation on support set and computes query loss.
    """
    def __init__(self, inner_lr: float = 0.1, adapt_steps: int = 5):
        super().__init__()
        # base logic model
        self.net = LogicBottleneck()
        # MAML meta-learner
        self.maml = l2l.algorithms.MAML(self.net, lr=inner_lr)
        self.adapt_steps = adapt_steps

    def forward(self, support_x, support_y, query_x, query_y):
        # support_x: [n_support,9], support_y: [n_support]
        # query_x: [n_query,9], query_y: [n_query]
        learner = self.maml.clone()
        # Inner loop adaptation
        for _ in range(self.adapt_steps):
            preds_sup = learner(support_x)
            loss_sup = F.mse_loss(preds_sup.view(-1), support_y.float())
            learner.adapt(loss_sup)
        # Query loss
        preds_q = learner(query_x)
        loss_q = F.mse_loss(preds_q.view(-1), query_y.float())
        return loss_q
