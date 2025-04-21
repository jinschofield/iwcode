import torch
import numpy as np

def patch_accuracy(model, patches, labels):
    """
    Compute per-patch accuracy for a patch-based model.
    patches: Tensor [N,9]
    labels: Tensor [N]
    """
    model.eval()
    with torch.no_grad():
        out = model(patches.float())
        pred = (out.view(-1) > 0.5).long()
    return (pred == labels.long()).float().mean().item()


def extract_thresholds(logic_model):
    """
    Extract neighbor weights and birth/survival thresholds from a LogicBottleneck model.
    Returns: (w: np.ndarray[9], b_birth: float, b_surv: float)
    """
    w = logic_model.w.detach().cpu().numpy()
    b_birth = logic_model.b_birth.item()
    b_surv = logic_model.b_surv.item()
    return w, b_birth, b_surv


def rollout_error(model, board, steps=10):
    """
    Placeholder for multi-step rollout error metric.
    """
    raise NotImplementedError("rollout_error not implemented yet")


def board_accuracy_one_step(model, board_t, board_t1):
    """
    Placeholder for one-step board accuracy.
    """
    raise NotImplementedError("board_accuracy_one_step not implemented yet")
