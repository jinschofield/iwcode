import torch

def pgd_patch_adversary(model, patch, label, eps=0.3, alpha=0.1, steps=3):
    """
    Generate adversarial patch for a patch-predictor model via PGD.
    patch: Tensor[1,9] float
    label: Tensor[1]
    """
    patch_adv = patch.clone().detach().requires_grad_(True)
    loss_fn = torch.nn.BCELoss()
    for _ in range(steps):
        out = model(patch_adv)
        loss = loss_fn(out.view(-1), label.float())
        loss.backward()
        with torch.no_grad():
            patch_adv += alpha * patch_adv.grad.sign()
            # project to [0,1]
            patch_adv = patch_adv.clamp(0, 1)
            patch_adv.grad.zero_()
    # threshold back to binary
    return (patch_adv > 0.5).float()
