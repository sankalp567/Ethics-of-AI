import torch
import torch.nn as nn

def normalize(x):
    mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std


def pgd_attack(model, images, labels, epsilon, alpha, iters):
    original_images = images.clone().detach()

    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)

    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(iters):
        adv_images.requires_grad = True

        outputs = model(normalize(adv_images))
        loss = loss_fn(outputs, labels.float().unsqueeze(1))

        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()

        eta = torch.clamp(
            adv_images - original_images,
            min=-epsilon,
            max=epsilon
        )

        adv_images = torch.clamp(original_images + eta, 0, 1).detach()

    return adv_images