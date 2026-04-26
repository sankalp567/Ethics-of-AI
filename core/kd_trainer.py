import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import pgd_attack, normalize
from core.metrics import calculate_accuracy
from config import Config


def get_teacher_feat(model, x):
    # ResNet18 penultimate
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)   # [B,512]
    return x


def get_student_feat(model, x):
    # MobileNet Small penultimate
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x

def train_kd_epoch(
    teacher,
    student,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    use_pgd=True,
    epsilon=8/255,
    attack_steps=10,
    T=4.0,
    alpha=0.5,
    beta=0.5
):
    import torch.nn.functional as F

    student.train()
    teacher.eval()

    running_loss = 0
    correct_clean = 0
    correct_rob = 0
    total = 0

    for batch_idx, (images, labels, _) in enumerate(dataloader):

        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        # ==================================================
        # PGD ADV EXAMPLES (attack student)
        # ==================================================
        if use_pgd:
            student.eval()

            adv_images = pgd_attack(
                student,
                images,
                labels.squeeze(1).long(),
                epsilon,
                Config.ROBUST_ALPHA,
                attack_steps
            )

            student.train()
        else:
            adv_images = images

        optimizer.zero_grad()

        # ==================================================
        # CLEAN / ADV LOGITS
        # ==================================================
        x_clean = normalize(images)
        x_adv   = normalize(adv_images)

        out_clean = student(x_clean)
        out_adv   = student(x_adv)

        # ==================================================
        # HARD LABEL LOSSES
        # ==================================================
        loss_clean = criterion(out_clean, labels)
        loss_adv   = criterion(out_adv, labels)

        # ==================================================
        # FEATURE DISTILLATION
        # Teacher: ResNet18 penultimate = 512 dim
        # Student: MobileNet Small projected -> 512
        # ==================================================
        with torch.no_grad():

            t = teacher.conv1(x_clean)
            t = teacher.bn1(t)
            t = teacher.relu(t)
            t = teacher.maxpool(t)

            t = teacher.layer1(t)
            t = teacher.layer2(t)
            t = teacher.layer3(t)
            t = teacher.layer4(t)

            t = teacher.avgpool(t)
            teacher_feat = torch.flatten(t, 1)      # [B,512]

        s = student.features(x_clean)
        s = student.avgpool(s)
        student_feat = torch.flatten(s, 1)          # [B,D]

        student_feat = student.proj(student_feat)   # [B,512]

        loss_kd = F.mse_loss(student_feat, teacher_feat)

        # ==================================================
        # TOTAL LOSS
        # ==================================================
        loss = (
            0.4 * loss_clean +
            0.4 * loss_adv +
            0.2 * loss_kd
        )

        loss.backward()
        optimizer.step()

        # ==================================================
        # METRICS
        # ==================================================
        c1, _ = calculate_accuracy(
            out_clean,
            labels.squeeze(1).long()
        )

        c2, _ = calculate_accuracy(
            out_adv,
            labels.squeeze(1).long()
        )

        correct_clean += c1
        correct_rob += c2
        total += images.size(0)

        running_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1} "
                f"[{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return (
        running_loss / total,
        100 * correct_clean / total,
        100 * correct_rob / total
    )