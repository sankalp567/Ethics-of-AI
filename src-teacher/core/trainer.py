import time
import torch
import torch.nn as nn
import numpy as np
from core.attacks import pgd_attack
from core.metrics import calculate_accuracy
from core.attacks import pgd_attack, normalize
from config import Config

def train_teacher_epoch(model, dataloader, optimizer, criterion, device, epoch, use_pgd=True, epsilon=Config.EPSILON, attack_steps=Config.ATTACK_STEPS):
    """
    Trains the Teacher model for one epoch using standard Adversarial Training (AT).
    """
    model.train()
    running_loss = 0.0
    correct_standard = 0
    correct_robust = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, targets, _) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        
        # 1. Generate Adversarial Examples
        if use_pgd:
            # Turn on eval mode for BN layers during attack (standard practice in some AT, but we keep train() to match original TRADES/Madry unless specified)
            model.eval()
            adv_images = pgd_attack(model, images, targets, epsilon, Config.ROBUST_ALPHA, attack_steps)
            model.train()
        else:
            adv_images = images
        
        # 2. Forward pass with both adversarial and clean examples
        optimizer.zero_grad()
        
        outputs_adv = model(normalize(adv_images))
        loss_adv = criterion(outputs_adv, targets.float().unsqueeze(1))
        
        if use_pgd:
            outputs_clean = model(normalize(images))
            loss_clean = criterion(outputs_clean, targets.float().unsqueeze(1))
            loss = 0.5 * loss_clean + 0.5 * loss_adv
        else:
            loss = loss_adv
            
        # 3. Backward and Optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Calculate standard accuracy for logging (needs clean forward pass)
        with torch.no_grad():
            outputs_clean = model(normalize(images))
            c_std, _ = calculate_accuracy(outputs_clean, targets)
            correct_standard += c_std
            
            c_rob, _ = calculate_accuracy(outputs_adv, targets)
            correct_robust += c_rob
            
        total += targets.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1} [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Clean Acc: {c_std/targets.size(0)*100:.2f}% "
                  f"Rob Acc: {c_rob/targets.size(0)*100:.2f}%")
            
    epoch_loss = running_loss / total
    epoch_std_acc = correct_standard / total * 100
    epoch_rob_acc = correct_robust / total * 100
    
    print(f"==> Epoch {epoch+1} Summary: "
          f"Time: {time.time() - start_time:.1f}s | "
          f"Loss: {epoch_loss:.4f} | "
          f"Clean Acc: {epoch_std_acc:.2f}% | "
          f"Rob Acc: {epoch_rob_acc:.2f}%")
          
    return epoch_loss, epoch_std_acc, epoch_rob_acc

def evaluate_teacher(model, dataloader, criterion, device, is_robust=True):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    # multi-sensitive store
    all_sensitives = {k: [] for k in Config.SENSITIVE_ATTRS}

    for images, targets, sensitives in dataloader:
        images, targets = images.to(device), targets.to(device)

        if is_robust:
            images = pgd_attack(
                model, images, targets,
                Config.EPSILON,
                Config.ROBUST_ALPHA,
                Config.EVAL_ATTACK_STEPS
            )

        with torch.no_grad():
            outputs = model(normalize(images))
            loss = criterion(outputs, targets.float().unsqueeze(1))
            running_loss += loss.item() * targets.size(0)

            preds = (outputs >= 0.0).squeeze().long()

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        for attr in Config.SENSITIVE_ATTRS:
            all_sensitives[attr].extend(
                sensitives[attr].cpu().numpy()
            )

    avg_loss = running_loss / len(all_targets)

    preds_t = np.array(all_preds)
    targets_t = np.array(all_targets)

    acc = (preds_t == targets_t).mean()

    return avg_loss, acc, preds_t, targets_t, all_sensitives