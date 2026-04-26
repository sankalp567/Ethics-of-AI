import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from core.metrics import calculate_full_fairness_metrics

from config import Config
from data.celeba import get_celeba_dataloaders
import torchvision.models as models
from core.trainer import train_teacher_epoch, evaluate_teacher

def main():
    parser = argparse.ArgumentParser(description="Train robust teacher")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Total number of epochs to train')
    args = parser.parse_args()
    
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_celeba_dataloaders()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing MobileNetSmall studentmnsmallA Model...")
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)

    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, 
                          momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY, nesterov=True)
                          
    # Standard StepLR used in original ARD setting
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS
    )

    # create checkpoint dir if not exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    start_epoch = 0
    best_rob_acc = 0.0
    history = []
    
    ckp_path = os.path.join(Config.CHECKPOINT_DIR, 'MobileNetSmall_studentmnsmallA_best.pth')
    if args.resume and os.path.exists(ckp_path):
        print(f"Loading checkpoint from {ckp_path}")
        checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_rob_acc = checkpoint.get('best_rob_acc', 0.0)
        
        # Load history if exists
        hist_path = os.path.join(Config.CHECKPOINT_DIR, 'training_history.json')
        if os.path.exists(hist_path):
            with open(hist_path, 'r') as f:
                history = json.load(f)
                
        # Fast forward scheduler
        for _ in range(start_epoch):
            scheduler.step()
            
        print(f"Resumed from epoch {start_epoch} with best robust acc {best_rob_acc*100:.2f}%")

    print(f"Starting Pre-training of MobileNetSmall (studentmnsmallA) from epoch {start_epoch+1} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Progressive PGD Schedule
        if epoch < 2:
            use_pgd = False
            current_eps = 0.0
            current_steps = 0
        else:
            use_pgd = True
            # We hardcode the 7.0 divisor (10-3) to maintain the identical 
            # original epoch 3 to 10 progression rate, but capped at 1.0 for epochs ≥ 10
            progress = min(1.0, (epoch - 2) / 7.0)
            
            # Epsilon grows from 2/255 to 8/255
            current_eps = (2.0 + progress * (8.0 - 2.0)) / 255.0
            # Steps grow from 2 to 10
            current_steps = int(round(2.0 + progress * (10.0 - 2.0)))
            
        print(f"--- PGD Schedule -> Enabled: {use_pgd}, Epsilon: {current_eps*255:.2f}/255, Steps: {current_steps} ---")
        
        train_loss, train_std_acc, train_rob_acc = train_teacher_epoch(
            model, train_loader, optimizer, criterion, device, epoch, 
            use_pgd=use_pgd, epsilon=current_eps, attack_steps=current_steps
        )
                
        # Validation
        print("Running Validation...")

        # -----------------------------------
        # CLEAN EVAL
        # returns:
        # loss, acc, preds, labels, sens_dict
        # -----------------------------------
        val_clean_loss, val_clean_acc, preds_c, labels_c, sens_c = evaluate_teacher(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            is_robust=False
        )

        # -----------------------------------
        # ROBUST EVAL
        # -----------------------------------
        val_rob_loss, val_rob_acc, preds_r, labels_r, sens_r = evaluate_teacher(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            is_robust=True
        )

        print(f"[Val] Clean Acc: {val_clean_acc*100:.2f}% | Rob Acc: {val_rob_acc*100:.2f}%")

        # ==========================================================
        # FAIRNESS MATRIX
        # ==========================================================
        rows = []

        for attr in Config.SENSITIVE_ATTRS:

            # ---------------- CLEAN ----------------
            clean_metrics = calculate_full_fairness_metrics(
                preds=torch.tensor(preds_c),
                labels=torch.tensor(labels_c),
                sensitive_attrs=torch.tensor(sens_c[attr])
            )

            # ---------------- ROBUST ----------------
            robust_metrics = calculate_full_fairness_metrics(
                preds=torch.tensor(preds_r),
                labels=torch.tensor(labels_r),
                sensitive_attrs=torch.tensor(sens_r[attr])
            )

            row = {
                "Sensitive": attr,

                # sample counts
                "Pos N": clean_metrics["group1_total"],
                "Neg N": clean_metrics["group0_total"],

                # subgroup clean acc
                "Pos Clean Acc": round(clean_metrics["group1_acc"] * 100, 2),
                "Neg Clean Acc": round(clean_metrics["group0_acc"] * 100, 2),

                # subgroup robust acc
                "Pos Robust Acc": round(robust_metrics["group1_acc"] * 100, 2),
                "Neg Robust Acc": round(robust_metrics["group0_acc"] * 100, 2),

                # overall F1
                "Clean F1": round(clean_metrics["overall_f1"] * 100, 2),
                "Robust F1": round(robust_metrics["overall_f1"] * 100, 2),

                # gaps
                "Acc Gap": round(clean_metrics["acc_gap"] * 100, 2),
                "Robust Gap": round(robust_metrics["acc_gap"] * 100, 2),
                "F1 Gap": round(robust_metrics["f1_gap"] * 100, 2),
            }

            rows.append(row)

        df_matrix = pd.DataFrame(rows)

        print("\n==============================================================")
        print(" FAIRNESS AUDIT MATRIX (Validation Set)")
        print("==============================================================")
        print(df_matrix.to_string(index=False))
        print("==============================================================\n")

        # ==========================================================
        # SAVE HISTORY
        # ==========================================================
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_clean_acc": train_std_acc,
            "train_rob_acc": train_rob_acc,

            "val_clean_loss": val_clean_loss,
            "val_clean_acc": val_clean_acc * 100,

            "val_rob_loss": val_rob_loss,
            "val_rob_acc": val_rob_acc * 100
        }

        history.append(metrics)

        with open(os.path.join(Config.CHECKPOINT_DIR, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # step scheduler
        scheduler.step()

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }

        torch.save(
            ckpt,
            f"{Config.CHECKPOINT_DIR}/studentmnsmallA_epoch_{epoch+1}.pth"
        )

        # ==========================================================
        # SAVE BEST MODEL
        # ==========================================================
        if val_rob_acc > best_rob_acc:
            best_rob_acc = val_rob_acc
            print(f"New best robust accuracy: {best_rob_acc*100:.2f}%. Saving model...")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rob_acc': best_rob_acc,
            }

            torch.save(
                checkpoint,
                os.path.join(Config.CHECKPOINT_DIR, 'MobileNetSmall_studentmnsmallA_best.pth')
            )

if __name__ == '__main__':
    main()
