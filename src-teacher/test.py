# audit_checkpoint.py
# Audits checkpoint across ALL CelebA attributes automatically

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models

from config import Config
from data.celeba import get_celeba_dataloaders
from core.trainer import evaluate_teacher
from core.metrics import calculate_full_fairness_metrics


def load_model(device):
    # model = models.resnet34(weights=None)
    # model.fc = nn.Linear(model.fc.in_features, 1)
    # model = model.to(device)
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    model = model.to(device)
    return model


def get_checkpoint_path(args):
    if args.ckpt:
        return args.ckpt

    if args.epoch is None:
        return os.path.join(
            Config.CHECKPOINT_DIR,
            f"studentmnsmallA_epoch_5.pth"
        )

    return os.path.join(
        Config.CHECKPOINT_DIR,
        f"teacher_epoch_{args.epoch}.pth"
    )


def detect_all_attributes():
    df = pd.read_csv(Config.ATTR_FILE)

    ignore_cols = {"image_id"}

    attrs = [c for c in df.columns if c not in ignore_cols]

    # remove target attr from sensitive scan
    attrs = [c for c in attrs if c != Config.TARGET_ATTR]

    return attrs


def run_audit(model, loader, criterion, device, attrs):
    print("Running Clean Evaluation...")
    clean_loss, clean_acc, preds_c, labels_c, sens_c = evaluate_teacher(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        is_robust=False
    )

    print("Running Robust Evaluation...")
    rob_loss, rob_acc, preds_r, labels_r, sens_r = evaluate_teacher(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        is_robust=True
    )

    print(f"\n[VAL] Clean Acc : {clean_acc*100:.2f}%")
    print(f"[VAL] Robust Acc: {rob_acc*100:.2f}%")

    rows = []

    for attr in attrs:

        if attr not in sens_c:
            continue

        clean_metrics = calculate_full_fairness_metrics(
            preds=torch.tensor(preds_c),
            labels=torch.tensor(labels_c),
            sensitive_attrs=torch.tensor(sens_c[attr])
        )

        robust_metrics = calculate_full_fairness_metrics(
            preds=torch.tensor(preds_r),
            labels=torch.tensor(labels_r),
            sensitive_attrs=torch.tensor(sens_r[attr])
        )

        # Replace the rows.append({...}) block inside run_audit() with this

        rows.append({
            "Sensitive": attr,
            "Pos N": clean_metrics["group1_total"],
            "Neg N": clean_metrics["group0_total"],

            # Accuracy
            "Pos Clean Acc": round(clean_metrics["group1_acc"] * 100, 2),
            "Neg Clean Acc": round(clean_metrics["group0_acc"] * 100, 2),

            "Pos Robust Acc": round(robust_metrics["group1_acc"] * 100, 2),
            "Neg Robust Acc": round(robust_metrics["group0_acc"] * 100, 2),

            # F1 (per-group)
            "Pos Clean F1": round(clean_metrics["group1_f1"] * 100, 2),
            "Neg Clean F1": round(clean_metrics["group0_f1"] * 100, 2),

            "Pos Robust F1": round(robust_metrics["group1_f1"] * 100, 2),
            "Neg Robust F1": round(robust_metrics["group0_f1"] * 100, 2),

            # Overall F1
            "Clean F1": round(clean_metrics["overall_f1"] * 100, 2),
            "Robust F1": round(robust_metrics["overall_f1"] * 100, 2),

            # Gaps
            "Acc Gap": round(clean_metrics["acc_gap"] * 100, 2),
            "Robust Gap": round(robust_metrics["acc_gap"] * 100, 2),
            "F1 Gap": round(robust_metrics["f1_gap"] * 100, 2),
        })
    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="Robust Gap",
        ascending=False
    ).reset_index(drop=True)

    print("\n==============================================================")
    print(" FULL FAIRNESS AUDIT MATRIX (ALL ATTRIBUTES)")
    print("==============================================================")
    print(df.to_string(index=False))
    print("==============================================================")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = get_checkpoint_path(args)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    attrs = detect_all_attributes()
    print(f"Detected {len(attrs)} sensitive attributes")

    # temporarily override so dataloader returns all attrs
    Config.SENSITIVE_ATTRS = attrs

    print("Initializing DataLoaders...")
    _, val_loader, _ = get_celeba_dataloaders()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = load_model(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    print("Loaded checkpoint:", ckpt_path)

    df = run_audit(
        model,
        val_loader,
        criterion,
        device,
        attrs
    )

    save_name = os.path.basename(
        ckpt_path
    ).replace(".pth", "_full_audit.csv")

    save_path = os.path.join(
        Config.CHECKPOINT_DIR,
        save_name
    )

    df.to_csv(save_path, index=False)

    print("\nSaved:", save_path)


if __name__ == "__main__":
    main()