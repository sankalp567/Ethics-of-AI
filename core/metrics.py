import torch
import numpy as np


def calculate_accuracy(outputs, labels):
    """
    Standard binary accuracy from logits.
    """
    preds = (outputs >= 0.0).squeeze().long()
    correct = (preds == labels).sum().item()
    return correct, len(labels)


# =====================================================
# BASIC METRICS
# =====================================================

def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def binary_confusion_metrics(preds, labels):
    """
    preds, labels : torch tensors of shape [N], values {0,1}

    Returns:
        TP TN FP FN
        acc precision recall specificity f1 balanced_acc
    """
    preds = preds.long()
    labels = labels.long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn

    acc = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)              # sensitivity / TPR
    specificity = _safe_div(tn, tn + fp)        # TNR
    f1 = _safe_div(2 * precision * recall, precision + recall)
    balanced_acc = (recall + specificity) / 2.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,

        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": balanced_acc
    }


# =====================================================
# FULL FAIRNESS METRICS
# =====================================================

def calculate_full_fairness_metrics(preds, labels, sensitive_attrs):
    """
    preds           tensor [N] binary {0,1}
    labels          tensor [N] binary {0,1}
    sensitive_attrs tensor [N] binary {0,1}

    Group1 = sensitive=1
    Group0 = sensitive=0

    Returns comprehensive metrics dict.
    """

    preds = preds.long()
    labels = labels.long()
    sensitive_attrs = sensitive_attrs.long()

    # --------------------------
    # overall metrics
    # --------------------------
    overall = binary_confusion_metrics(preds, labels)

    # --------------------------
    # subgroup masks
    # --------------------------
    g1_mask = sensitive_attrs == 1
    g0_mask = sensitive_attrs == 0

    preds_g1 = preds[g1_mask]
    labels_g1 = labels[g1_mask]

    preds_g0 = preds[g0_mask]
    labels_g0 = labels[g0_mask]

    group1 = binary_confusion_metrics(preds_g1, labels_g1)
    group0 = binary_confusion_metrics(preds_g0, labels_g0)

    # --------------------------
    # gaps
    # --------------------------
    acc_gap = abs(group1["acc"] - group0["acc"])
    f1_gap = abs(group1["f1"] - group0["f1"])
    recall_gap = abs(group1["recall"] - group0["recall"])          # TPR gap
    specificity_gap = abs(group1["specificity"] - group0["specificity"])  # TNR gap
    bal_acc_gap = abs(group1["balanced_acc"] - group0["balanced_acc"])

    # Equalized Odds style gap
    # = max(TPR gap, FPR gap)
    fpr_g1 = 1.0 - group1["specificity"]
    fpr_g0 = 1.0 - group0["specificity"]

    fpr_gap = abs(fpr_g1 - fpr_g0)
    eo_gap = max(recall_gap, fpr_gap)

    # worst-group
    worst_group_acc = min(group1["acc"], group0["acc"])
    worst_group_f1 = min(group1["f1"], group0["f1"])
    worst_group_bal_acc = min(group1["balanced_acc"], group0["balanced_acc"])

    # macro subgroup metrics
    macro_group_acc = (group1["acc"] + group0["acc"]) / 2.0
    macro_group_f1 = (group1["f1"] + group0["f1"]) / 2.0

    metrics = {
        # ---------------- overall ----------------
        "overall_total": overall["total"],
        "overall_acc": overall["acc"],
        "overall_precision": overall["precision"],
        "overall_recall": overall["recall"],
        "overall_specificity": overall["specificity"],
        "overall_f1": overall["f1"],
        "overall_balanced_acc": overall["balanced_acc"],

        # ---------------- group sizes ----------------
        "group1_total": group1["total"],
        "group0_total": group0["total"],

        # ---------------- group1 ----------------
        "group1_acc": group1["acc"],
        "group1_f1": group1["f1"],
        "group1_precision": group1["precision"],
        "group1_recall": group1["recall"],
        "group1_specificity": group1["specificity"],
        "group1_balanced_acc": group1["balanced_acc"],

        # ---------------- group0 ----------------
        "group0_acc": group0["acc"],
        "group0_f1": group0["f1"],
        "group0_precision": group0["precision"],
        "group0_recall": group0["recall"],
        "group0_specificity": group0["specificity"],
        "group0_balanced_acc": group0["balanced_acc"],

        # ---------------- fairness gaps ----------------
        "acc_gap": acc_gap,
        "f1_gap": f1_gap,
        "tpr_gap": recall_gap,
        "tnr_gap": specificity_gap,
        "fpr_gap": fpr_gap,
        "balanced_acc_gap": bal_acc_gap,
        "equalized_odds_gap": eo_gap,

        # ---------------- worst-group ----------------
        "worst_group_acc": worst_group_acc,
        "worst_group_f1": worst_group_f1,
        "worst_group_balanced_acc": worst_group_bal_acc,

        # ---------------- macro subgroup ----------------
        "macro_group_acc": macro_group_acc,
        "macro_group_f1": macro_group_f1
    }

    return metrics