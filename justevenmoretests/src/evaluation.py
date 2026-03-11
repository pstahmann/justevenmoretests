import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    recall_score, precision_score, balanced_accuracy_score,
    matthews_corrcoef, f1_score, fbeta_score, precision_recall_curve,
)
from sklearn.isotonic import IsotonicRegression
import torch
from src import plotting


def expected_calibration_error(y_true, probs, n_bins=15):
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) == 0:
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(probs[mask])
        ece += np.abs(acc - conf) * (np.sum(mask) / len(y_true))
    return ece


def evaluate_model(model, data_splits, tracker, model_type, dataset_name, seed=42):
    print("Starte Evaluation...")
    X_test, y_test = data_splits['X_test'], data_splits['y_test']
    X_cal, y_cal   = data_splits['X_cal'],  data_splits['y_cal']

    # --- Vorhersagen (einheitlich batched) ---
    def get_probs(model, X):
        # Alle Modelle (DL, TabNet, Tree) haben predict_proba
        return model.predict_proba(X)[:, 1]

    probs_raw = get_probs(model, X_test)
    probs_val_raw = get_probs(model, X_cal)

    # --- Isotonische Kalibrierung ---
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs_val_raw, y_cal)
    probs_val_cal = iso.predict(probs_val_raw)
    probs_cal = iso.predict(probs_raw)

    # --- Threshold-Optimierung (F2-Score auf Cal-Set) ---
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_cal, probs_val_cal)
    beta = 2
    f2_scores = (
        (1 + beta ** 2) * (precision_vals[:-1] * recall_vals[:-1])
        / ((beta ** 2 * precision_vals[:-1]) + recall_vals[:-1] + 1e-12)
    )
    best_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[best_idx]
    y_pred = (probs_cal >= optimal_threshold).astype(int)

    print(f"  Optimaler Threshold (F2): {optimal_threshold:.4f}")

    # --- Klassifikationsmetriken ---
    classification_metrics = {
        "AUC":       roc_auc_score(y_test, probs_cal),
        "AUPRC":     average_precision_score(y_test, probs_cal),
        "Brier":     brier_score_loss(y_test, probs_cal),
        "Recall":    recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "BAC":       balanced_accuracy_score(y_test, y_pred),
        "MCC":       matthews_corrcoef(y_test, y_pred),
        "F1_micro":  f1_score(y_test, y_pred, average="micro"),
        "F1_macro":  f1_score(y_test, y_pred, average="macro"),
        "F2":        fbeta_score(y_test, y_pred, beta=2),
        "ECE_raw":   expected_calibration_error(y_test, probs_raw),
        "ECE_cal":   expected_calibration_error(y_test, probs_cal),
    }

    print(f"  AUC:    {classification_metrics['AUC']:.4f}")
    print(f"  AUPRC:  {classification_metrics['AUPRC']:.4f}")
    print(f"  BAC:    {classification_metrics['BAC']:.4f}")
    print(f"  F2:     {classification_metrics['F2']:.4f}")
    print(f"  Recall: {classification_metrics['Recall']:.4f}")
    print(f"  ECE (raw):        {classification_metrics['ECE_raw']:.4f}")
    print(f"  ECE (kalibriert): {classification_metrics['ECE_cal']:.4f}")

    # --- History → Long Format (DL-Modelle + Boosting-Tracker) ---
    records = []
    if tracker is not None and hasattr(tracker, 'history'):
        for m in tracker.history:
            epoch = m["epoch"]
            loss_class_0 = m.get("loss_class_0", np.nan)
            loss_class_1 = m.get("loss_class_1", np.nan)
            val_loss = m.get("val_loss", np.nan)

            for class_id in [0, 1]:
                for name, value in m["per_class"][class_id].items():
                    key = f"per_class_{class_id}/{name}"
                    avg_change = m["avg_abs_change"].get(key, np.nan)
                    records.append({
                        "model_type": model_type,
                        "epoch": epoch, "metric": name, "value": value,
                        "avg_abs_change": avg_change,
                        "category": f"per_class_{class_id}",
                        "loss_class_0": loss_class_0,
                        "loss_class_1": loss_class_1,
                        "val_loss": val_loss,
                    })

            for name, value in m["between_classes"].items():
                key = f"between_classes/{name}"
                avg_change = m["avg_abs_change"].get(key, np.nan)
                records.append({
                    "model_type": model_type,
                    "epoch": epoch, "metric": name, "value": value,
                    "avg_abs_change": avg_change,
                    "category": "between_classes",
                    "loss_class_0": loss_class_0,
                    "loss_class_1": loss_class_1,
                    "val_loss": val_loss,
                })

            for name, value in m["global"].items():
                key = f"global/{name}"
                avg_change = m["avg_abs_change"].get(key, np.nan)
                records.append({
                    "model_type": model_type,
                    "epoch": epoch, "metric": name, "value": value,
                    "avg_abs_change": avg_change,
                    "category": "global",
                    "loss_class_0": loss_class_0,
                    "loss_class_1": loss_class_1,
                    "val_loss": val_loss,
                })

    df_metrics_long = pd.DataFrame(records)
    if not df_metrics_long.empty:
        df_metrics_long = df_metrics_long.sort_values(by=["category", "metric", "epoch"])

    # --- Plots ---
    if tracker is not None and len(tracker.history) > 0:
        plotting.save_training_plots(tracker, model_type, dataset_name, seed=seed)
    plotting.save_evaluation_plots(y_test, probs_cal, model_type, dataset_name,
                                   y_pred=y_pred, seed=seed)

    return {
        "metrics": df_metrics_long,
        "predictions": pd.DataFrame({
            "y_true": y_test,
            "y_prob": probs_cal,
            "y_pred": y_pred,
        }),
        "classification_metrics": classification_metrics,
        "ece_raw": classification_metrics["ECE_raw"],
        "ece_calibrated": classification_metrics["ECE_cal"],
    }
