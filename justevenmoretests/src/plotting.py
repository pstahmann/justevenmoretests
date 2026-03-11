import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from src import config


def save_training_plots(tracker, model_name, dataset_name, seed=42):
    """Plottet Train/Val Loss und eine Beispiel-Metrik über Epochen."""
    save_path = os.path.join(
        config.RESULTS_DIR,
        f"plot_history_{model_name}_{dataset_name}_seed{seed}.png",
    )

    if not tracker.history:
        return

    epochs = [x['epoch'] for x in tracker.history]
    train_losses = [x['loss'] for x in tracker.history]
    val_losses = [x.get('val_loss', np.nan) for x in tracker.history]
    bhatt = [x['between_classes'].get('bhattacharyya', np.nan) for x in tracker.history]
    fisher = [x['between_classes'].get('fisher_ratio', np.nan) for x in tracker.history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Train + Val Loss
    axes[0].plot(epochs, train_losses, marker='.', markersize=3, label='Train Loss')
    if not all(np.isnan(v) for v in val_losses):
        axes[0].plot(epochs, val_losses, marker='.', markersize=3, label='Val Loss')
    axes[0].set_title(f'{model_name.upper()} Loss (Seed {seed})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Bhattacharyya
    axes[1].plot(epochs, bhatt, marker='.', markersize=3, color='orange')
    axes[1].set_title('Bhattacharyya Distance')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Fisher Ratio
    axes[2].plot(epochs, fisher, marker='.', markersize=3, color='green')
    axes[2].set_title('Fisher Discriminant Ratio')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot gespeichert: {save_path}")


def save_evaluation_plots(y_true, y_prob, model_name, dataset_name,
                          y_pred=None, seed=42):
    """ROC, PRC und Confusion Matrix."""
    prefix = f"{model_name}_{dataset_name}_seed{seed}"

    # 1. ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'{model_name}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f'ROC Curve — {dataset_name} (Seed {seed})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(config.RESULTS_DIR, f"plot_roc_{prefix}.png"), dpi=150)
    plt.close()

    # 2. Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(rec, prec, color='purple')
    plt.title(f'PR Curve — {dataset_name} (Seed {seed})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(config.RESULTS_DIR, f"plot_prc_{prefix}.png"), dpi=150)
    plt.close()

    # 3. Confusion Matrix
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {dataset_name} (Seed {seed})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(config.RESULTS_DIR, f"plot_cm_{prefix}.png"), dpi=150)
    plt.close()

    print(f"  Evaluations-Plots gespeichert in {config.RESULTS_DIR}")
