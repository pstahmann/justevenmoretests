import os
import random
import numpy as np
import pandas as pd
import torch
from src import config


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # FIX #4: Auch den Metrik-RNG zurücksetzen
    from src.metrics import reset_metric_rng
    reset_metric_rng(seed)


class CheckpointManager:
    def __init__(self, model_name, dataset_name):
        self.save_dir = os.path.join(config.CHECKPOINT_DIR, model_name, dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_score = -float('inf')

    def save(self, model, optimizer, epoch, score, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'score': score,
        }
        torch.save(state, os.path.join(self.save_dir, "last.pth"))
        if is_best:
            self.best_score = score
            torch.save(state, os.path.join(self.save_dir, "best.pth"))

    def load_best(self, model):
        best_path = os.path.join(self.save_dir, "best.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"  [Checkpoint] Geladen: Score {checkpoint['score']:.4f}, "
                  f"Epoch {checkpoint['epoch']}")
        else:
            print(f"  [Checkpoint] WARNUNG: Kein best.pth unter {best_path}. "
                  f"Nutze aktuellen Zustand.")
        return model


def save_results(results_dict, dataset_name, model_name, seed=42):
    """Seed im Dateinamen für Multi-Seed-Aggregation."""
    prefix = f"{model_name}_{dataset_name}_seed{seed}"

    met_path = os.path.join(config.RESULTS_DIR, f"metrics_{prefix}.csv")
    results_dict["metrics"].to_csv(
        met_path, sep=";", index=False, decimal=",", encoding="utf-8-sig",
    )

    pred_path = os.path.join(config.RESULTS_DIR, f"predictions_{prefix}.csv")
    results_dict["predictions"].to_csv(
        pred_path, sep=";", index=False, decimal=",", encoding="utf-8-sig",
    )

    # Imbalance-Ratio aus Predictions (y_true) berechnen
    y_true = results_dict["predictions"]["y_true"]
    n0 = int((y_true == 0).sum())
    n1 = int((y_true == 1).sum())
    ir = n0 / max(n1, 1)

    summary_path = os.path.join(config.RESULTS_DIR, f"summary_{prefix}.csv")
    summary_row = {
        "Dataset": dataset_name, "Model": model_name, "Seed": seed,
        "N_test": len(y_true), "IR_test": round(ir, 1),
    }
    summary_row.update(results_dict["classification_metrics"])
    pd.DataFrame([summary_row]).to_csv(
        summary_path, sep=";", index=False, decimal=",", encoding="utf-8-sig",
    )

    print(f"  Gespeichert in {config.RESULTS_DIR} (Seed {seed})")
