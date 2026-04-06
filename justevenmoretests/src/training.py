import torch
import torch.nn as nn
import numpy as np
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.architectures import MLP, ResNetTabular, FTTransformer
from src.metrics import PyTorchTracker, TabNetTracker, BoostingTracker
from src.utils import CheckpointManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def _compute_val_loss(model, X_val, y_val, criterion, batch_size=1024):
    """Returns: (weighted_loss, unweighted_loss)."""
    model.eval()
    dev = next(model.parameters()).device
    total_w, total_uw, n = 0.0, 0.0, 0
    uw_fn = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            xb = torch.tensor(X_val[i:i + batch_size], dtype=torch.float32).to(dev)
            yb = torch.tensor(y_val[i:i + batch_size], dtype=torch.long).to(dev)
            logits = model(xb)
            bs = len(xb)
            total_w += criterion(logits, yb).item() * bs
            total_uw += uw_fn(logits, yb).item() * bs
            n += bs
    return total_w / n, total_uw / n


# ============================================================================
# PyTorch HPO (einmalig pro Modell × Dataset)
# ============================================================================

def run_pytorch_hpo(model_type, data_splits, n_trials, hpo_seed):
    """Optuna HPO. Gibt best_params als dict zurück."""
    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    input_dim = X_train.shape[1]

    # FIX: Klassengewichte bei Fraud kappen, um explodierende Gradienten zu vermeiden
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = np.clip(cw, a_min=None, a_max=50.0) 
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
# --- NEU: Subsampling für extrem große Datensätze in der HPO ---
    MAX_HPO_SAMPLES = 50000
    if len(X_train) > MAX_HPO_SAMPLES:
        print(f"  Reduziere Trainingsdaten für HPO von {len(X_train)} auf {MAX_HPO_SAMPLES} Samples...")
        X_hpo_base, _, y_hpo_base, _ = train_test_split(
            X_train, y_train, train_size=MAX_HPO_SAMPLES, stratify=y_train, random_state=hpo_seed
        )
    else:
        X_hpo_base, y_hpo_base = X_train, y_train
    def objective(trial):
        if model_type == "mlp":
            params = {
                "n_layers":     trial.suggest_int("n_layers", 2, 6),
                "hidden_dim":   trial.suggest_int("hidden_dim", 64, 512, step=64),
                "lr":           trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "batch_size":   trial.suggest_categorical("batch_size", [256, 512, 1024]),
            }
            mdl = MLP(input_dim, params["hidden_dim"], params["n_layers"],
                       params["dropout"]).to(device)
        elif model_type == "resnet":
            params = {
                "n_layers":     trial.suggest_int("n_layers", 2, 10),
                "d_model":      trial.suggest_categorical("d_model", [128, 256, 512]),
                "lr":           trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "batch_size":   trial.suggest_categorical("batch_size", [256, 512, 1024]),
            }
            mdl = ResNetTabular(input_dim, 2, params["n_layers"],
                                 params["d_model"], params["dropout"]).to(device)
        elif model_type == "ftt":
            params = {
                "n_layers":     trial.suggest_int("n_layers", 2, 6),
                "d_token":      trial.suggest_categorical("d_token", [64, 128, 192, 256]),
                "lr":           trial.suggest_float("lr", 5e-5, 5e-3, log=True),
                "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "batch_size":   trial.suggest_categorical("batch_size", [1024, 2048]),
            }
            mdl = FTTransformer(input_dim, params["d_token"], params["n_layers"],
                                 dropout=params["dropout"]).to(device)

        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=hpo_seed
        )
        ds = TensorDataset(torch.tensor(X_t, dtype=torch.float32),
                           torch.tensor(y_t, dtype=torch.long))
        dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=True)

        opt = torch.optim.AdamW(mdl.parameters(), lr=params["lr"],
                                weight_decay=params.get("weight_decay", 1e-2))
        crit = nn.CrossEntropyLoss(weight=cw_tensor)

        mdl.train()
        for epoch in range(50):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(mdl(xb), yb).backward()
                opt.step()
            mdl.eval()
            with torch.no_grad():
                val_metric = average_precision_score(y_v, mdl.predict_proba(X_v)[:, 1])
            mdl.train()
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return val_metric

    if n_trials > 0:
        print(f"  Starte {model_type.upper()} HPO ({n_trials} Trials)...")
        sampler = TPESampler(seed=hpo_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial.params
        print(f"  Beste Params: {best}")
        print(f"  Bester AUPRC: {study.best_value:.4f}")
    else:
        best = {"hidden_dim": 256, "n_layers": 3, "lr": 1e-3, "dropout": 0.2,
                "batch_size": 1024, "d_model": 128, "d_token": 128, "weight_decay": 1e-2}
    return best


# ============================================================================
# PyTorch Final Training (pro Seed)
# ============================================================================

def run_pytorch_final(model_type, data_splits, best_params, epochs, dataset_name, seed):
    """Finales Training mit fixen Hyperparametern. Returns: (model, tracker)."""
    print(f"  Training {model_type.upper()} (Seed {seed}, {epochs} Epochen)...")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    X_cal, y_cal     = data_splits['X_cal'],   data_splits['y_cal']
    input_dim = X_train.shape[1]
    best = best_params

    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = np.clip(cw, a_min=None, a_max=50.0)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    if model_type == "mlp":
        model = MLP(input_dim, best["hidden_dim"], best["n_layers"],
                     best["dropout"]).to(device)
    elif model_type == "resnet":
        model = ResNetTabular(input_dim, 2, best["n_layers"],
                               best.get("d_model", 128), best["dropout"]).to(device)
    elif model_type == "ftt":
        model = FTTransformer(input_dim, best.get("d_token", 128),
                               best["n_layers"],
                               dropout=best.get("dropout", 0.1)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best["lr"],
                                  weight_decay=best.get("weight_decay", 1e-2))
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=best["batch_size"], shuffle=True)

    tracker = PyTorchTracker(X_train, y_train, X_cal, y_cal, device)
    ckpt = CheckpointManager(f"{model_type}_seed{seed}", dataset_name)

    best_val_auprc = -float('inf')
    patience_counter = 0
    MAX_PATIENCE = 20
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        train_loss = np.mean(losses)
        _, val_loss_uw = _compute_val_loss(model, X_cal, y_cal, criterion)

        model.eval()
        with torch.no_grad():
            val_probs = model.predict_proba(X_cal)[:, 1]
            val_auprc = average_precision_score(y_cal, val_probs)

        scheduler.step(val_auprc)
        tracker.on_epoch_end(model, epoch, train_loss, val_loss_uw)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            ckpt.save(model, optimizer, epoch, val_auprc, is_best=True)
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Ep {epoch + 1:03d} | Train {train_loss:.4f} | Val AUPRC {val_auprc:.4f} "
              f"| LR {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= MAX_PATIENCE:
            print(f"  Early Stopping bei Epoche {epoch + 1}")
            break

    model = ckpt.load_best(model)
    return model, tracker


# ============================================================================
# TabNet HPO
# ============================================================================

class TabNetOptunaCallback(Callback):
    """Custom Callback für Optuna-Pruning in TabNet."""
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        # Logloss des Validierungssets auslesen (TabNet nutzt den Präfix val_0_)
        val_loss = logs.get("val_0_logloss")
        if val_loss is not None:
            self.trial.report(val_loss, epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()


def run_tabnet_hpo(data_splits, n_trials, hpo_seed):
    """Optuna HPO für TabNet. Gibt best_params als dict zurück."""
    X_train, y_train = data_splits['X_train'], data_splits['y_train']

    def objective(trial):
        params = {
            "n_d":             trial.suggest_int("n_d", 8, 64, step=8),
            "n_a":             trial.suggest_int("n_a", 8, 64, step=8),
            "n_steps":         trial.suggest_int("n_steps", 3, 10),
            "gamma":           trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse":   trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "lr":              trial.suggest_float("lr", 1e-3, 2e-2, log=True),
            "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "mask_type":       trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
            "batch_size":      trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
            "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [128, 256]),
        }

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=hpo_seed
        )

        model_params = {k: v for k, v in params.items()
                        if k not in ("lr", "weight_decay", "batch_size", "virtual_batch_size")}
        model_params["optimizer_params"] = dict(lr=params["lr"], weight_decay=params["weight_decay"])

        clf = TabNetClassifier(**model_params, verbose=0, seed=hpo_seed)
        pruning_callback = TabNetOptunaCallback(trial)
        
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=['logloss'],
                max_epochs=50, patience=10,
                batch_size=params["batch_size"],
                virtual_batch_size=params["virtual_batch_size"], weights=1,
                callbacks=[pruning_callback])
        preds = clf.predict_proba(X_val)[:, 1]
        try:
            return average_precision_score(y_val, preds)
        except ValueError:
            return 0.0

    if n_trials > 0:
        print(f"  Starte TabNet HPO ({n_trials} Trials)...")
        sampler = TPESampler(seed=hpo_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial.params
        print(f"  Beste Params: {best}")
        print(f"  Bester AUPRC: {study.best_value:.4f}")
    else:
        best = {"n_d": 64, "n_a": 64, "n_steps": 5, "gamma": 1.5,
                "lambda_sparse": 1e-4, "lr": 2e-2, "weight_decay": 1e-5,
                "mask_type": "sparsemax", "batch_size": 1024, "virtual_batch_size": 256}
    return best


# ============================================================================
# TabNet Final Training
# ============================================================================

def run_tabnet_final(data_splits, best_params, epochs, dataset_name, seed):
    """Finales TabNet Training. Returns: (model, tracker)."""
    print(f"  Training TABNET (Seed {seed}, {epochs} Epochen)...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    X_cal, y_cal     = data_splits['X_cal'],   data_splits['y_cal']

    bp = dict(best_params)
    batch_size = bp.pop("batch_size", 1024)
    virtual_batch_size = bp.pop("virtual_batch_size", 256)
    lr = bp.pop("lr", 2e-2)
    wd = bp.pop("weight_decay", 1e-5)
    model_params = {k: v for k, v in bp.items()}
    model_params["optimizer_params"] = dict(lr=lr, weight_decay=wd)

    tracker = TabNetTracker(X_train, y_train, X_cal, y_cal)
    model = TabNetClassifier(**model_params, verbose=1, seed=seed)
    model.fit(
        X_train, y_train,
        eval_set=[(X_cal, y_cal)], eval_metric=['logloss'],
        max_epochs=epochs, patience=15,
        batch_size=int(batch_size), virtual_batch_size=int(virtual_batch_size),
        callbacks=[tracker], weights=1,
    )
    return model, tracker


# ============================================================================
# Tree HPO
# ============================================================================

def run_tree_hpo(model_type, data_splits, n_trials, hpo_seed):
    """Optuna HPO für Tree-Modelle. Gibt best_params als dict zurück."""
    if model_type == "xgboost":
        from xgboost import XGBClassifier
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier
    elif model_type == "catboost":
        from catboost import CatBoostClassifier

    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    def objective(trial):
        X_hpo_tr, X_hpo_val, y_hpo_tr, y_hpo_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=hpo_seed
        )

        if model_type == "xgboost":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 2000),
                "max_depth":        trial.suggest_int("max_depth", 3, 12),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            }
            # Early Stopping auf 150 erhöht
            clf = XGBClassifier(**params, scale_pos_weight=scale_pos, eval_metric="logloss",
                                early_stopping_rounds=150, random_state=hpo_seed,
                                n_jobs=-1, tree_method="hist")
            clf.fit(X_hpo_tr, y_hpo_tr, eval_set=[(X_hpo_val, y_hpo_val)], verbose=False)

        elif model_type == "lgbm":
            import lightgbm as lgb # Wichtig für das Early Stopping Callback
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 2000),
                # max_depth kann nun -1 sein (unbegrenzt) oder höhere Werte annehmen
                "max_depth":        trial.suggest_categorical("max_depth", [-1, 5, 7, 9, 12]),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 15, 255),
            }
            clf = LGBMClassifier(**params, is_unbalance=True, metric="binary_logloss",
                                 random_state=hpo_seed, n_jobs=-1, verbosity=-1)
            # Hier hat das Early Stopping (150) in der HPO gefehlt!
            clf.fit(X_hpo_tr, y_hpo_tr, eval_set=[(X_hpo_val, y_hpo_val)], 
                    callbacks=[lgb.early_stopping(150, verbose=False)])

        elif model_type == "catboost":
            params = {
                "iterations":       trial.suggest_int("iterations", 100, 2000),
                "depth":            trial.suggest_int("depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.3, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            }
            clf = CatBoostClassifier(**params, auto_class_weights="Balanced",
                                     eval_metric="Logloss", random_seed=hpo_seed, verbose=0)
            # Early Stopping auf 150 erhöht
            clf.fit(X_hpo_tr, y_hpo_tr, eval_set=(X_hpo_val, y_hpo_val),
                    early_stopping_rounds=150, verbose=0)

        preds = clf.predict_proba(X_hpo_val)[:, 1]
        try:
            return average_precision_score(y_hpo_val, preds)
        except ValueError:
            return 0.0

    if n_trials > 0:
        print(f"  Starte {model_type.upper()} HPO ({n_trials} Trials)...")
        sampler = TPESampler(seed=hpo_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial.params
        print(f"  Beste Params: {best}")
        print(f"  Bester AUPRC: {study.best_value:.4f}")
    else:
        if model_type == "xgboost":
            best = {
                "n_estimators": 848, 
                "max_depth": 11, 
                "learning_rate": 0.08258782132344537,
                "subsample": 0.9060075624343468, 
                "colsample_bytree": 0.8871697198303783, 
                "min_child_weight": 5,
                "reg_alpha": 0.00093888717389766, 
                "reg_lambda": 9.90538596164951e-08, 
                "gamma": 0.0034761212866313007
            }
        elif model_type == "lgbm":
            best = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
                     "reg_alpha": 0.1, "reg_lambda": 1.0, "num_leaves": 63}
        elif model_type == "catboost":
            best = {"iterations": 500, "depth": 6, "learning_rate": 0.05,
                     "l2_leaf_reg": 3.0, "subsample": 0.8, "colsample_bylevel": 0.8,
                     "min_data_in_leaf": 10}
    return best


# ============================================================================
# Tree Final Training
# ============================================================================

def run_tree_final(model_type, data_splits, best_params, dataset_name, seed):
    """Finales Tree-Training + BoostingTracker. Returns: (model, tracker)."""
    if model_type == "xgboost":
        from xgboost import XGBClassifier
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier
        import lightgbm as lgb
    elif model_type == "catboost":
        from catboost import CatBoostClassifier

    print(f"  Training {model_type.upper()} (Seed {seed})...")
    np.random.seed(seed)

    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    X_cal, y_cal     = data_splits['X_cal'],   data_splits['y_cal']
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    if model_type == "xgboost":
        model = XGBClassifier(**best_params, scale_pos_weight=scale_pos,
                              eval_metric="logloss", early_stopping_rounds=150, # Geändert
                              random_state=seed, n_jobs=-1, tree_method="hist")
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)

    elif model_type == "lgbm":
        model = LGBMClassifier(**best_params, is_unbalance=True, metric="binary_logloss",
                               random_state=seed, n_jobs=-1, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)],
                  callbacks=[lgb.early_stopping(150, verbose=False)]) # Geändert

    elif model_type == "catboost":
        model = CatBoostClassifier(**best_params, auto_class_weights="Balanced",
                                   eval_metric="Logloss", random_seed=seed, verbose=0)
        model.fit(X_train, y_train, eval_set=(X_cal, y_cal),
                  early_stopping_rounds=150, verbose=0) # Geändert

    cal_auprc = average_precision_score(y_cal, model.predict_proba(X_cal)[:, 1])
    print(f"  Cal AUPRC: {cal_auprc:.4f}")

    # --- FIX: Speicherschonendes Subsampling NUR für den Tracker ---
    print("  Bereite speicherfreundliches Tracking vor...")
    mask_1 = (y_train == 1)
    mask_0 = (y_train == 0)
    
    idx_1 = np.where(mask_1)[0]
    # Max. 10.000 Klasse-0-Samples (oder alle, falls es weniger gibt)
    idx_0 = np.random.choice(np.where(mask_0)[0], size=min(10000, sum(mask_0)), replace=False)
    
    tracker_idx = np.concatenate([idx_1, idx_0])
    np.random.shuffle(tracker_idx) # Zur Sicherheit mischen
    
    X_track_train = X_train[tracker_idx]
    y_track_train = y_train[tracker_idx]
    # -------------------------------------------------------------

    # Tracker mit den verkleinerten Daten füttern
    tracker = BoostingTracker(X_track_train, y_track_train, X_cal, y_cal)
    
    # Reduziere ggf. auch die Checkpoints leicht, um noch mehr RAM zu sparen
    tracker.compute_from_model(model, model_type, n_checkpoints=50, pca_dim=32)

    return model, tracker
