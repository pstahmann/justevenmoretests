import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cityblock, cosine, mahalanobis
from scipy.linalg import pinv, det, eigvalsh, inv, eigh
from scipy.stats import wasserstein_distance
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from pytorch_tabnet.callbacks import Callback
from src.architectures import FTTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold

EPS = 1e-9

# ============================================================================
# 1. MATHEMATISCHE HILFSFUNKTIONEN
# ============================================================================

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def manhattan_distance(a, b):
    return cityblock(a, b)

def cosine_distance(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < EPS or norm_b < EPS: return 0.0
    return cosine(a, b)

def population_vector_angle(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < EPS or norm_b < EPS: return 0.0
    a_n = a / norm_a
    b_n = b / norm_b
    return np.arccos(np.clip(np.dot(a_n, b_n), -1.0, 1.0))

def mahalanobis_distance(a, b, cov0, cov1):
    reg = 1e-6 * np.eye(cov0.shape[0])
    cov = (cov0 + cov1) / 2 + reg
    try:
        inv_cov = pinv(cov)
        return mahalanobis(a, b, inv_cov)
    except:
        return np.nan

def bhattacharyya_distance(a, b, cov0, cov1):
    reg = 1e-6 * np.eye(cov0.shape[0])
    cov = (cov0 + cov1) / 2 + reg
    cov0_reg = cov0 + reg
    cov1_reg = cov1 + reg
    diff = b - a
    try:
        inv_cov = pinv(cov)
        term1 = 0.125 * diff.T @ inv_cov @ diff
        sign_c, logdet_c = np.linalg.slogdet(cov)
        sign_0, logdet_0 = np.linalg.slogdet(cov0_reg)
        sign_1, logdet_1 = np.linalg.slogdet(cov1_reg)
        if sign_c <= 0 or sign_0 <= 0 or sign_1 <= 0: return np.nan
        term2 = 0.5 * (logdet_c - 0.5 * logdet_0 - 0.5 * logdet_1)
        return term1 + term2
    except:
        return np.nan

def hellinger_distance(bhat):
    if np.isnan(bhat) or bhat < 0: return np.nan
    return np.sqrt(1 - np.exp(-bhat))

def wasserstein_dist(A, B):
    if A.shape[1] == 0 or B.shape[1] == 0: return np.nan
    if len(A) > 2000: A = A[np.random.choice(len(A), 2000, replace=False)]
    if len(B) > 2000: B = B[np.random.choice(len(B), 2000, replace=False)]
    return np.mean([
        wasserstein_distance(A[:, i], B[:, i]) for i in range(min(A.shape[1], B.shape[1]))
    ])

def within_class_variance(X):
    if len(X) < 2: return 0.0
    return np.trace(np.cov(X, rowvar=False))

def between_class_variance(mu0, mu1):
    return np.linalg.norm(mu0 - mu1)**2

def within_class_scatter(cov0, cov1):
    return cov0 + cov1

def between_class_scatter(mu0, mu1, mu):
    return np.outer(mu0 - mu, mu0 - mu) + np.outer(mu1 - mu, mu1 - mu)

def global_mean(mu0, mu1, n0, n1):
    return (n0 * mu0 + n1 * mu1) / (n0 + n1)

def fisher_discriminant_ratio(Sb, Sw):
    tr_Sw = np.trace(Sw)
    if tr_Sw < EPS: return 0.0
    return np.trace(Sb) / tr_Sw

def cov_effective_rank(cov, eps=1e-12):
    try:
        eigvals = eigvalsh(cov)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0: return 0.0
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        return np.exp(entropy)
    except: return 1.0

def participation_ratio(cov, eps=1e-12):
    try:
        eigvals = eigvalsh(cov)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0: return 0.0
        denom = np.sum(eigvals ** 2)
        if denom < eps: return 0.0
        return (eigvals.sum() ** 2) / denom
    except: return 1.0

def mixed_selectivity_index(X0, X1):
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    means = np.stack([mu0, mu1])
    signal_var = np.var(means, axis=0).mean()
    X_comb = np.vstack([X0, X1])
    total_var = np.var(X_comb, axis=0).mean()
    if total_var < EPS: return 0.0
    return 1.0 - (signal_var / total_var)

def ccgp_score(Z, y, cv=5):
    # 1. Zähle die Anzahl der Samples pro Klasse
    counts = np.bincount(y)
    if len(counts) < 2: 
        return 0.5
    min_class_count = counts.min()
    
    # 2. Failsafe: Mindestens 2 Samples in der Minderheitsklasse nötig für CV
    if min_class_count < 2: 
        return 0.5
    
    # 3. Dynamische Anpassung der Folds (verhindert Fold-Größen > Minderheitsklasse)
    actual_cv = min(cv, min_class_count)
    
    try:
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
        # 4. Explizites StratifiedKFold mit Shuffle
        skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, Z, y, cv=skf)
        return np.nanmean(scores)
    except: 
        return 0.5

def manifold_tangling_index(X0, X1, k=5):
    if len(X0) < k or len(X1) < k: return 0.0
    X = np.vstack([X0, X1])
    if len(X) > 2000:
        idx = np.random.choice(len(X), 2000, replace=False)
        X = X[idx]
        y = np.array([0]*len(X0) + [1]*len(X1))[idx]
    else:
        y = np.array([0]*len(X0) + [1]*len(X1))
    try:
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        neighbor_labels = y[indices[:,1:]]
        cross_class_frac = (neighbor_labels != y[:,None]).mean()
        return cross_class_frac
    except: return 0.0

def neural_subspace_overlap(X0, X1, n_components=5):
    n_comp0 = min(n_components, X0.shape[0], X0.shape[1])
    n_comp1 = min(n_components, X1.shape[0], X1.shape[1])
    if n_comp0 < 1 or n_comp1 < 1: return 0.0
    try:
        pca0 = PCA(n_components=n_comp0).fit(X0)
        pca1 = PCA(n_components=n_comp1).fit(X1)
        U0, U1 = pca0.components_.T, pca1.components_.T  # Shape: (n_features, n_comp)
        if U0.shape[0] != U1.shape[0]: return 0.0  # FIX: Feature-Dim prüfen, nicht n_components
        s = np.linalg.svd(U0.T @ U1, compute_uv=False)
        return np.sum(s**2) / max(n_comp0, n_comp1)
    except: return 0.0

def covariance_alignment(cov0, cov1):
    norm0, norm1 = np.linalg.norm(cov0), np.linalg.norm(cov1)
    if norm0 < EPS or norm1 < EPS: return 0.0
    return np.sum(cov0 * cov1) / (norm0 * norm1)

def temporal_decoding_accuracy(X0, X1, cv=5):
    # Identischer Failsafe wie oben
    min_class_count = min(len(X0), len(X1))
    if min_class_count < 2: 
        return 0.5
        
    actual_cv = min(cv, min_class_count)
    
    X = np.vstack([X0, X1])
    y = np.array([0]*len(X0) + [1]*len(X1))
    
    try:
        clf = LogisticRegression(max_iter=1000)
        skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf)
        return np.nanmean(scores)
    except: 
        return 0.5

def representational_drift(mu0, mu1, mu0_ref=None, mu1_ref=None):
    if mu0_ref is None or mu1_ref is None: return 0.0
    drift0 = np.linalg.norm(mu0 - mu0_ref)
    drift1 = np.linalg.norm(mu1 - mu1_ref)
    return (drift0 + drift1) / 2

def signal_correlation(X0, X1):
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    if np.std(mu0) < EPS or np.std(mu1) < EPS: return 0.0
    corr = np.corrcoef(mu0, mu1)[0,1]
    return 0.0 if np.isnan(corr) else corr

def noise_correlation(X0, X1):
    def corr_matrix(X):
        X_res = X - X.mean(axis=0)
        std = X_res.std(axis=0)
        valid = std > EPS
        if valid.sum() < 2: return 0.0
        C = np.corrcoef(X_res[:, valid], rowvar=False)
        off_diag = C[np.triu_indices_from(C, k=1)]
        return np.nanmean(off_diag) if len(off_diag) > 0 else 0.0
    return np.mean([corr_matrix(X0), corr_matrix(X1)])

def population_sparseness(X):
    r_mean = np.abs(X).mean(axis=0)
    N = len(r_mean)
    if N <= 1: return 0.0
    sq_sum = (r_mean**2).sum() / N
    sum_sq = (r_mean.sum() / N) ** 2
    if sq_sum < EPS: return 0.0
    return (1 - sum_sq / sq_sum) / (1 - 1/N)

def lifetime_sparseness(X):
    N = X.shape[0]
    if N < 2: return 0.0
    r_mean = np.abs(X).mean(axis=1)
    r_sq_mean = (X**2).mean(axis=1)
    mask = r_sq_mean > EPS
    if not mask.any(): return 0.0
    sparsity = (1 - (r_mean[mask]**2)/r_sq_mean[mask]) / (1 - 1/N)
    return np.mean(sparsity)

def effective_rank_score(cov, eps=1e-12):
    try:
        eigvals = eigh(cov, eigvals_only=True)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0: return 1.0
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p + eps))
        return np.exp(entropy)
    except np.linalg.LinAlgError:
        return 1.0

def spectral_entropy_score(cov, eps=1e-12):
    try:
        eigvals = eigh(cov, eigvals_only=True)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0: return 0.0
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p + eps))
        max_entropy = np.log(len(eigvals))
        if max_entropy == 0: return 0.0
        return entropy / max_entropy
    except:
        return 0.0

def log_determinant_entropy(cov, eps=1e-6):
    try:
        sign, logdet = np.linalg.slogdet(cov + np.eye(len(cov)) * eps)
        if sign <= 0: return -100.0
        return logdet
    except:
        return -100.0

def total_variance_trace(cov):
    return np.trace(cov)

def gaussian_kl_divergence(mu1, cov1, mu0, cov0, eps=1e-6):
    dim = len(mu1)
    cov0_reg = cov0 + np.eye(dim) * eps
    cov1_reg = cov1 + np.eye(dim) * eps
    try:
        cov0_inv = np.linalg.inv(cov0_reg)
        term_trace = np.trace(cov0_inv @ cov1_reg)
        diff = mu1 - mu0
        term_quad = diff.T @ cov0_inv @ diff
        _, logdet0 = np.linalg.slogdet(cov0_reg)
        _, logdet1 = np.linalg.slogdet(cov1_reg)
        kl = 0.5 * (term_trace + term_quad - dim + logdet0 - logdet1)
        return max(0.0, kl)
    except:
        return 10.0

# ============================================================================
# 2. METRIC REGISTRY (Vollständig)
# ============================================================================

METRIC_REGISTRY = {
    "euclidean": lambda c: euclidean_distance(c["mu0"], c["mu1"]),
    "manhattan": lambda c: manhattan_distance(c["mu0"], c["mu1"]),
    "cosine": lambda c: cosine_distance(c["mu0"], c["mu1"]),
    "population_vector_angle": lambda c: population_vector_angle(c["mu0"], c["mu1"]),
    "mahalanobis": lambda c: mahalanobis_distance(c["mu0"], c["mu1"], c["cov0"], c["cov1"]),
    "bhattacharyya": lambda c: bhattacharyya_distance(c["mu0"], c["mu1"], c["cov0"], c["cov1"]),
    "hellinger": lambda c: hellinger_distance(c["bhattacharyya"]),
    "wasserstein": lambda c: wasserstein_dist(c["X0"], c["X1"]),
    "within_var_0": lambda c: within_class_variance(c["X0"]),
    "within_var_1": lambda c: within_class_variance(c["X1"]),
    "between_class_variance": lambda c: between_class_variance(c["mu0"], c["mu1"]),
    "fisher_ratio": lambda c: fisher_discriminant_ratio(
        between_class_scatter(c["mu0"], c["mu1"], global_mean(c["mu0"], c["mu1"], len(c["X0"]), len(c["X1"]))),
        within_class_scatter(c["cov0"], c["cov1"])
    ),
    "cov_trace": lambda c: np.trace(c["cov_all"]),
    "cov_det": lambda c: det(c["cov_all"]),
    "cov_effective_rank": lambda c: cov_effective_rank(c["cov_all"]),
    "participation_ratio": lambda c: participation_ratio(c["cov_all"]),
    "mixed_selectivity_index": lambda c: mixed_selectivity_index(c["X0"], c["X1"]),
    "manifold_tangling": lambda c: manifold_tangling_index(c["X0"], c["X1"]),
    "subspace_overlap": lambda c: neural_subspace_overlap(c["X0"], c["X1"]),
    "cov_alignment": lambda c: covariance_alignment(c["cov0"], c["cov1"]),
    "ccgp": lambda c: ccgp_score(np.vstack([c["X0"], c["X1"]]), np.array([0]*len(c["X0"]) + [1]*len(c["X1"]))),
    "temporal_decoding_accuracy": lambda c: temporal_decoding_accuracy(c["X0"], c["X1"]),
    "representational_drift": lambda c: representational_drift(c["mu0"], c["mu1"], c.get("mu0_ref"), c.get("mu1_ref")),
    "signal_correlation": lambda c: signal_correlation(c["X0"], c["X1"]),
    "noise_correlation": lambda c: noise_correlation(c["X0"], c["X1"]),
    "population_sparseness": lambda c: np.mean([population_sparseness(c["X0"]), population_sparseness(c["X1"])]),
    "lifetime_sparseness": lambda c: np.mean([lifetime_sparseness(c["X0"]), lifetime_sparseness(c["X1"])]),
    "effective_rank_score": lambda c: effective_rank_score(c["cov_all"]),
    "spectral_entropy": lambda c: spectral_entropy_score(c["cov_all"]),
    "log_determinant": lambda c: log_determinant_entropy(c["cov_all"]),
    "total_variance": lambda c: total_variance_trace(c["cov_all"]),
}


# ============================================================================
# 3. COMPUTE & TRACKERS
# ============================================================================

def compute_metrics_structured(Z, y, ref_mu0=None, ref_mu1=None):
    X0 = Z[y == 0]
    X1 = Z[y == 1]
    n0, n1 = len(X0), len(X1)
    
    if n0 < 2 or n1 < 2: return None

    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    mu = Z.mean(axis=0)
    
    try:
        cov0 = np.cov(X0, rowvar=False)
        cov1 = np.cov(X1, rowvar=False)
        cov_all = np.cov(Z, rowvar=False)
    except:
        n_feat = Z.shape[1]
        cov0 = cov1 = cov_all = np.eye(n_feat)

    ctx = {
        "X0": X0, "X1": X1, "mu0": mu0, "mu1": mu1, "mu": mu,
        "cov0": cov0, "cov1": cov1, "cov_all": cov_all,
        "mu0_ref": ref_mu0, "mu1_ref": ref_mu1,
        "bhattacharyya": 0.0
    }

    flat_metrics = {}
    for name, fn in METRIC_REGISTRY.items():
        try:
            val = fn(ctx)
            if hasattr(val, 'item'): val = val.item()
            if not np.isfinite(val): val = 0.0
            flat_metrics[name] = val
            ctx[name] = val
        except:
            flat_metrics[name] = 0.0

    metrics = {
        "per_class": {
            0: {
                "within_variance": flat_metrics.get("within_var_0", 0),
                "population_sparseness": population_sparseness(X0),
                "lifetime_sparseness": lifetime_sparseness(X0),
                "cov_trace": np.trace(cov0),
                "cov_effective_rank": cov_effective_rank(cov0),
                "participation_ratio": participation_ratio(cov0),
            },
            1: {
                "within_variance": flat_metrics.get("within_var_1", 0),
                "population_sparseness": population_sparseness(X1),
                "lifetime_sparseness": lifetime_sparseness(X1),
                "cov_trace": np.trace(cov1),
                "cov_effective_rank": cov_effective_rank(cov1),
                "participation_ratio": participation_ratio(cov1),
            },
        },
        "between_classes": {k: flat_metrics.get(k, 0) for k in [
            "euclidean", "manhattan", "cosine", "population_vector_angle",
            "mahalanobis", "bhattacharyya", "hellinger", "wasserstein",
            "between_class_variance", "fisher_ratio"
        ]},
        "global": {k: flat_metrics.get(k, 0) for k in [
            "cov_trace", "cov_effective_rank", "participation_ratio",
            "mixed_selectivity_index", "manifold_tangling", "subspace_overlap",
            "cov_alignment", "ccgp", "temporal_decoding_accuracy",
            "representational_drift", "signal_correlation", "noise_correlation",
            "population_sparseness", "lifetime_sparseness"
        ]},
        "flat": flat_metrics,
        "means": {"mu0": mu0, "mu1": mu1, "mu": mu},
        "covs": {"cov_all": cov_all}
    }
    return metrics


class BaseTracker:
    def __init__(self):
        self.history = []
        self.ref_mu0 = None
        self.ref_mu1 = None
        self.last_state = None

        # --- PUNKT 2: avg_abs_change Tracking ---
        # Kumulativer gleitender Durchschnitt der absoluten Veränderungen pro Metrik
        self._prev_values = {}
        self._cum_abs_delta = {}
        self._delta_count = {}

    def _update_online_change(self, key, value):
        """Berechnet den kumulativen Durchschnitt der absoluten Veränderung für eine Metrik."""
        if key not in self._prev_values:
            self._prev_values[key] = value
            self._cum_abs_delta[key] = 0.0
            self._delta_count[key] = 0
            return 0.0
        delta = abs(value - self._prev_values[key])
        self._cum_abs_delta[key] += delta
        self._delta_count[key] += 1
        self._prev_values[key] = value
        return self._cum_abs_delta[key] / self._delta_count[key]

    def _process(self, metrics, epoch, loss, loss_class_0=np.nan, loss_class_1=np.nan):
        metrics["epoch"] = epoch + 1
        metrics["loss"] = loss

        # --- PUNKT 1: Klassenspezifischer Loss ---
        metrics["loss_class_0"] = loss_class_0
        metrics["loss_class_1"] = loss_class_1

        flat = metrics["flat"]

        # KL-Divergenz zwischen aufeinanderfolgenden Epochen
        current_mu = metrics["means"]["mu"]
        current_cov = metrics["covs"]["cov_all"]

        if self.last_state is not None:
            kl_val = gaussian_kl_divergence(
                current_mu, current_cov,
                self.last_state["mu"], self.last_state["cov"]
            )
            flat["kl_divergence_epoch"] = kl_val
        else:
            flat["kl_divergence_epoch"] = 0.0

        self.last_state = {"mu": current_mu, "cov": current_cov}

        # --- PUNKT 2: avg_abs_change pro Metrik (kumulativer Durchschnitt) ---
        avg_changes = {}
        for cls, metrics_cls in metrics["per_class"].items():
            for name, value in metrics_cls.items():
                key = f"per_class_{cls}/{name}"
                avg_changes[key] = self._update_online_change(key, value)

        for name, value in metrics["between_classes"].items():
            key = f"between_classes/{name}"
            avg_changes[key] = self._update_online_change(key, value)

        for name, value in metrics["global"].items():
            key = f"global/{name}"
            avg_changes[key] = self._update_online_change(key, value)

        metrics["avg_abs_change"] = avg_changes
        self.history.append(metrics)



class PyTorchTracker(BaseTracker):
    def __init__(self, X, y, device):
        super().__init__()
        self.device = device
        self.X_cpu = X
        self.y_cpu = y.astype(np.int64)
        
        # Für MLP/ResNet weiterhin den großen Tensor bereithalten
        self.X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        self.y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    def on_epoch_end(self, model, epoch, loss):
        model.eval()
        
        # --- NEU: BATched Tracking nur für FTT ---
        if isinstance(model, FTTransformer):
            batch_size = 1024
            Z_list = []
            losses_list = []
            loss_fn = nn.CrossEntropyLoss(reduction='none')

            with torch.no_grad():
                for i in range(0, len(self.X_cpu), batch_size):
                    xb = torch.tensor(self.X_cpu[i:i+batch_size], dtype=torch.float32).to(self.device)
                    yb = torch.tensor(self.y_cpu[i:i+batch_size], dtype=torch.long).to(self.device)

                    features = model.get_features(xb)
                    Z_list.append(features.cpu().numpy())

                    logits = model(xb)
                    batch_losses = loss_fn(logits, yb)
                    losses_list.append(batch_losses.cpu().numpy())

            Z = np.vstack(Z_list)
            all_losses = np.concatenate(losses_list)

            mask0 = self.y_cpu == 0
            mask1 = self.y_cpu == 1
            loss_class_0 = all_losses[mask0].mean() if mask0.any() else np.nan
            loss_class_1 = all_losses[mask1].mean() if mask1.any() else np.nan

        # --- ALTES VERHALTEN FÜR MLP & RESNET ---
        else:
            with torch.no_grad():
                if hasattr(model, 'get_features_batched'):
                    Z = model.get_features_batched(self.X_tensor).cpu().numpy()
                else:
                    Z = model.get_features(self.X_tensor).cpu().numpy()

                logits = model(self.X_tensor)
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fn(logits.cpu(), self.y_tensor.cpu())

            mask0 = self.y_tensor.cpu() == 0
            mask1 = self.y_tensor.cpu() == 1
            loss_class_0 = losses[mask0].mean().item() if mask0.any() else np.nan
            loss_class_1 = losses[mask1].mean().item() if mask1.any() else np.nan

        # --- GEMEINSAME METRIK-BERECHNUNG ---
        metrics = compute_metrics_structured(Z, self.y_cpu, self.ref_mu0, self.ref_mu1)
        if metrics:
            if self.ref_mu0 is None:
                self.ref_mu0 = metrics["means"]["mu0"]
                self.ref_mu1 = metrics["means"]["mu1"]
            self._process(metrics, epoch, loss, loss_class_0, loss_class_1)

class TabNetTracker(Callback, BaseTracker):
    def __init__(self, X, y):
        BaseTracker.__init__(self)
        Callback.__init__(self)
        self.X_cpu = X
        self.y_cpu = y.astype(np.int64)

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.network.eval()
        device = next(self.trainer.network.parameters()).device

        batch_size = 2048
        Z_list = []
        losses_list = []
        
        # Loss-Funktion definieren (ohne Reduktion, um per-Sample Loss zu kriegen)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        with torch.no_grad():
            for i in range(0, len(self.X_cpu), batch_size):
                xb = torch.tensor(self.X_cpu[i:i+batch_size], dtype=torch.float32).to(device)
                yb = torch.tensor(self.y_cpu[i:i+batch_size], dtype=torch.long).to(device)
                
                # 1. Vorhersagen (Logits) aus dem Netzwerk holen
                logits, _ = self.trainer.network(xb)
                
                # 2. Per-Sample Loss berechnen und speichern
                batch_losses = loss_fn(logits, yb)
                losses_list.append(batch_losses.cpu().numpy())
                
                # 3. Features (für die Metriken) aus dem inneren TabNet holen
                features, _ = self.trainer.network.tabnet(xb)
                Z_list.append(features.cpu().numpy())

        Z = np.vstack(Z_list)
        all_losses = np.concatenate(losses_list)
        
        # --- NEU: Klassenspezifischen Loss berechnen ---
        mask0 = self.y_cpu == 0
        mask1 = self.y_cpu == 1
        loss_class_0 = all_losses[mask0].mean() if mask0.any() else np.nan
        loss_class_1 = all_losses[mask1].mean() if mask1.any() else np.nan

        metrics = compute_metrics_structured(Z, self.y_cpu, self.ref_mu0, self.ref_mu1)

        loss = logs.get("loss", 0.0) if logs else 0.0

        if metrics:
            if self.ref_mu0 is None:
                self.ref_mu0 = metrics["means"]["mu0"]
                self.ref_mu1 = metrics["means"]["mu1"]
            self._process(metrics, epoch, loss, loss_class_0, loss_class_1)
