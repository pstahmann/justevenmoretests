import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cityblock, cosine, mahalanobis
from scipy.linalg import pinv, det, eigvalsh, inv, eigh
from scipy.stats import wasserstein_distance
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from pytorch_tabnet.callbacks import Callback

EPS = 1e-9

# FIX #4: Globaler deterministischer RNG für Subsampling in Metriken
_METRIC_RNG = np.random.RandomState(42)

def reset_metric_rng(seed=42):
    """Setzt den Metrik-RNG zurück (z.B. zu Beginn jedes Seeds)."""
    _METRIC_RNG.seed(seed)


# ============================================================================
# 1. MATHEMATISCHE HILFSFUNKTIONEN
# ============================================================================

# FIX: Robuste Kovarianzberechnung für extrem kleine Minderheitsklassen
def _robust_cov(X_mat, reg_base=1e-6):
    """Verhindert singuläre Matrizen, wenn N < D (zu wenige Samples für Features)."""
    N, D = X_mat.shape
    if N == 0:
        return np.eye(D) * reg_base
        
    # Wenn weniger Samples als Dimensionen, brauche stärkere Regularisierung
    reg_val = reg_base if N >= D else reg_base * 1e4
    
    c = np.cov(X_mat, rowvar=False)
    # Edge Case: 1D Feature-Space
    if c.ndim == 0:
        c = np.array([[c]])
        
    return c + np.eye(D) * reg_val

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def manhattan_distance(a, b):
    return cityblock(a, b)

def cosine_distance(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < EPS or norm_b < EPS:
        return 0.0
    return cosine(a, b)

def population_vector_angle(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < EPS or norm_b < EPS:
        return 0.0
    a_n = a / norm_a
    b_n = b / norm_b
    return np.arccos(np.clip(np.dot(a_n, b_n), -1.0, 1.0))

def mahalanobis_distance(a, b, cov0, cov1):
    cov = (cov0 + cov1) / 2
    try:
        inv_cov = pinv(cov)
        return mahalanobis(a, b, inv_cov)
    except np.linalg.LinAlgError:
        return np.nan

def bhattacharyya_distance(a, b, cov0, cov1):
    cov = (cov0 + cov1) / 2
    diff = b - a
    try:
        inv_cov = pinv(cov)
        term1 = 0.125 * diff.T @ inv_cov @ diff
        sign_c, logdet_c = np.linalg.slogdet(cov)
        sign_0, logdet_0 = np.linalg.slogdet(cov0)
        sign_1, logdet_1 = np.linalg.slogdet(cov1)
        if sign_c <= 0 or sign_0 <= 0 or sign_1 <= 0:
            return np.nan
        term2 = 0.5 * (logdet_c - 0.5 * logdet_0 - 0.5 * logdet_1)
        return term1 + term2
    except np.linalg.LinAlgError:
        return np.nan

def hellinger_distance(bhat):
    """Akzeptiert NaN sauber, kein fragiler Registry-Zugriff nötig."""
    if np.isnan(bhat) or bhat < 0:
        return np.nan
    return np.sqrt(1 - np.exp(-bhat))

def wasserstein_dist(A, B):
    """Deterministisches Subsampling (getrennt nach Klasse A und B)."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.nan
    if len(A) > 2000:
        A = A[_METRIC_RNG.choice(len(A), 2000, replace=False)]
    if len(B) > 2000:
        B = B[_METRIC_RNG.choice(len(B), 2000, replace=False)]
    return np.mean([
        wasserstein_distance(A[:, i], B[:, i])
        for i in range(min(A.shape[1], B.shape[1]))
    ])

def within_class_variance(X):
    if len(X) < 2:
        return np.nan
    return np.trace(np.cov(X, rowvar=False))

def between_class_variance(mu0, mu1):
    return np.linalg.norm(mu0 - mu1) ** 2

def within_class_scatter(cov0, cov1):
    return cov0 + cov1

def between_class_scatter(mu0, mu1, mu):
    return np.outer(mu0 - mu, mu0 - mu) + np.outer(mu1 - mu, mu1 - mu)

def global_mean(mu0, mu1, n0, n1):
    return (n0 * mu0 + n1 * mu1) / (n0 + n1)

def fisher_discriminant_ratio(Sb, Sw):
    tr_Sw = np.trace(Sw)
    if tr_Sw < EPS:
        return np.nan
    return np.trace(Sb) / tr_Sw

def cov_effective_rank(cov, eps=1e-12):
    try:
        eigvals = eigvalsh(cov)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0:
            return 0.0
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        return np.exp(entropy)
    except np.linalg.LinAlgError:
        return 1.0

def participation_ratio(cov, eps=1e-12):
    try:
        eigvals = eigvalsh(cov)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0:
            return 0.0
        denom = np.sum(eigvals ** 2)
        if denom < eps:
            return 0.0
        return (eigvals.sum() ** 2) / denom
    except np.linalg.LinAlgError:
        return 1.0

def mixed_selectivity_index(X0, X1):
    if len(X0) < 2 or len(X1) < 2:
        return np.nan
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    means = np.stack([mu0, mu1])
    signal_var = np.var(means, axis=0).mean()
    X_comb = np.vstack([X0, X1])
    total_var = np.var(X_comb, axis=0).mean()
    if total_var < EPS:
        return 0.0
    return 1.0 - (signal_var / total_var)

def ccgp_score(Z, y, cv=5, max_samples=5000):
    # FIX: Prüfen, ob beide Klassen vertreten sind und genügend Samples pro Fold da sind
    unique_y, counts = np.unique(y, return_counts=True)
    if len(unique_y) < 2 or len(Z) < cv * 2 or min(counts) < cv:
        return np.nan
        
    if len(Z) > max_samples:
        try:
            # FIX: Stratifiziertes Subsampling
            _, Z, _, y = train_test_split(Z, y, test_size=max_samples, stratify=y, random_state=_METRIC_RNG.randint(10000))
        except ValueError:
            # Fallback, falls Klasse zu selten für Stratifizierung ist
            idx = _METRIC_RNG.choice(len(Z), max_samples, replace=False)
            Z, y = Z[idx], y[idx]
            
    try:
        # FIX: Stratifizierte Folds
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
        scores = cross_val_score(clf, Z, y, cv=skf)
        return scores.mean()
    except ValueError:
        return np.nan

def manifold_tangling_index(X0, X1, k=5):
    if len(X0) < k or len(X1) < k:
        return np.nan
        
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1])
    
    if len(X) > 2000:
        try:
            # FIX: Stratifiziertes Subsampling
            _, X, _, y = train_test_split(X, y, test_size=2000, stratify=y, random_state=_METRIC_RNG.randint(10000))
        except ValueError:
            idx = _METRIC_RNG.choice(len(X), 2000, replace=False)
            X, y = X[idx], y[idx]
            
    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, indices = nbrs.kneighbors(X)
        neighbor_labels = y[indices[:, 1:]]
        cross_class_frac = (neighbor_labels != y[:, None]).mean()
        return cross_class_frac
    except ValueError:
        return np.nan

def neural_subspace_overlap(X0, X1, n_components=5):
    n_comp0 = min(n_components, X0.shape[0], X0.shape[1])
    n_comp1 = min(n_components, X1.shape[0], X1.shape[1])
    if n_comp0 < 1 or n_comp1 < 1:
        return np.nan
    try:
        pca0 = PCA(n_components=n_comp0).fit(X0)
        pca1 = PCA(n_components=n_comp1).fit(X1)
        U0, U1 = pca0.components_.T, pca1.components_.T
        if U0.shape[0] != U1.shape[0]:
            return 0.0
        s = np.linalg.svd(U0.T @ U1, compute_uv=False)
        return np.sum(s ** 2) / max(n_comp0, n_comp1)
    except np.linalg.LinAlgError:
        return np.nan

def covariance_alignment(cov0, cov1):
    norm0, norm1 = np.linalg.norm(cov0), np.linalg.norm(cov1)
    if norm0 < EPS or norm1 < EPS:
        return np.nan
    return np.sum(cov0 * cov1) / (norm0 * norm1)

def temporal_decoding_accuracy(X0, X1, cv=5, max_samples=5000):
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1])
    
    unique_y, counts = np.unique(y, return_counts=True)
    if len(unique_y) < 2 or len(X) < cv * 2 or min(counts) < cv:
        return np.nan
        
    if len(X) > max_samples:
        try:
            # FIX: Stratifiziertes Subsampling
            _, X, _, y = train_test_split(X, y, test_size=max_samples, stratify=y, random_state=_METRIC_RNG.randint(10000))
        except ValueError:
            idx = _METRIC_RNG.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
            
    try:
        # FIX: Stratifizierte Folds
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=skf)
        return scores.mean()
    except ValueError:
        return np.nan

def representational_drift(mu0, mu1, mu0_ref=None, mu1_ref=None):
    if mu0_ref is None or mu1_ref is None:
        return np.nan
    drift0 = np.linalg.norm(mu0 - mu0_ref)
    drift1 = np.linalg.norm(mu1 - mu1_ref)
    return (drift0 + drift1) / 2

def signal_correlation(X0, X1):
    if len(X0) < 2 or len(X1) < 2:
        return np.nan
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    if np.std(mu0) < EPS or np.std(mu1) < EPS:
        return 0.0
    corr = np.corrcoef(mu0, mu1)[0, 1]
    return np.nan if np.isnan(corr) else corr

def noise_correlation(X0, X1):
    def corr_matrix(X):
        if len(X) < 2:
            return np.nan
        X_res = X - X.mean(axis=0)
        std = X_res.std(axis=0)
        valid = std > EPS
        if valid.sum() < 2:
            return 0.0
        C = np.corrcoef(X_res[:, valid], rowvar=False)
        off_diag = C[np.triu_indices_from(C, k=1)]
        return np.nanmean(off_diag) if len(off_diag) > 0 else 0.0
    
    val0, val1 = corr_matrix(X0), corr_matrix(X1)
    if np.isnan(val0) or np.isnan(val1):
        return np.nanmean([val0, val1]) # Retten, falls mindestens eine Klasse Daten hat
    return np.mean([val0, val1])

def population_sparseness(X):
    if len(X) == 0:
        return np.nan
    r_mean = np.abs(X).mean(axis=0)
    N = len(r_mean)
    if N <= 1:
        return 0.0
    sq_sum = (r_mean ** 2).sum() / N
    sum_sq = (r_mean.sum() / N) ** 2
    if sq_sum < EPS:
        return 0.0
    return (1 - sum_sq / sq_sum) / (1 - 1 / N)

def lifetime_sparseness(X):
    N = X.shape[0]
    if N < 2:
        return np.nan
    r_mean = np.abs(X).mean(axis=1)
    r_sq_mean = (X ** 2).mean(axis=1)
    mask = r_sq_mean > EPS
    if not mask.any():
        return 0.0
    sparsity = (1 - (r_mean[mask] ** 2) / r_sq_mean[mask]) / (1 - 1 / N)
    return np.mean(sparsity)

def effective_rank_score(cov, eps=1e-12):
    try:
        eigvals = eigh(cov, eigvals_only=True)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0:
            return 1.0
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p + eps))
        return np.exp(entropy)
    except np.linalg.LinAlgError:
        return 1.0

def spectral_entropy_score(cov, eps=1e-12):
    try:
        eigvals = eigh(cov, eigvals_only=True)
        eigvals = eigvals[eigvals > eps]
        if len(eigvals) == 0:
            return 0.0
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p + eps))
        max_entropy = np.log(len(eigvals))
        if max_entropy == 0:
            return 0.0
        return entropy / max_entropy
    except np.linalg.LinAlgError:
        return 0.0

def log_determinant_entropy(cov, eps=1e-6):
    try:
        sign, logdet = np.linalg.slogdet(cov + np.eye(len(cov)) * eps)
        if sign <= 0:
            return np.nan
        return logdet
    except np.linalg.LinAlgError:
        return np.nan

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
    except np.linalg.LinAlgError:
        return np.nan


# ============================================================================
# 2. METRIC REGISTRY
# ============================================================================

METRIC_REGISTRY = {
    "euclidean":              lambda c: euclidean_distance(c["mu0"], c["mu1"]),
    "manhattan":              lambda c: manhattan_distance(c["mu0"], c["mu1"]),
    "cosine":                 lambda c: cosine_distance(c["mu0"], c["mu1"]),
    "population_vector_angle":lambda c: population_vector_angle(c["mu0"], c["mu1"]),
    "mahalanobis":            lambda c: mahalanobis_distance(c["mu0"], c["mu1"], c["cov0"], c["cov1"]),
    "bhattacharyya":          lambda c: bhattacharyya_distance(c["mu0"], c["mu1"], c["cov0"], c["cov1"]),
    "hellinger":              lambda c: hellinger_distance(
                                  bhattacharyya_distance(c["mu0"], c["mu1"], c["cov0"], c["cov1"])
                              ),
    "wasserstein":            lambda c: wasserstein_dist(c["X0"], c["X1"]),
    "within_var_0":           lambda c: within_class_variance(c["X0"]),
    "within_var_1":           lambda c: within_class_variance(c["X1"]),
    "between_class_variance": lambda c: between_class_variance(c["mu0"], c["mu1"]),
    "fisher_ratio":           lambda c: fisher_discriminant_ratio(
        between_class_scatter(c["mu0"], c["mu1"],
                              global_mean(c["mu0"], c["mu1"], len(c["X0"]), len(c["X1"]))),
        within_class_scatter(c["cov0"], c["cov1"]),
    ),
    "cov_trace":              lambda c: np.trace(c["cov_all"]),
    "cov_det":                lambda c: det(c["cov_all"]),
    "cov_effective_rank":     lambda c: cov_effective_rank(c["cov_all"]),
    "participation_ratio":    lambda c: participation_ratio(c["cov_all"]),
    "mixed_selectivity_index":lambda c: mixed_selectivity_index(c["X0"], c["X1"]),
    "manifold_tangling":      lambda c: manifold_tangling_index(c["X0"], c["X1"]),
    "subspace_overlap":       lambda c: neural_subspace_overlap(c["X0"], c["X1"]),
    "cov_alignment":          lambda c: covariance_alignment(c["cov0"], c["cov1"]),
    "ccgp":                   lambda c: ccgp_score(
        np.vstack([c["X0"], c["X1"]]),
        np.array([0] * len(c["X0"]) + [1] * len(c["X1"])),
    ),
    "temporal_decoding_accuracy": lambda c: temporal_decoding_accuracy(c["X0"], c["X1"]),
    "representational_drift":    lambda c: representational_drift(
        c["mu0"], c["mu1"], c.get("mu0_ref"), c.get("mu1_ref")),
    "signal_correlation":     lambda c: signal_correlation(c["X0"], c["X1"]),
    "noise_correlation":      lambda c: noise_correlation(c["X0"], c["X1"]),
    "population_sparseness":  lambda c: np.nanmean([population_sparseness(c["X0"]),
                                                  population_sparseness(c["X1"])]),
    "lifetime_sparseness":    lambda c: np.nanmean([lifetime_sparseness(c["X0"]),
                                                  lifetime_sparseness(c["X1"])]),
    "effective_rank_score":   lambda c: effective_rank_score(c["cov_all"]),
    "spectral_entropy":       lambda c: spectral_entropy_score(c["cov_all"]),
    "log_determinant":        lambda c: log_determinant_entropy(c["cov_all"]),
    "total_variance":         lambda c: total_variance_trace(c["cov_all"]),
}


# ============================================================================
# 3. COMPUTE & TRACKERS
# ============================================================================

def compute_metrics_structured(Z, y, ref_mu0=None, ref_mu1=None):
    X0 = Z[y == 0]
    X1 = Z[y == 1]
    n0, n1 = len(X0), len(X1)

    if n0 < 2 or n1 < 2:
        return None

    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    mu = Z.mean(axis=0)

    try:
        # FIX: Auf _robust_cov umgestellt für Schutz vor Singularitäten
        cov0 = _robust_cov(X0)
        cov1 = _robust_cov(X1)
        cov_all = _robust_cov(Z)
    except np.linalg.LinAlgError:
        n_feat = Z.shape[1]
        cov0 = cov1 = cov_all = np.eye(n_feat) * 1e-4

    ctx = {
        "X0": X0, "X1": X1, "mu0": mu0, "mu1": mu1, "mu": mu,
        "cov0": cov0, "cov1": cov1, "cov_all": cov_all,
        "mu0_ref": ref_mu0, "mu1_ref": ref_mu1,
    }

    flat_metrics = {}
    for name, fn in METRIC_REGISTRY.items():
        try:
            val = fn(ctx)
            if hasattr(val, 'item'):
                val = val.item()
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                val = np.nan
            flat_metrics[name] = val
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
            flat_metrics[name] = np.nan

    metrics = {
        "per_class": {
            0: {
                "within_variance":     flat_metrics.get("within_var_0", np.nan),
                "population_sparseness": population_sparseness(X0),
                "lifetime_sparseness": lifetime_sparseness(X0),
                "cov_trace":           np.trace(cov0),
                "cov_effective_rank":  cov_effective_rank(cov0),
                "participation_ratio": participation_ratio(cov0),
            },
            1: {
                "within_variance":     flat_metrics.get("within_var_1", np.nan),
                "population_sparseness": population_sparseness(X1),
                "lifetime_sparseness": lifetime_sparseness(X1),
                "cov_trace":           np.trace(cov1),
                "cov_effective_rank":  cov_effective_rank(cov1),
                "participation_ratio": participation_ratio(cov1),
            },
        },
        "between_classes": {k: flat_metrics.get(k, np.nan) for k in [
            "euclidean", "manhattan", "cosine", "population_vector_angle",
            "mahalanobis", "bhattacharyya", "hellinger", "wasserstein",
            "between_class_variance", "fisher_ratio",
        ]},
        "global": {k: flat_metrics.get(k, np.nan) for k in [
            "cov_trace", "cov_effective_rank", "participation_ratio",
            "mixed_selectivity_index", "manifold_tangling", "subspace_overlap",
            "cov_alignment", "ccgp", "temporal_decoding_accuracy",
            "representational_drift", "signal_correlation", "noise_correlation",
            "population_sparseness", "lifetime_sparseness",
        ]},
        "flat": flat_metrics,
        "means": {"mu0": mu0, "mu1": mu1, "mu": mu},
        "covs": {"cov_all": cov_all},
    }
    return metrics


# ============================================================================
# 4. BASE TRACKER
# ============================================================================

class BaseTracker:
    def __init__(self):
        self.history = []
        self.ref_mu0 = None
        self.ref_mu1 = None
        self.last_state = None

        self._prev_values = {}
        self._cum_abs_delta = {}
        self._delta_count = {}

    def _update_online_change(self, key, value):
        if np.isnan(value):
            return np.nan
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

    def _process(self, metrics, epoch, loss, val_loss=np.nan,
                 loss_class_0=np.nan, loss_class_1=np.nan):
        metrics["epoch"] = epoch + 1
        metrics["loss"] = loss
        metrics["val_loss"] = val_loss
        metrics["loss_class_0"] = loss_class_0
        metrics["loss_class_1"] = loss_class_1

        flat = metrics["flat"]

        # KL-Divergenz zwischen Epochen
        current_mu = metrics["means"]["mu"]
        current_cov = metrics["covs"]["cov_all"]

        if self.last_state is not None:
            kl_val = gaussian_kl_divergence(
                current_mu, current_cov,
                self.last_state["mu"], self.last_state["cov"],
            )
            flat["kl_divergence_epoch"] = kl_val
        else:
            flat["kl_divergence_epoch"] = 0.0

        self.last_state = {"mu": current_mu, "cov": current_cov}

        # avg_abs_change pro Metrik
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


# ============================================================================
# 5. PYTORCH TRACKER
# ============================================================================

def _extract_features_and_losses(model, X_np, y_np, device, batch_size=1024):
    model.eval()
    Z_list = []
    losses_list = []
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[i:i + batch_size], dtype=torch.float32).to(device)
            yb = torch.tensor(y_np[i:i + batch_size], dtype=torch.long).to(device)

            features = model.get_features(xb)
            Z_list.append(features.cpu().numpy())

            logits = model(xb)
            batch_losses = loss_fn(logits, yb)
            losses_list.append(batch_losses.cpu().numpy())

    Z = np.vstack(Z_list)
    all_losses = np.concatenate(losses_list)
    return Z, all_losses


class PyTorchTracker(BaseTracker):
    def __init__(self, X_train, y_train, X_cal, y_cal, device):
        super().__init__()
        self.device = device
        self.X_cpu = X_train
        self.y_cpu = y_train.astype(np.int64)
        self.X_cal_cpu = X_cal
        self.y_cal_cpu = y_cal.astype(np.int64)

    def on_epoch_end(self, model, epoch, loss, val_loss=np.nan):
        model.eval()
        Z, all_losses = _extract_features_and_losses(
            model, self.X_cpu, self.y_cpu, self.device,
        )

        mask0 = self.y_cpu == 0
        mask1 = self.y_cpu == 1
        loss_class_0 = all_losses[mask0].mean() if mask0.any() else np.nan
        loss_class_1 = all_losses[mask1].mean() if mask1.any() else np.nan

        metrics = compute_metrics_structured(Z, self.y_cpu, self.ref_mu0, self.ref_mu1)
        if metrics:
            if self.ref_mu0 is None:
                self.ref_mu0 = metrics["means"]["mu0"]
                self.ref_mu1 = metrics["means"]["mu1"]
            self._process(metrics, epoch, loss, val_loss, float(loss_class_0), float(loss_class_1))


# ============================================================================
# 6. TABNET TRACKER
# ============================================================================

class TabNetTracker(Callback, BaseTracker):
    def __init__(self, X_train, y_train, X_cal, y_cal):
        BaseTracker.__init__(self)
        Callback.__init__(self)
        self.X_cpu = X_train
        self.y_cpu = y_train.astype(np.int64)
        self.X_cal_cpu = X_cal
        self.y_cal_cpu = y_cal.astype(np.int64)

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.network.eval()
        device = next(self.trainer.network.parameters()).device

        batch_size = 2048
        Z_list = []
        losses_list = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        with torch.no_grad():
            for i in range(0, len(self.X_cpu), batch_size):
                xb = torch.tensor(self.X_cpu[i:i + batch_size],
                                  dtype=torch.float32).to(device)
                yb = torch.tensor(self.y_cpu[i:i + batch_size],
                                  dtype=torch.long).to(device)

                logits, _ = self.trainer.network(xb)
                batch_losses = loss_fn(logits, yb)
                losses_list.append(batch_losses.cpu().numpy())

                features, _ = self.trainer.network.tabnet(xb)
                Z_list.append(features.cpu().numpy())

        Z = np.vstack(Z_list)
        all_losses = np.concatenate(losses_list)

        mask0 = self.y_cpu == 0
        mask1 = self.y_cpu == 1
        loss_class_0 = all_losses[mask0].mean() if mask0.any() else np.nan
        loss_class_1 = all_losses[mask1].mean() if mask1.any() else np.nan

        val_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(self.X_cal_cpu), batch_size):
                xb = torch.tensor(self.X_cal_cpu[i:i + batch_size],
                                  dtype=torch.float32).to(device)
                yb = torch.tensor(self.y_cal_cpu[i:i + batch_size],
                                  dtype=torch.long).to(device)
                logits, _ = self.trainer.network(xb)
                bs = len(xb)
                val_total += val_loss_fn(logits, yb).item() * bs
                val_n += bs
        val_loss = val_total / val_n if val_n > 0 else np.nan

        metrics = compute_metrics_structured(Z, self.y_cpu, self.ref_mu0, self.ref_mu1)
        loss = logs.get("loss", 0.0) if logs else 0.0

        if metrics:
            if self.ref_mu0 is None:
                self.ref_mu0 = metrics["means"]["mu0"]
                self.ref_mu1 = metrics["means"]["mu1"]
            self._process(metrics, epoch, float(loss), float(val_loss), float(loss_class_0), float(loss_class_1))


# ============================================================================
# 7. BOOSTING TRACKER (XGBoost / LightGBM / CatBoost)
# ============================================================================

def _get_n_estimators(model, model_type):
    if model_type == "xgboost":
        best = getattr(model, 'best_iteration', None)
        if best is not None:
            return best + 1
        return model.n_estimators
    elif model_type == "lgbm":
        return model.n_estimators_
    elif model_type == "catboost":
        return model.tree_count_
    return 0


def _staged_predict_proba(model, X, n_trees, model_type):
    if model_type == "xgboost":
        return model.predict_proba(X, iteration_range=(0, n_trees))
    elif model_type == "lgbm":
        return model.predict_proba(X, num_iteration=n_trees)
    elif model_type == "catboost":
        raw = model.predict(X, prediction_type='Probability',
                            ntree_start=0, ntree_end=n_trees)
        return np.asarray(raw)
    raise ValueError(f"Unbekannter model_type: {model_type}")


def _get_leaf_indices(model, X, n_trees, model_type):
    if model_type == "xgboost":
        all_leaves = model.apply(X)
        return all_leaves[:, :n_trees]
    elif model_type == "lgbm":
        all_leaves = model.predict(X, pred_leaf=True)
        return all_leaves[:, :n_trees]
    elif model_type == "catboost":
        all_leaves = np.asarray(model.calc_leaf_indexes(X))
        return all_leaves[:, :n_trees]
    raise ValueError(f"Unbekannter model_type: {model_type}")


class BoostingTracker(BaseTracker):
    def __init__(self, X_train, y_train, X_cal, y_cal):
        super().__init__()
        self.X_cpu = X_train
        self.y_cpu = y_train.astype(np.int64)
        self.X_cal_cpu = X_cal
        self.y_cal_cpu = y_cal.astype(np.int64)

    def compute_from_model(self, model, model_type, n_checkpoints=100, pca_dim=32):
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.decomposition import TruncatedSVD

        n_total = _get_n_estimators(model, model_type)
        X_train_n = len(self.X_cpu)
        if n_total < 2:
            print("  [BoostingTracker] Zu wenige Bäume für Tracking.")
            return

        steps = np.unique(np.linspace(1, n_total, min(n_checkpoints, n_total), dtype=int))
        steps = steps[steps >= 2]

        all_leaves = _get_leaf_indices(model, self.X_cpu, n_total, model_type).astype(np.float32)

        # FIX: OneHotEncoder für kategoriale Leaf-IDs
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        leaves_ohe = ohe.fit_transform(all_leaves)

        fixed_dim = min(pca_dim, n_total, X_train_n - 1)
        if fixed_dim < 2:
            print("  [BoostingTracker] Zu wenige Dimensionen für TruncatedSVD.")
            return

        # FIX: TruncatedSVD für spärliche Matrizen
        svd = TruncatedSVD(n_components=fixed_dim, random_state=42)
        svd.fit(leaves_ohe)

        print(f"  [BoostingTracker] Analysiere {len(steps)} Boosting-Schritte, "
              f"OHE+SVD -> {fixed_dim}D (gefittet auf finale Embeddings)...")

        mask0 = self.y_cpu == 0
        mask1 = self.y_cpu == 1
        eps = 1e-12

        for i, step in enumerate(steps):
            probs = _staged_predict_proba(model, self.X_cpu, int(step), model_type)
            p1 = np.clip(probs[:, 1], eps, 1 - eps)

            ce = -(self.y_cpu * np.log(p1) + (1 - self.y_cpu) * np.log(1 - p1))
            loss = float(ce.mean())
            loss_class_0 = float(ce[mask0].mean()) if mask0.any() else np.nan
            loss_class_1 = float(ce[mask1].mean()) if mask1.any() else np.nan

            probs_cal = _staged_predict_proba(model, self.X_cal_cpu, int(step), model_type)
            p1_cal = np.clip(probs_cal[:, 1], eps, 1 - eps)
            ce_cal = -(self.y_cal_cpu * np.log(p1_cal) + (1 - self.y_cal_cpu) * np.log(1 - p1_cal))
            val_loss = float(ce_cal.mean())

            # FIX: Zukünftige Bäume mit -1 padden (als ignorierte Kategorie)
            padded_leaves = np.full((X_train_n, n_total), -1.0)
            padded_leaves[:, :int(step)] = _get_leaf_indices(model, self.X_cpu, int(step), model_type)
            
            step_ohe = ohe.transform(padded_leaves)
            Z = svd.transform(step_ohe)

            metrics = compute_metrics_structured(Z, self.y_cpu, self.ref_mu0, self.ref_mu1)

            if metrics:
                if self.ref_mu0 is None:
                    self.ref_mu0 = metrics["means"]["mu0"]
                    self.ref_mu1 = metrics["means"]["mu1"]

                self._process(metrics, int(step) - 1, loss, val_loss, loss_class_0, loss_class_1)

            if (i + 1) % 25 == 0:
                print(f"    Step {step}/{n_total} | Loss {loss:.4f} | Val {val_loss:.4f}")

        print(f"  [BoostingTracker] Fertig. {len(self.history)} Datenpunkte.")
