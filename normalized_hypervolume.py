import numpy as np
from pymoo.indicators.hv import Hypervolume

def _is_nondominated(points):
    """
    Return boolean mask for nondominated points (minimization).
    points: (n,2) array
    """
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # any point j dominates i?
        for j in range(n):
            if j == i:
                continue
            # j dominates i if j <= i in all objectives and < in at least one (minimization)
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
        # remove points dominated by i to speed up
        if mask[i]:
            for j in range(n):
                if j == i or not mask[j]:
                    continue
                if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                    mask[j] = False
    return mask

def compute_nhv(pareto_front, ref_point=None, eps=1e-9):
    """
    Compute HV, L (front length) and NHV = HV / L for a bi-objective minimization front.
    pareto_front: array-like shape (N,2)
    ref_point: optional 2-element sequence. If None, will use max(pf,axis=0)+0.1*range
    Returns: dict { 'hv': float, 'L': float, 'nhv': float, 'hv_raw': float }
    """
    pf = np.asarray(pareto_front, dtype=float)
    if pf.ndim != 2 or pf.shape[1] != 2:
        raise ValueError("pareto_front must be shape (N,2) for bi-objective problems.")
    # 1) keep nondominated points
    mask = _is_nondominated(pf)
    nd = pf[mask]
    if nd.shape[0] == 0:
        return {'hv': 0.0, 'L': 0.0, 'nhv': np.nan}
    # 2) sort by first objective (increasing) so distances along front are meaningful
    nd = nd[np.argsort(nd[:, 0])]
    # 3) choose reference point (worse than all solutions)
    if ref_point is None:
        fmin = np.min(nd, axis=0)
        fmax = np.max(nd, axis=0)
        span = fmax - fmin
        # if span is zero in an objective, add small margin
        margin = np.where(span > 0, 0.1 * span, 0.1 + eps)
        ref_point = fmax + margin
    else:
        ref_point = np.asarray(ref_point, dtype=float)
        if ref_point.shape != (2,):
            raise ValueError("ref_point must be length 2.")
    # 4) compute hypervolume (pymoo's Hypervolume expects minimization and ref_point worse than solutions)
    hv_calc = Hypervolume(ref_point=ref_point)
    hv_value = float(hv_calc(nd))
    # 5) compute front length L as sum of Euclidean distances between consecutive sorted points
    if nd.shape[0] < 2:
        L = 0.0
    else:
        diffs = nd[1:] - nd[:-1]
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        L = float(seg_lengths.sum())
    # 6) NHV = HV / L (guard against L==0)
    nhv = hv_value / L if L > eps else np.nan
    return {'hv': hv_value, 'L': L, 'nhv': nhv, 'ref_point': ref_point, 'nd_front': nd}

# -------------------------
# Example usage with a CSV
# -------------------------
if __name__ == "__main__":
    # load your pareto front CSV (each row: f1,f2). Replace filename.
    pf = np.loadtxt("f:/PYTHON/your_pareto_front.csv", delimiter=",")
    res = compute_nhv(pf)
    print(f"Hypervolume (HV)      : {res['hv']:.6f}")
    print(f"Front length (L)      : {res['L']:.6f}")
    print(f"Normalized HV (NHV)   : {res['nhv']}")
    print(f"Reference point used  : {res['ref_point']}")
