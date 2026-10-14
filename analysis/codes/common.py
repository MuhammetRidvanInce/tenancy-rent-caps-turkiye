"""Shared helpers: paths, TurkStat file resolution, weighted statistics, estimation and inference."""
from pathlib import Path
import hashlib
import os
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

PROJECT = Path(__file__).resolve().parents[2]
CODES = PROJECT / "analysis" / "codes"
OUT = PROJECT / "analysis" / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
DIAG = OUT / "diagnostics"
VALID = OUT / "validation"
PROCESSED = PROJECT / "datas" / "processed"
METADATA = PROJECT / "datas" / "metadata"
for _d in (TABLES, FIGURES, DIAG, VALID, PROCESSED, METADATA):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20261010
B_WCB = 9999
REGION_RECODE = {"TR11": "TR10", "TR91": "TR90"}  # 2014-2015 SILC NUTS-2 codes
QUAKE_REGIONS = ["TR63", "TRC1", "TRB1", "TR62", "TRC2"]
METRO_REGIONS = ["TR10", "TR31", "TR51"]  # Istanbul, Izmir, Ankara


def data_root() -> Path:
    configured = os.environ.get("TUIK_MICRODATA_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"TUIK_MICRODATA_ROOT is not a directory: {root}")
        return root
    for parent in Path(__file__).resolve().parents:
        cand = parent / "0-MikroDatas" / "datas"
        if cand.is_dir():
            return cand
    raise FileNotFoundError("Set TUIK_MICRODATA_ROOT to the local directory containing the authorised microdata year folders.")


def _candidates(year: int, survey_dir: str, filename: str):
    root = data_root() / str(year) / survey_dir / "CSV"
    hits = [p for p in root.rglob(filename) if "EN" not in p.parts and "kopyalar" not in p.parts]
    order = {"Mevcut_MikroDatas": 0, "TUIK_2026": 1, "TUIK-Yeni": 2, "TUIK_GYKA": 3, "TUIK-VERI": 4}
    return sorted(hits, key=lambda p: min([order.get(x, 9) for x in p.parts] + [9]))


def resolve(year: int, survey: str, table: str) -> Path:
    """survey: 'GYKA' (SILC cross-section) or 'HHBA' (Household Budget Survey)."""
    if survey == "GYKA":
        name, sdir = f"gyk{str(year)[2:]}_{table}.csv", "GYKA_Kesit"
    else:
        name, sdir = f"{table}_cd_{year}.csv", "HHBA_Hanehalki_Butce"
    c = _candidates(year, sdir, name)
    if not c:
        raise FileNotFoundError(f"{survey} {year} {table}")
    return c[0]


_MANIFEST = []


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tuik(path: Path) -> pd.DataFrame:
    with open(path, encoding="ISO-8859-9") as fh:
        head = fh.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    d = pd.read_csv(path, sep=sep, encoding="ISO-8859-9", low_memory=False,
                    decimal="," if sep == ";" else ".")
    d.columns = [str(c).strip().strip('"').upper() for c in d.columns]
    rel = Path(path).resolve().relative_to(data_root()).as_posix()
    _MANIFEST.append({"file": rel, "sha256": sha256(path), "bytes": Path(path).stat().st_size,
                      "separator": sep, "encoding": "ISO-8859-9", "rows": int(len(d)), "columns": int(d.shape[1])})
    return d


def write_manifest(name: str):
    p = METADATA / f"sources_{name}.json"
    p.write_text(json.dumps({"institution": "TurkStat (TÜİK)", "files": _MANIFEST,
                             "note": "Raw files are read in place and never modified or copied."},
                            ensure_ascii=False, indent=2), encoding="utf-8")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x) & ~np.isnan(w)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan


def wquantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x)
    x, w = x[m], w[m]
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])


# ----------------------------------------------------------------------------- estimation
def _design(formula, data, fe):
    """Build y, X with patsy; if fe is given, demean every column within fe groups (weighted later)."""
    if fe:
        formula = formula.replace(f"+ C({fe})", "").replace(f"C({fe}) +", "")
        formula = formula.strip()
        if " - 1" not in formula:
            formula = formula + " - 1"
    y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
    return y, X


def _absorb(M, groups, w):
    """Weighted within transformation of the columns of M by groups."""
    M = np.asarray(M, float)
    g = pd.factorize(groups)[0]
    sw = np.bincount(g, weights=w)
    means = np.vstack([np.bincount(g, weights=w * M[:, j]) / sw for j in range(M.shape[1])]).T
    return M - means[g]


def wls_cluster(formula, data, weight, cluster, fe=None, rank_revealing=False):
    """Weighted least squares with CR1 cluster-robust covariance and t(G-1) reference.

    fe: name of a categorical column absorbed by the weighted within transformation (Frisch-Waugh-Lovell);
    the formula may contain the term C(fe), which is then removed and absorbed. Fixed effects nested in the
    clusters are not counted in the small-sample correction (same convention as reghdfe).
    """
    if fe is None and "C(ry)" in formula:
        fe = "ry"
    y, X = _design(formula, data, fe)
    idx = y.index
    w = data.loc[idx, weight].to_numpy(float)
    groups = pd.factorize(data.loc[idx, cluster])[0]
    if fe:
        yv = _absorb(y.to_numpy(), data.loc[idx, fe], w)
        Xv = _absorb(X.to_numpy(), data.loc[idx, fe], w)
        keep = np.abs(Xv).sum(0) > 1e-10
        X = pd.DataFrame(Xv[:, keep], index=idx, columns=X.columns[keep])
        y = pd.DataFrame(yv, index=idx, columns=y.columns)
        # drop collinear columns (e.g. dummies absorbed by fe)
        if rank_revealing:
            from scipy.linalg import qr
            _, r, piv = qr(X.to_numpy() * np.sqrt(w)[:, None], mode="economic", pivoting=True)
            rank = int((np.abs(np.diag(r)) > np.abs(r[0, 0]) * 1e-10).sum())
            X = X.iloc[:, sorted(piv[:rank])]
        else:
            q, r = np.linalg.qr(X.to_numpy() * np.sqrt(w)[:, None])
            good = np.abs(np.diag(r)) > 1e-8 * np.abs(np.diag(r)).max()
            X = X.loc[:, good]
    m = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    return m, X, y, idx


class WildClusterBootstrap:
    """Restricted Webb bootstrap-t for H0: c'b = 0, including linear contrasts.

    Works in the fitted model's weighted/absorbed metric. Absorbed effects must
    be nested in clusters. Cluster sufficient statistics exactly reproduce OLS
    refitting and CR1 studentisation for every draw; no observation-level draws
    are needed. All tests use the same seeded sequence and (exceedances+1)/(B+1).
    """
    def __init__(self, m, X, y, data, weight, cluster, B=B_WCB, seed=SEED):
        self.m = m
        self.names = list(X.columns)
        sw = np.sqrt(data.loc[X.index, weight].to_numpy(float))
        self.A = X.to_numpy(float) * sw[:, None]
        self.y = y.iloc[:, 0].to_numpy(float) * sw
        self.g = pd.factorize(data.loc[X.index, cluster])[0]
        self.G = int(self.g.max() + 1)
        n, k = self.A.shape
        if self.G < 2 or n <= k or B < 1:
            raise ValueError("Bootstrap needs at least two clusters, n > k and B >= 1")
        self.inv = np.linalg.inv(self.A.T @ self.A)
        self.cross = np.stack([self.A[self.g == j].T @ self.A[self.g == j] for j in range(self.G)])
        self.corr = self.G / (self.G - 1) * (n - 1) / (n - k)
        self.B, self.seed = int(B), int(seed)
        webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
        self.draws = np.random.default_rng(seed).choice(webb, size=(B, self.G))

    def test(self, weights):
        if isinstance(weights, str):
            weights = {weights: 1}
        c = np.zeros(len(self.names))
        for name, value in weights.items():
            c[self.names.index(name)] = value
        denom = float(c @ self.inv @ c)
        if denom <= 0:
            raise ValueError("Non-estimable or zero contrast")
        beta = self.m.params.to_numpy()
        restricted = beta - self.inv @ c * (c @ beta) / denom
        ur = self.y - self.A @ restricted
        scores = np.vstack([self.A[self.g == j].T @ ur[self.g == j] for j in range(self.G)])
        h = c @ self.inv
        q = scores @ h
        L = np.einsum('k,gkl->gl', h, self.cross)
        bs = self.draws @ scores @ self.inv
        sc = self.draws * q[None, :] - bs @ L.T
        se = np.sqrt(self.corr * np.sum(sc ** 2, axis=1))
        ts = (bs @ c) / se
        t0 = float(c @ beta / np.sqrt(c @ self.m.cov_params().to_numpy() @ c))
        if not np.isfinite(ts).all():
            raise FloatingPointError("Non-finite bootstrap statistics")
        exceedances = int(np.sum(np.abs(ts) >= abs(t0)))
        return {"p_wcb": (exceedances + 1) / (self.B + 1),
                "wcb_B": self.B, "wcb_seed": self.seed, "wcb_exceedances": exceedances}


def wild_cluster_p(formula, data, weight, cluster, term, B=B_WCB, seed=SEED, fe=None):
    m, X, y, idx = wls_cluster(formula, data, weight, cluster, fe)
    return WildClusterBootstrap(m, X, y, data, weight, cluster, B, seed).test(term)["p_wcb"]


REGISTRY = []


def register(model_id, outcome, sample, spec, term, m, n, clusters=None, p_wcb=None, extra=None):
    b = float(m.params[term]); se = float(m.bse[term])
    ci = m.conf_int().loc[term]
    row = {"model_id": model_id, "outcome": outcome, "sample": sample, "spec": spec, "term": term,
           "coef": b, "se": se, "t": b / se, "p": float(m.pvalues[term]), "ci_low": float(ci[0]),
           "ci_high": float(ci[1]), "p_wcb": p_wcb, "n": int(n), "clusters": clusters}
    if extra:
        row.update(extra)
    REGISTRY.append(row)
    return row


def save_registry(name):
    df = pd.DataFrame(REGISTRY)
    df.to_csv(TABLES / f"registry_{name}.csv", index=False, encoding="utf-8")
    return df


def lincom(m, weights: dict):
    """Linear combination of coefficients; uses the fitted model's covariance and t reference."""
    r = np.zeros(len(m.params))
    for k, v in weights.items():
        r[list(m.params.index).index(k)] = v
    tt = m.t_test(r)
    return float(np.squeeze(tt.effect)), float(np.squeeze(tt.sd)), float(np.squeeze(tt.pvalue))


def register_lincom(model_id, outcome, sample, spec, term, m, weights, n, clusters, bootstrap=None):
    r = np.zeros(len(m.params))
    for name, value in weights.items():
        r[list(m.params.index).index(name)] = value
    tt = m.t_test(r)
    b, se, p = lincom(m, weights)
    ci = np.asarray(tt.conf_int()).ravel()
    row = {"model_id": model_id, "outcome": outcome, "sample": sample, "spec": spec,
           "term": term, "coef": b, "se": se, "t": b / se, "p": p,
           "ci_low": float(ci[0]), "ci_high": float(ci[1]), "n": int(n), "clusters": clusters,
           "p_wcb": np.nan, "contrast": json.dumps(weights)}
    if bootstrap is not None:
        row.update(bootstrap.test(weights))
    REGISTRY.append(row)
    return row
