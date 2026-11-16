"""Data loading, pre-call features and policy arithmetic for the bank campaign notebooks (imported by notebooks and tests).

The raw file is read with the ETL's own `load()` (etl/build_data.py), and the calendar year is rebuilt from row order
exactly as the ETL does it, so the notebooks and the Power BI model see the same 41,188 contacts on the same dates.

Timing rule for features: a feature is allowed only if it is known before the call is dialled.
  * `duration` is known only after the call ends (and a zero-length call is always a "no"): outcome, never an input.
  * `campaign` (calls to this client in this campaign, including the last) is known call by call, but its final value
    depends on the outcome: subscribers stop being called. It is used by the contact policy, not by the propensity score.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
REPO = Path(__file__).resolve().parents[1]
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
TARGET = "subscribed"
OUTCOME_ONLY = ["duration"]                          # after the call: never an input
POLICY_ONLY = ["campaign"]                           # outcome-dependent final count: policy, not propensity
CLIENT = ["age", "job", "marital", "education", "default", "housing", "loan"]
CONTACT = ["contact", "day_of_week"]
HISTORY = ["previous", "poutcome", "contacted_before", "days_since_prior"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
FEATURES_CORE = CLIENT + CONTACT + HISTORY
FEATURES_MACRO = FEATURES_CORE + MACRO
LEVELS = {                                           # documented levels, fixed so train and test share the same categories
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"], "housing": ["no", "yes", "unknown"], "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"], "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
}
CATEGORICAL = [c for c in FEATURES_MACRO if c in LEVELS]
TEST_START = pd.Timestamp("2010-06-01")               # out-of-time test: the last six months (Jun - Nov 2010)


def _etl():
    spec = importlib.util.spec_from_file_location("bank_etl_build_data", REPO / "etl" / "build_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_year(month: pd.Series, start: int = 2008) -> np.ndarray:
    """The file is in date order but has no year: the year increments whenever the month number falls (ETL rule)."""
    m = month.map(MONTHS).to_numpy()
    step = np.r_[0, (np.diff(m) < 0).astype(int)]
    return start + np.cumsum(step)


def load(raw: str | None = None) -> pd.DataFrame:
    """All 41,188 contacts via the ETL's loader, with the rebuilt call month and an integer target."""
    cands = [raw, os.environ.get("BANK_RAW"), REPO.parent / "_raw", REPO / ".raw", REPO / "etl" / ".raw"]
    found = next((str(c) for c in cands if c and (Path(c) / "bank.zip").exists()), None)
    df = _etl().load(found).reset_index(drop=True)       # downloads from UCI into etl/.raw if no local copy exists
    return add_dates(df)


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = rebuild_year(df["month"])
    df["call_month"] = pd.to_datetime(dict(year=df["year"], month=df["month"].map(MONTHS), day=1))
    if TARGET not in df and "y" in df:
        df[TARGET] = (df["y"] == "yes").astype(int)
    return df


def precall_features(df: pd.DataFrame, macro: bool = True) -> pd.DataFrame:
    """Inputs known before dialling. pdays = 999 is a sentinel for 'no previous contact date on record', so it becomes a
    flag plus a days-since-prior value that is missing (not 999) when there is no date."""
    X = df[CLIENT + CONTACT + ["previous", "poutcome"]].copy()
    X["contacted_before"] = (df["pdays"] != 999).astype(int)
    X["days_since_prior"] = df["pdays"].where(df["pdays"] != 999).astype(float)
    if macro:
        X[MACRO] = df[MACRO]
    X = X[FEATURES_MACRO if macro else FEATURES_CORE]
    for c in CATEGORICAL:
        if c in X:
            X[c] = pd.Categorical(X[c].astype(str), categories=LEVELS[c])
    return X


def oot_split(df: pd.DataFrame, test_start: pd.Timestamp = TEST_START) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-time split on the rebuilt call month: boolean masks (train = before test_start)."""
    te = (df["call_month"] >= test_start).to_numpy()
    return ~te, te


def time_folds(months: pd.Series, bounds) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: for each (start, end), train on months < start, validate on start <= month < end."""
    m = pd.to_datetime(pd.Series(months)).reset_index(drop=True)
    return [(np.where(m < pd.Timestamp(a))[0], np.where((m >= pd.Timestamp(a)) & (m < pd.Timestamp(b)))[0]) for a, b in bounds]


# ------------------------------------------------------------------------------------------------ policy arithmetic
def cap_effect(calls, subscribed, cap: int) -> dict:
    """Exact attempt-cap simulation (ETL/DAX definition): `calls` counts every call to the client including the last,
    and a subscriber converted on their last call. Capping at N saves sum(max(calls - N, 0)) calls and loses every
    subscriber who needed more than N calls."""
    calls, subscribed = np.asarray(calls), np.asarray(subscribed).astype(bool)
    saved = np.clip(calls - cap, 0, None).sum()
    lost = (subscribed & (calls > cap)).sum()
    return {"cap": cap, "calls": int(calls.sum()), "calls_saved": int(saved), "calls_saved_pct": saved / calls.sum(),
            "subscriptions": int(subscribed.sum()), "subscriptions_lost": int(lost), "subscriptions_kept_pct": 1 - lost / max(1, subscribed.sum())}


def expected_value_per_call(p, margin: float, cost_per_call: float):
    """EV of dialling a client once: p(subscribe) x margin - cost of the call. Break-even p = cost / margin."""
    return np.asarray(p, float) * margin - cost_per_call


def ranked_calling(score, calls, subscribed, group=None, cap: int | None = None, shares=None) -> pd.DataFrame:
    """Call the top share of clients by score (within each group, e.g. each monthly call list), optionally capping the
    calls per client. Returns calls used and subscriptions won for each share."""
    d = pd.DataFrame({"s": np.asarray(score, float), "c": np.asarray(calls), "y": np.asarray(subscribed).astype(int),
                      "g": 0 if group is None else np.asarray(group)})
    d["pct"] = d.groupby("g")["s"].rank(ascending=False, method="first", pct=True)
    used = d["c"].clip(upper=cap) if cap else d["c"]
    won = d["y"] * ((d["c"] <= cap) if cap else 1)
    shares = shares if shares is not None else np.linspace(0.05, 1.0, 20)
    rows = []
    for s in shares:
        sel = d["pct"] <= s + 1e-12
        rows.append({"share_called": s, "clients": int(sel.sum()), "calls": int(used[sel].sum()), "subscriptions": int(won[sel].sum())})
    t = pd.DataFrame(rows)
    t["calls_pct"] = t["calls"] / d["c"].sum()
    t["subscriptions_pct"] = t["subscriptions"] / d["y"].sum()
    return t


# ------------------------------------------------------------------------------------------------ time-aware scoring
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def intercept_shift(p, y) -> float:
    """The shift d on the log-odds scale that makes mean(sigmoid(logit(p) + d)) equal the observed rate of y.
    Recalibration for a base-rate change: ranking is untouched, only the level moves."""
    z, target = _logit(p), float(np.mean(y))
    d = np.log(target / (1 - target)) - np.log(np.mean(p) / (1 - np.mean(p)))   # good starting point
    for _ in range(50):                                                          # Newton steps
        q = 1 / (1 + np.exp(-(z + d)))
        g, h = q.mean() - target, (q * (1 - q)).mean()
        if abs(g) < 1e-10 or h <= 0:
            break
        d -= g / h
    return float(d)


def apply_shift(p, d: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(_logit(p) + d)))


def walk_forward(make_model, X: pd.DataFrame, y, months: pd.Series, start) -> pd.Series:
    """Score each call month with a model trained on every earlier month only (monthly retrain). Months before `start`
    are left unscored. This is how the model would have been used live, so it is the honest input for policy work."""
    months = pd.to_datetime(pd.Series(months)).reset_index(drop=True)
    y = np.asarray(y)
    out = pd.Series(np.nan, index=range(len(months)))
    for m in sorted(months[months >= pd.Timestamp(start)].unique()):
        trn, tst = (months < m).to_numpy(), (months == m).to_numpy()
        out[tst] = make_model().fit(X[trn], y[trn]).predict_proba(X[tst])[:, 1]
    return out


def rolling_recalibration(p, y, months: pd.Series, window: int = 3) -> pd.Series:
    """Recalibrate each month's scores with the intercept shift estimated on the previous `window` scored call months
    (their outcomes are known by then). The first scored month(s) keep the raw score."""
    d = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y), "m": pd.to_datetime(pd.Series(months)).to_numpy()})
    out = d["p"].copy()
    scored = sorted(d.loc[d["p"].notna(), "m"].unique())
    for i, m in enumerate(scored):
        if i == 0:
            continue
        ref = d[d["m"].isin(scored[max(0, i - window):i])]
        out[d["m"] == m] = apply_shift(d.loc[d["m"] == m, "p"], intercept_shift(ref["p"], ref["y"]))
    return out


# ------------------------------------------------------------------------------------------------ the champion pipeline
NUM_CORE = ["age", "previous", "contacted_before", "days_since_prior"]


def make_logistic(macro: bool = True, C: float = 0.5):
    """Regularised logistic regression with all preprocessing inside the pipeline (fitted on the training fold only):
    missing days-since-prior imputed with an indicator, numbers standardised, categories one-hot encoded (levels rarer
    than 30 rows pooled). Shared by notebooks 02 and 03 so the policy uses exactly the model that was validated."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = NUM_CORE + (MACRO if macro else [])
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1, add_indicator=True), StandardScaler()), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAL)])
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=3000))
