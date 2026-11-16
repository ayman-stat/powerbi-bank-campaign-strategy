"""Project tests for ds/features.py on small synthetic frames (no raw data needed, so they run in CI)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ds"))
import features as F  # noqa: E402

RNG = np.random.default_rng(3)


def bank_like(months):
    """A frame with the bank-additional-full layout, one row per entry in `months` (file order = date order)."""
    n = len(months)
    d = {"age": RNG.integers(18, 90, n), "duration": RNG.integers(0, 900, n), "campaign": RNG.integers(1, 8, n),
         "pdays": RNG.choice([999, 3, 6], n), "previous": RNG.integers(0, 3, n), "month": months,
         "y": RNG.choice(["yes", "no"], n)}
    for c in ("job", "marital", "education", "default", "housing", "loan", "contact", "day_of_week", "poutcome"):
        d[c] = RNG.choice(F.LEVELS[c], n)
    for c in F.MACRO:
        d[c] = RNG.normal(size=n)
    return pd.DataFrame(d)


def test_year_is_rebuilt_from_row_order_like_the_etl():
    months = pd.Series(["may", "may", "jun", "dec", "mar", "nov", "apr", "nov"])
    assert list(F.rebuild_year(months)) == [2008, 2008, 2008, 2008, 2009, 2009, 2010, 2010]
    df = F.add_dates(bank_like(list(months)))
    assert df["call_month"].min() == pd.Timestamp("2008-05-01") and df["call_month"].max() == pd.Timestamp("2010-11-01")
    assert df["call_month"].is_monotonic_increasing


def test_duration_and_campaign_are_never_propensity_inputs():
    df = F.add_dates(bank_like(["may"] * 50))
    for macro in (True, False):
        cols = list(F.precall_features(df, macro=macro).columns)
        assert "duration" not in cols and "campaign" not in cols and F.TARGET not in cols and "y" not in cols
        assert ("euribor3m" in cols) == macro
    assert "duration" not in F.FEATURES_MACRO + F.FEATURES_CORE


def test_pdays_sentinel_becomes_a_flag_and_a_missing_value():
    df = F.add_dates(bank_like(["may"] * 3))
    df["pdays"] = [999, 0, 6]
    X = F.precall_features(df)
    assert list(X["contacted_before"]) == [0, 1, 1]
    assert np.isnan(X["days_since_prior"].iloc[0]) and list(X["days_since_prior"].iloc[1:]) == [0, 6]
    assert (X["days_since_prior"] != 999).all()


def test_categories_are_fixed_so_unseen_levels_cannot_shift_encodings():
    df = F.add_dates(bank_like(["may"] * 5))
    df["job"] = "student"
    X = F.precall_features(df)
    assert list(X["job"].cat.categories) == F.LEVELS["job"]


def test_out_of_time_split_and_folds_never_look_ahead():
    months = ["may"] * 5 + ["nov"] * 5 + ["mar"] * 5 + ["dec"] * 5 + ["mar"] * 5 + ["jun"] * 5 + ["nov"] * 5
    df = F.add_dates(bank_like(months))
    tr, te = F.oot_split(df)
    assert df.loc[tr, "call_month"].max() < F.TEST_START <= df.loc[te, "call_month"].min()
    for a, b in F.time_folds(df["call_month"], [("2009-03-01", "2009-12-01"), ("2009-12-01", "2010-06-01")]):
        assert df["call_month"].iloc[a].max() < df["call_month"].iloc[b].min()


def test_attempt_cap_simulation_is_exact():
    calls = np.array([1, 2, 6, 7, 3, 9])
    subs = np.array([1, 0, 1, 0, 1, 0])
    r = F.cap_effect(calls, subs, cap=5)
    assert r["calls_saved"] == (6 - 5) + (7 - 5) + (9 - 5)
    assert r["subscriptions_lost"] == 1                                  # the subscriber who converted on call 6
    assert r["subscriptions_kept_pct"] == pytest.approx(2 / 3)


def test_ranked_calling_and_expected_value():
    score = np.array([0.9, 0.8, 0.1, 0.05])
    t = F.ranked_calling(score, calls=[1, 7, 2, 3], subscribed=[1, 1, 0, 0], cap=5, shares=[0.5, 1.0])
    assert list(t["calls"]) == [6, 11] and list(t["subscriptions"]) == [1, 1]   # the 7-call subscriber is lost to the cap
    assert F.expected_value_per_call([0.05], margin=100, cost_per_call=5)[0] == pytest.approx(0.0)


def test_intercept_shift_matches_the_new_base_rate_and_keeps_the_ranking():
    p = RNG.uniform(0.01, 0.3, 2000)
    y = (RNG.uniform(size=2000) < 0.5).astype(int)               # base rate jumped to ~50%
    q = F.apply_shift(p, F.intercept_shift(p, y))
    assert q.mean() == pytest.approx(y.mean(), abs=1e-6)
    assert (np.argsort(q) == np.argsort(p)).all()


def test_walk_forward_only_uses_earlier_months():
    from sklearn.dummy import DummyClassifier
    months = pd.Series(pd.to_datetime(["2009-01-01"] * 4 + ["2009-02-01"] * 4 + ["2009-03-01"] * 4))
    y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1])
    X = pd.DataFrame({"x": np.arange(12)})
    s = F.walk_forward(lambda: DummyClassifier(strategy="prior"), X, y, months, "2009-02-01")
    assert s.iloc[:4].isna().all()                                 # nothing before the start month is scored
    assert np.allclose(s.iloc[4:8], 1 / 4)                        # Feb scored from Jan only, not from Feb itself
    assert np.allclose(s.iloc[8:], 3 / 8)                         # Mar scored from Jan + Feb, never from Mar's outcomes


def test_champion_pipeline_fits_on_precall_features_only():
    df = F.add_dates(bank_like(["may"] * 120 + ["jun"] * 120))
    df["y"] = np.where(RNG.uniform(size=len(df)) < 0.3, "yes", "no")
    df = F.add_dates(df.drop(columns=[F.TARGET, "year", "call_month"]))
    for macro in (True, False):
        X = F.precall_features(df, macro=macro)
        m = F.make_logistic(macro=macro).fit(X, df[F.TARGET])
        names = list(m[:-1].get_feature_names_out())
        assert not any("duration" in n or "campaign" in n for n in names)
        assert any("euribor3m" in n for n in names) == macro
        assert m.predict_proba(X).shape == (len(X), 2)
