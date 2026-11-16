# Model card: term-deposit propensity score

Built and evaluated in [`notebooks/02_propensity_model`](notebooks/02_propensity_model.ipynb); used for the contact
policy in [`notebooks/03_contact_policy`](notebooks/03_contact_policy.ipynb). Owner: Mohamed Ayman (portfolio
project). Status: demonstration model on public data, not approved for live campaigns.

**Intended use.** Order a monthly call list for a term-deposit telemarketing campaign: who to call first when capacity
is short. Not for deciding campaign size on its own (see calibration), not for credit or pricing decisions, and not as
evidence that a call causes a subscription (it predicts propensity, not uplift).

**Data and period.** UCI Bank Marketing, `bank-additional-full` (Moro, Rita & Cortez, 2014; CC BY 4.0): 41,188 client
contacts of a Portuguese retail bank, May 2008 - November 2010, 11.3% subscribed. The year is rebuilt from row order
exactly as `etl/build_data.py` does. **Target:** the client subscribed to a term deposit by the end of the campaign.

**Features (18, all known before dialling).** Client: age, job, marital status, education, credit-default status,
housing loan, personal loan. Contact: channel (mobile or landline), weekday. Previous campaign: outcome, number of
contacts, a contacted-before flag and days since that contact (`pdays = 999` becomes the flag plus a missing value).
Macro: employment variation rate, consumer price index, consumer confidence, 3-month Euribor, number employed.
Excluded on purpose: `duration` (known only after the call; single-feature AUC 0.82) and `campaign` (its final value
depends on the outcome; used by the attempt cap instead). Both exclusions are unit-tested.

**Training and validation.** Out of time. Training: May 2008 - May 2010 (39,780 contacts, 9.9% subscribed). Test:
June - November 2010 (1,408 contacts, 50.0% subscribed), untouched until the final comparison. Tuning and model choice
use three expanding time-ordered folds inside the training period. Random cross-validation is shown once to measure
the trap: 0.797 AUC against 0.721 out of time for the same LightGBM.

**Model.** Logistic regression (C = 0.5) inside one sklearn pipeline: imputation with a missing indicator, standardised
numbers, one-hot categories (levels rarer than 30 rows pooled). Chosen over a tuned LightGBM because the two tie and
the logistic is simpler to govern and refit monthly.

**Metrics (out-of-time test).**

| Model | AUC (95% CI) | PR AUC | Brier |
|---|---|---|---|
| **Logistic regression with macro inputs (champion)** | **0.751 (0.726-0.776)** | **0.723** | **0.210** |
| LightGBM tuned with macro inputs (challenger) | 0.754 (0.729-0.778) | 0.712 | 0.209 |
| Logistic regression without macro inputs | 0.704 (0.677-0.730) | 0.680 | 0.259 |
| Previous-campaign outcome rule | 0.651 (0.628-0.676) | 0.638 | 0.282 |

Paired bootstrap: challenger - champion +0.002 (CI -0.010 to +0.014); macro inputs +0.047 (CI +0.033 to +0.063).
On the time-ordered folds every model is weak (0.585-0.611): AUC depends heavily on how new the regime is.

**Calibration.** Trained once, the champion predicts 1.17x the observed test rate. Retrained monthly (walk-forward), it
runs at 1.05x the observed rate before the test and 1.10x during it; without macro inputs the same model predicts
0.52-0.66x. A rolling three-month intercept shift helps the no-macro model on the test but overshoots earlier and makes
the champion worse (pre-test Brier 0.296 vs 0.158), so it is not used. The level fails at regime turns: November 2008
(82% predicted vs 5% observed), March 2009 (about 0% vs 45%) and March 2010 (27% vs 57%).

**Segment performance (test).** AUC 0.70-0.78 by age band and 0.69-0.86 by job (small groups). Landline contacts are
over-predicted almost 2x (observed 20%, predicted 39%) and students by 44% (47% vs 68%). Among previous successes the
model barely ranks (AUC 0.53); they convert at 76%. Age is the only protected attribute in the data and it is an input:
conversion is U-shaped in age (highest for under-25s and over-65s), the linear champion captures only a trend, and the
challenger captures the U. Any live use should check that a high score does not turn into extra contact pressure on
older clients.

**Explainability.** Largest coefficients: employment variation rate (-1.06 log-odds per sd), landline contact (odds
x0.46), failed previous campaign (x0.51), consumer price index (+0.62 per sd), unknown credit-default status (x0.60),
students (x1.42), Monday calls (x0.72). Grouped permutation on the test: previous-campaign history 0.18 AUC, channel
0.06, weekday 0.01, all five macro inputs together 0.001. SHAP on the challenger puts 72% on the macro inputs, because
SHAP explains the score's level and permutation explains its ranking. Macro inputs earn their place in training, by
keeping the regime out of the client coefficients.

**Decision use (notebook 03).** On 29,051 walk-forward contacts, ranking each monthly list reaches 80% of subscriptions
with 71% of calls, but within-list AUC is only 0.57. At EUR 100 margin and EUR 5 per call, the cap at five adds EUR 30.9k
over calling everyone, and an expected-value floor plus the cap EUR 59.0k. Almost all of it comes from 2008, when every
listed client lost money on average. Recommended use: cap at five, wave size set from observed conversion, the score for
call order.

**Limitations.** One bank, 2008-2010, a rate collapse and a shrinking, warmer call list; selection and economics cannot
be separated. Small test window (1,408 contacts). No client ID, so repeat clients may sit on both sides of the split.
Propensity is not uplift. Margin and call cost are assumptions.

**Monitoring and retraining.** Retrain monthly on all closed months. Monthly checks: predicted vs observed rate on the
closed month (recalibrate if the ratio leaves 0.8-1.25; treat the first month after a gap or a macro shock as a new
regime and recalibrate on its first outcomes); AUC on the last closed month (review if below 0.60, the time-ordered fold
level); PSI per input (macro shifts are expected; above 0.25 on a client feature means the list source changed).
Weekly: calls per subscription and the share of clients reaching the cap. Prove the policy with a randomised 80/20
holdout (about 2,600 listed clients at a 45% base rate to detect a 15% relative change) before rolling it out.
