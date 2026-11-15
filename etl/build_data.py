"""Build the star-schema CSVs for the Bank Campaign Strategy Power BI model.

Source: UCI Machine Learning Repository, "Bank Marketing" (bank-additional-full.csv), CC BY 4.0.
    Moro, S., Rita, P., & Cortez, P. (2014). Real direct-marketing (telemarketing) campaigns of a
    Portuguese retail bank, May 2008 - Nov 2010. https://archive.ics.uci.edu/dataset/222/bank+marketing

Usage:  python etl/build_data.py [--raw <folder containing bank.zip>]
Output: data/*.csv

Modelling decisions (also in README):
  * Grain: one row per client record in the campaign (41,188 rows). `campaign` = number of calls made to that
    client during this campaign, including the last one, so a subscriber converted on call number `campaign`.
    That makes an "attempt cap" simulation exact: capping at N saves sum(max(calls-N,0)) calls and loses the
    subscribers whose converting call came after N.
  * The file has no year. It is ordered by date (May 2008 -> Nov 2010), so the year is rebuilt from the row
    order: it increments whenever the month number falls. The rebuilt range is asserted below.
  * Call duration is only known after a call ends, so it is reported as an outcome metric, never as a targeting input.
"""
import argparse, io, os, urllib.request, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
WEEKDAYS = {"mon": (1, "Monday"), "tue": (2, "Tuesday"), "wed": (3, "Wednesday"), "thu": (4, "Thursday"), "fri": (5, "Friday")}


def load(raw):
    zpath = os.path.join(raw, "bank.zip") if raw else None
    if not (zpath and os.path.exists(zpath)):
        os.makedirs(os.path.join(HERE, ".raw"), exist_ok=True)
        zpath = os.path.join(HERE, ".raw", "bank.zip")
        if not os.path.exists(zpath):
            urllib.request.urlretrieve(URL, zpath)
    outer = zipfile.ZipFile(zpath)
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    name = [n for n in inner.namelist() if n.endswith("bank-additional-full.csv")][0]
    return pd.read_csv(io.BytesIO(inner.read(name)), sep=";")


def label(v):
    return str(v).replace(".", " ").replace("-", " ").replace("_", " ").strip().capitalize() if v != "unknown" else "Unknown"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", default=None); a = ap.parse_args()
    df = load(a.raw).reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)

    # rebuild the year from row order
    mnum = df.month.map(MONTHS).to_numpy()
    year = np.empty(len(df), dtype=int); y = 2008
    for i in range(len(df)):
        if i and mnum[i] < mnum[i - 1]:
            y += 1
        year[i] = y
    df["MonthStart"] = pd.to_datetime(dict(year=year, month=mnum, day=1)).dt.date
    assert str(df.MonthStart.min()) == "2008-05-01" and str(df.MonthStart.max()) == "2010-11-01", (df.MonthStart.min(), df.MonthStart.max())

    df["ClientKey"] = np.arange(1, len(df) + 1)
    bands = [(17, 24, "18-24"), (25, 34, "25-34"), (35, 44, "35-44"), (45, 54, "45-54"), (55, 64, "55-64"), (65, 120, "65+")]
    def age_band(x):
        for lo, hi, lab in bands:
            if lo <= x <= hi:
                return lab
    edu = {"basic.4y": "Basic (4 years)", "basic.6y": "Basic (6 years)", "basic.9y": "Basic (9 years)", "high.school": "High school",
           "professional.course": "Professional course", "university.degree": "University degree", "illiterate": "Illiterate", "unknown": "Unknown"}
    edu_order = {"Illiterate": 1, "Basic (4 years)": 2, "Basic (6 years)": 3, "Basic (9 years)": 4, "High school": 5,
                 "Professional course": 6, "University degree": 7, "Unknown": 8}
    dim_client = pd.DataFrame({
        "ClientKey": df.ClientKey, "Age": df.age, "AgeBand": df.age.map(age_band),
        "Job": df.job.map(label), "Marital": df.marital.map(label), "Education": df.education.map(edu),
        "CreditDefault": df.default.map(label), "HousingLoan": df.housing.map(label), "PersonalLoan": df.loan.map(label)})
    dim_client["EducationOrder"] = dim_client.Education.map(edu_order)
    dim_client["AgeBandOrder"] = dim_client.AgeBand.map({b[2]: i for i, b in enumerate(bands, 1)})

    def attempt_band(c):
        return "1" if c == 1 else "2" if c == 2 else "3" if c == 3 else "4-5" if c <= 5 else "6-10" if c <= 10 else "11+"
    att_order = {"1": 1, "2": 2, "3": 3, "4-5": 4, "6-10": 5, "11+": 6}
    prior = df.poutcome.map({"nonexistent": "Not contacted before", "failure": "Previous: no", "success": "Previous: yes"})
    fact = pd.DataFrame({
        "ClientKey": df.ClientKey, "MonthStart": df.MonthStart, "WeekdayNum": df.day_of_week.map(lambda w: WEEKDAYS[w][0]),
        "Channel": df.contact.map({"cellular": "Mobile", "telephone": "Landline"}), "Calls": df.campaign,
        "AttemptBand": df.campaign.map(attempt_band), "PreviousContacts": df.previous, "PriorOutcome": prior,
        "DaysSincePrior": np.where(df.pdays == 999, np.nan, df.pdays), "CallDurationSec": df.duration,
        "Subscribed": (df.y == "yes").astype(int)})
    dim_attempt = pd.DataFrame({"AttemptBand": list(att_order), "AttemptBandOrder": list(att_order.values())})
    dim_weekday = pd.DataFrame([(n, lab) for n, lab in WEEKDAYS.values()], columns=["WeekdayNum", "Weekday"])
    dim_prior = pd.DataFrame({"PriorOutcome": ["Not contacted before", "Previous: no", "Previous: yes"], "PriorOutcomeOrder": [1, 2, 3]})
    macro = df.groupby("MonthStart").agg(Euribor3M=("euribor3m", "mean"), EmploymentVariationRate=("emp.var.rate", "mean"),
                                          ConsumerPriceIndex=("cons.price.idx", "mean"), ConsumerConfidenceIndex=("cons.conf.idx", "mean"),
                                          EmployedThousands=("nr.employed", "mean")).reset_index().round(4)

    for name, d in [("dim_client", dim_client), ("dim_attempt_band", dim_attempt), ("dim_weekday", dim_weekday),
                    ("dim_prior_outcome", dim_prior), ("fact_contact", fact), ("fact_macro_month", macro)]:
        d.to_csv(os.path.join(OUT, f"{name}.csv"), index=False, encoding="utf-8")
        print(f"{name:20s} {len(d):>7,d} rows")
    print("conversion", round(fact.Subscribed.mean() * 100, 2), "% | calls", int(fact.Calls.sum()), "| months", macro.shape[0])


if __name__ == "__main__":
    main()
