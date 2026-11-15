"""Semantic model definition for the Bank Campaign Strategy report (writes TMDL)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import col, meas, m_table, calc_table, measure_table, field_param, rel, write_model, csv_expressions  # noqa: E402

REPO = "powerbi-bank-campaign-strategy"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "BankCampaignStrategy"

DATE = '''ADDCOLUMNS(
    CALENDAR(DATE(2008, 5, 1), DATE(2010, 11, 30)),
    "Year", YEAR([Date]),
    "Quarter", "Q" & QUARTER([Date]),
    "Month Number", MONTH([Date]),
    "Month", FORMAT([Date], "MMM"),
    "Year Month", FORMAT([Date], "MMM yy"),
    "Year Month Sort", YEAR([Date]) * 100 + MONTH([Date]),
    "Month Start", DATE(YEAR([Date]), MONTH([Date]), 1)
)'''

TABLES = [
    calc_table("Date", DATE, [
        col("Date", "dateTime", "dd mmm yyyy", key=True), col("Year", "int64", "0"), col("Quarter"),
        col("Month Number", "int64", "0", hidden=True), col("Month", sort="Month Number"),
        col("Year Month", sort="Year Month Sort"), col("Year Month Sort", "int64", "0", hidden=True),
        col("Month Start", "dateTime", "mmm yyyy", desc="First day of the month (continuous monthly axis)")],
        desc="Calendar May 2008 - Nov 2010. Contacts join on the first day of their campaign month.", is_date=True),
    m_table("Client", "dim_client.csv", [
        col("ClientKey", "int64", "0", hidden=True, key=True), col("Age", "int64", "0"), col("AgeBand", sort="AgeBandOrder", desc="Age group"),
        col("AgeBandOrder", "int64", "0", hidden=True), col("Job"), col("Marital"), col("Education", sort="EducationOrder"),
        col("EducationOrder", "int64", "0", hidden=True), col("CreditDefault", desc="Has credit in default"),
        col("HousingLoan"), col("PersonalLoan")], desc="Client profile at the time of the campaign (one row per client record)."),
    m_table("Attempt Band", "dim_attempt_band.csv", [col("AttemptBand", sort="AttemptBandOrder", desc="Calls made to the client in this campaign"),
                                                     col("AttemptBandOrder", "int64", "0", hidden=True)], desc="Call-attempt buckets."),
    m_table("Weekday", "dim_weekday.csv", [col("WeekdayNum", "int64", "0", hidden=True), col("Weekday", sort="WeekdayNum")], desc="Weekday of the last contact."),
    m_table("Prior Outcome", "dim_prior_outcome.csv", [col("PriorOutcome", sort="PriorOutcomeOrder", desc="Result of the previous campaign for this client"),
                                                       col("PriorOutcomeOrder", "int64", "0", hidden=True)], desc="Previous-campaign outcome."),
    m_table("Contacts", "fact_contact.csv", [
        col("ClientKey", "int64", "0", hidden=True), col("MonthStart", "dateTime", "mmm yyyy", hidden=True), col("WeekdayNum", "int64", "0", hidden=True),
        col("Channel", desc="Contact channel: mobile or landline"), col("Calls", "int64", "0", hidden=True), col("AttemptBand", hidden=True),
        col("PreviousContacts", "int64", "0", hidden=True), col("PriorOutcome", hidden=True), col("DaysSincePrior", "double", "0", hidden=True),
        col("CallDurationSec", "int64", "0", hidden=True), col("Subscribed", "int64", "0", hidden=True)],
        desc="Fact: one row per client in the campaign. Calls = calls made to that client in this campaign, including the last."),
    m_table("Macro", "fact_macro_month.csv", [
        col("MonthStart", "dateTime", "mmm yyyy", hidden=True), col("Euribor3M", "double", "0.00", hidden=True),
        col("EmploymentVariationRate", "double", "0.0", hidden=True), col("ConsumerPriceIndex", "double", "0.00", hidden=True),
        col("ConsumerConfidenceIndex", "double", "0.0", hidden=True), col("EmployedThousands", "double", "#,0", hidden=True)],
        desc="Fact: monthly macro-economic context (Portugal): Euribor 3M, employment, prices, confidence."),
    calc_table("Attempt Cap", "GENERATESERIES(1, 15, 1)", [col("Attempt Cap", "int64", "0", src="Value", desc="Maximum calls per client in the simulation")],
               desc="What-if parameter for the contact-strategy simulator."),
]

F_CAMP, F_SEG, F_SIM, F_MAC, F_UI = "1 Campaign", "2 Segments", "3 Simulator", "4 Macro", "9 Report UI"
MEASURES = [
    meas("Clients Contacted", "COUNTROWS('Contacts')", "#,0", F_CAMP, "Clients reached in the campaign (one row per client record)."),
    meas("Subscriptions", "SUM('Contacts'[Subscribed])", "#,0", F_CAMP, "Clients who subscribed to the term deposit."),
    meas("Conversion Rate", "DIVIDE([Subscriptions], [Clients Contacted])", "0.0%", F_CAMP, "Subscriptions / clients contacted."),
    meas("Calls", "SUM('Contacts'[Calls])", "#,0", F_CAMP, "Total calls made, including repeat attempts."),
    meas("Calls per Subscription", "DIVIDE([Calls], [Subscriptions])", "0.0", F_CAMP, "Calls needed for one subscription: the cost side of the campaign."),
    meas("Calls per Client", "DIVIDE([Calls], [Clients Contacted])", "0.00", F_CAMP),
    meas("Avg Call Minutes", "DIVIDE(AVERAGE('Contacts'[CallDurationSec]), 60)", "0.0", F_CAMP,
         "Outcome metric only: duration is known after the call ends, so it is never used for targeting."),
    meas("Prior Success Conversion", "CALCULATE([Conversion Rate], 'Prior Outcome'[PriorOutcome] = \"Previous: yes\")", "0.0%", F_CAMP),
    meas("Mobile Conversion", "CALCULATE([Conversion Rate], 'Contacts'[Channel] = \"Mobile\")", "0.0%", F_CAMP),
    meas("Campaign Average Conversion", '''CALCULATE([Conversion Rate], ALLSELECTED('Client'), ALLSELECTED('Prior Outcome'), ALLSELECTED('Attempt Band'),
    ALLSELECTED('Weekday'), ALLSELECTED('Contacts'[Channel]))''', "0.0%", F_SEG, "Conversion across the whole current selection, the baseline for lift."),
    meas("Lift vs Average", "Lift([Conversion Rate], [Campaign Average Conversion])", "0.00\"x\"", F_SEG,
         "Segment conversion relative to the campaign average (1.00x = average). Uses the Lift UDF."),
    meas("Share of Subscriptions", "DIVIDE([Subscriptions], CALCULATE([Subscriptions], ALLSELECTED('Client')))", "0.0%", F_SEG),

    meas("Attempt Cap Value", "SELECTEDVALUE('Attempt Cap'[Attempt Cap], 5)", "0", F_SIM, "Selected maximum calls per client (default 5)."),
    meas("Calls Saved at Cap", '''VAR _cap = [Attempt Cap Value]
RETURN SUMX(FILTER('Contacts', 'Contacts'[Calls] > _cap), 'Contacts'[Calls] - _cap)''', "#,0", F_SIM,
         "Calls not made if no client received more than the cap. Exact, because Calls counts every attempt."),
    meas("Subscriptions Lost at Cap", '''VAR _cap = [Attempt Cap Value]
RETURN CALCULATE([Subscriptions], 'Contacts'[Calls] > _cap)''', "#,0", F_SIM,
         "Subscribers who converted on a call after the cap (their last call was the converting one)."),
    meas("Calls Saved %", "DIVIDE([Calls Saved at Cap], [Calls])", "0.0%", F_SIM),
    meas("Subscriptions Kept %", "1 - DIVIDE([Subscriptions Lost at Cap], [Subscriptions])", "0.0%", F_SIM),
    meas("Calls per Subscription at Cap", "DIVIDE([Calls] - [Calls Saved at Cap], [Subscriptions] - [Subscriptions Lost at Cap])", "0.0", F_SIM),

    meas("Avg Euribor 3M", "AVERAGE('Macro'[Euribor3M])", "0.00\"%\"", F_MAC, "3-month Euribor, monthly average (percent)."),
    meas("Avg Consumer Confidence", "AVERAGE('Macro'[ConsumerConfidenceIndex])", "0.0", F_MAC),

    meas("Cap Caption", "\"Cap: \" & [Attempt Cap Value] & \" calls per client\"", None, F_UI),
]

RELS = [
    rel("Contacts", "ClientKey", "Client", "ClientKey"), rel("Contacts", "MonthStart", "Date", "Date"),
    rel("Contacts", "WeekdayNum", "Weekday", "WeekdayNum"), rel("Contacts", "AttemptBand", "Attempt Band", "AttemptBand"),
    rel("Contacts", "PriorOutcome", "Prior Outcome", "PriorOutcome"), rel("Macro", "MonthStart", "Date", "Date"),
]

FUNCTIONS = [
    dict(name="Lift", desc="Ratio of a segment rate to a baseline rate; blank when the baseline is blank or zero.",
         body="""(segmentRate : NUMERIC, baseRate : NUMERIC) =>
IF(NOT ISBLANK(baseRate) && baseRate <> 0, DIVIDE(segmentRate, baseRate))"""),
]


def build(local=True):
    seg = field_param("Segment by", [("Job", "'Client'[Job]"), ("Age band", "'Client'[AgeBand]"), ("Education", "'Client'[Education]"),
                                    ("Marital status", "'Client'[Marital]"), ("Housing loan", "'Client'[HousingLoan]"),
                                    ("Personal loan", "'Client'[PersonalLoan]"), ("Contact channel", "'Contacts'[Channel]"),
                                    ("Previous campaign", "'Prior Outcome'[PriorOutcome]")],
                      desc="Field parameter: lets the viewer choose the segment dimension on the Segment Lift page.")
    tables = TABLES + [measure_table("KPI", MEASURES, "All report measures, organised in display folders."), seg]
    exprs = csv_expressions(REPO, os.path.join(ROOT, "data") + "\\", default="Local" if local else "GitHub")
    write_model(os.path.join(ROOT, f"{NAME}.SemanticModel"), NAME, tables, RELS, exprs, functions=FUNCTIONS,
                desc="Bank Campaign Strategy: 41K real telemarketing contacts (Portuguese retail bank, 2008-2010) in a star schema.")


if __name__ == "__main__":
    build(local="--github" not in sys.argv)
    print("model written")
