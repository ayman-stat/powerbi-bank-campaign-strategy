"""Report definition (PBIR) for the Bank Campaign Strategy report.

Identity: PRIVATE_BANK - ivory paper, deep green for the data in focus, brass for the single highlight,
Georgia action titles, a dark left navigation rail and KPI tiles with a brass top rule.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import Report, write_pbip, C, M, proj, b, i, n, s, props, idprops, color  # noqa: E402
import design as D  # noqa: E402
from design import S, GUT, use, theme, chrome, kpi_band, bar, combo, line, matrix, button_slicer, dropdown, about_page, P, _title  # noqa: E402

use(D.PRIVATE_BANK)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "BankCampaignStrategy"
TITLE = "Retail Bank Campaign Strategy"
SRC = "UCI Machine Learning Repository, Bank Marketing (Moro, Rita & Cortez, 2014), CC BY 4.0: 41,188 real telemarketing contacts, May 2008 - Nov 2010."
K = lambda m: M("KPI", m)  # noqa: E731
YM = (C("Date", "Year Month"), "Month")
YMS = C("Date", "Year Month Sort")
SEG = C("Segment by", "Segment by")


def top_slicers(p):
    button_slicer(p, "year", S.slicer_x, S.title_y + 4, 270, 76, C("Date", "Year"), "Year", columns=3)
    dropdown(p, "chan", S.slicer_x + 290, S.title_y + 4, 190, 76, C("Contacts", "Channel"), "Contact channel")


def single_buttons(p, key, x, y, w, h, field, default, cols, rows, alt, value_literal):
    """Single-select button slicer with a default selection and a brass accent on the selected button."""
    ent = field["Column"]["Expression"]["SourceRef"]["Entity"]
    prop = field["Column"]["Property"]
    return p.add(key, "advancedSlicerVisual", x, y, w, h, query={"Values": [proj(field)]},
                 objects={"layout": props(columnCount=i(cols), rowCount=i(rows)),
                          "value": idprops(fontSize=n(12), fontFamily=s(S.font_b), fontColor=color(S.ink)),
                          "fillCustom": [{"properties": {"show": b(True), "fillColor": color(S.panel)}, "selector": {"id": "default"}},
                                         {"properties": {"show": b(True), "fillColor": color(S.sel_fill)}, "selector": {"id": "selected"}}],
                          "outline": [{"properties": {"show": b(True), "lineColor": color(S.line), "weight": n(1)}, "selector": {"id": "default"}},
                                      {"properties": {"show": b(True), "lineColor": color(S.primary), "weight": n(2)}, "selector": {"id": "selected"}}],
                          "accentBar": [{"properties": {"show": b(False)}, "selector": {"id": "default"}},
                                        {"properties": {"show": b(True), "position": s("Left" if rows > 1 else "Bottom"), "color": color(S.accent),
                                                        "width": n(4)}, "selector": {"id": "selected"}}],
                          "selection": props(singleSelect=b(True)),
                          "general": [{"properties": {"filter": {"filter": {"Version": 2, "From": [{"Name": "f", "Entity": ent, "Type": 0}],
                              "Where": [{"Condition": {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": default[0]}}],
                                                               "Values": [[{"Literal": {"Value": value_literal}}]]}}}]}}}}]},
                 vco=dict(D.NOBOX), alt=alt)


def build():
    r = Report(os.path.join(ROOT, f"{NAME}.Report"), NAME, f"{NAME}.SemanticModel", theme())
    r.n_pages = 5
    x0, cw, by, bh = S.x0, S.cw, S.body_y, S.body_h
    big = (cw - GUT) * 2 // 3
    small = cw - GUT - big
    half = (cw - GUT) // 2

    # ---------------------------------------------------------------- 1 Campaign overview
    p = r.page("overview", "Campaign overview", bg=S.page)
    chrome(p, TITLE, "Conversion rose from 5% to 52% as rates fell from 4.6% to 0.8%",
           "41K clients and 106K calls, May 2008 - Nov 2010  ·  11.3% overall conversion  ·  22.8 calls per subscription", "1 / 5", SRC)
    top_slicers(p)
    kpi_band(p, "kpi", [K("Clients Contacted"), K("Subscriptions"), K("Conversion Rate"), K("Calls"), K("Calls per Subscription"), K("Avg Call Minutes")])
    top_h = 440
    combo(p, "vol", x0, by, big, top_h, YM, [(K("Clients Contacted"), "Clients contacted")], [(K("Conversion Rate"), "Conversion rate")],
          "Volume fell while conversion climbed", "Columns: clients contacted per month  ·  Line: conversion rate", sort_cat=YMS)
    line(p, "macro", x0 + big + GUT, by, small, top_h, (C("Date", "Month Start"), "Month"), [(K("Conversion Rate"), "Conversion rate")],
         "Conversion moved against the Euribor rate", "Green: conversion  ·  Brass: 3-month Euribor (%)",
         colors=(S.primary,), y2=[(K("Avg Euribor 3M"), "Euribor 3M")], y2_colors=(S.accent,), sort_cat=C("Date", "Month Start"))
    p.visuals[-1]["visual"]["objects"]["categoryAxis"] = props(axisType=s("Scalar"))
    bh2 = bh - top_h - GUT
    bar(p, "prior", x0, by + top_h + GUT, half, bh2, (C("Prior Outcome", "PriorOutcome"), "Previous campaign"), (K("Conversion Rate"), "Conversion"),
        "Conversion by previous-campaign outcome", "Clients who said yes before convert at 65%")
    bar(p, "chanbar", x0 + half + GUT, by + top_h + GUT, half, bh2, (C("Contacts", "Channel"), "Channel"), (K("Conversion Rate"), "Conversion"),
        "Conversion by contact channel", "Mobile converts at almost 3x landline")

    # ---------------------------------------------------------------- 2 Segment lift (field parameter)
    p = r.page("segments", "Segment lift", bg=S.page)
    chrome(p, TITLE, "Students and retirees convert at 2-3x the average, blue-collar at 0.6x",
           "Lift = segment conversion / campaign average  ·  Choose the dimension on the left (a field parameter)", "2 / 5", SRC)
    top_slicers(p)
    kpi_band(p, "kpi", [K("Clients Contacted"), K("Conversion Rate"), K("Prior Success Conversion"), K("Mobile Conversion"), K("Subscriptions")])
    lw = 236
    D.panel(p, "segpanel", x0, by, lw, bh)
    D.textbox(p, "seghead", x0 + 20, by + 18, lw - 40, 52, [[("Segment by", 18, S.ink, False, S.font_title)], [("Pick one dimension", 11, S.muted, False)]])
    single_buttons(p, "segby", x0 + 16, by + 84, lw - 32, 8 * 58, SEG, ("Segment by Fields",), 1, 8, "Choose the segment dimension", "'''Client''[Job]'")
    mid_w = 640
    bar(p, "lift", x0 + lw + GUT, by, mid_w, bh, C("Client", "Job"), (K("Lift vs Average"), "Lift vs average"),
        "Conversion lift by segment", "1.00x = campaign average  ·  red below, green above", gradient=(S.neg, "#E9E4DA", S.primary),
        field_param=SEG)
    rx = x0 + lw + GUT + mid_w + GUT
    rw = cw - lw - mid_w - 2 * GUT
    sc_h = 330
    matrix(p, "seg", rx, by, rw, sc_h, [C("Client", "Job")],
           [(K("Clients Contacted"), "Clients"), K("Subscriptions"), (K("Conversion Rate"), "Conversion"), (K("Lift vs Average"), "Lift"),
            (K("Share of Subscriptions"), "Share of subs")],
           "Segment scorecard", "Data bars: clients  ·  Lift shaded red (below average) to green (above)",
           gradients=[(K("Lift vs Average"), S.soft_neg, S.soft_pos, "#FFFFFF", 1)], databars=[K("Clients Contacted")],
           sort=[(K("Lift vs Average"), "Descending")], field_param=SEG)
    matrix(p, "heat", rx, by + sc_h + GUT, rw, bh - sc_h - GUT, [(C("Client", "Job"), "Job")], [(K("Conversion Rate"), "Conversion")],
           "Conversion by job and age band", "Darker green = higher conversion",
           columns=[(C("Client", "AgeBand"), "Age band")], gradients=[(K("Conversion Rate"), "#FFFFFF", "#4F8A80")], font=9)

    # ---------------------------------------------------------------- 3 Contact-strategy simulator (what-if)
    p = r.page("simulator", "Contact strategy simulator", bg=S.page)
    chrome(p, TITLE, "A 5-call cap cuts 15% of calls and keeps 96% of subscriptions",
           "30.6% of calls go to clients called six or more times, where conversion halves (13.0% on a first call vs 6.3% at 6-10)", "3 / 5", SRC)
    kpi_band(p, "kpi", [K("Calls Saved at Cap"), K("Calls Saved %"), K("Subscriptions Lost at Cap"), K("Subscriptions Kept %"),
                        K("Calls per Subscription at Cap"), K("Calls per Subscription")])
    D.panel(p, "simpanel", x0, by, big, bh)
    D.textbox(p, "simhead", x0 + 24, by + 18, big - 48, 30, [[("Policy simulator", 18, S.ink, False, S.font_title),
                                                             ("     choose the maximum calls per client; the KPIs above recalculate exactly", 12, S.muted, False)]])
    cap = single_buttons(p, "cap", x0 + 24, by + 60, big - 48, 52, C("Attempt Cap", "Attempt Cap"), ("Attempt Cap",), 15, 1,
                         "Choose the maximum number of calls per client", "5L")
    curve = line(p, "curve", x0 + 8, by + 128, big - 16, bh - 136, (C("Attempt Cap", "Attempt Cap"), "Maximum calls per client"),
                 [(K("Subscriptions Kept %"), "Subscriptions kept"), (K("Calls Saved %"), "Calls saved")],
                 "Trade-off curve: subscriptions kept vs calls saved at every cap", "The sweet spot sits where the green line flattens (4-6 calls)",
                 colors=(S.primary, S.accent), sort_cat=C("Attempt Cap", "Attempt Cap"), markers=True)
    p.visuals[-1]["visual"]["visualContainerObjects"].update({"background": props(show=b(False)), "border": props(show=b(False))})
    p.interactions.append((cap, curve, "NoFilter"))
    rx = x0 + big + GUT
    hh = (bh - GUT) // 2
    bar(p, "conv", rx, by, small, hh, (C("Attempt Band", "AttemptBand"), "Calls to the client"), (K("Conversion Rate"), "Conversion"),
        "Each extra call converts less", "Conversion by number of calls made to the client", horizontal=False, sort_desc=False,
        sort_cat=C("Attempt Band", "AttemptBandOrder"))
    bar(p, "calls", rx, by + hh + GUT, small, hh, (C("Attempt Band", "AttemptBand"), "Calls to the client"), K("Calls"),
        "Where the calls go", "Total calls by attempt band", horizontal=False, sort_desc=False, sort_cat=C("Attempt Band", "AttemptBandOrder"),
        fmt_units=1000, gradient=("#EADCC4", S.accent))

    # ---------------------------------------------------------------- 4 Timing & macro
    p = r.page("timing", "Timing & macro", bg=S.page)
    chrome(p, TITLE, "Low-rate months converted up to 10x better than high-rate months",
           "Same campaign scripts, very different economic backdrop  ·  the weekday barely matters", "4 / 5", SRC)
    top_slicers(p)
    kpi_band(p, "kpi", [K("Avg Euribor 3M"), K("Avg Consumer Confidence"), K("Conversion Rate"), K("Calls per Client")])
    hm_h = 214
    matrix(p, "ym", x0, by, big, hm_h, [C("Date", "Year")], [(K("Conversion Rate"), "Conversion")],
           "Conversion by year and month", "Darker green = higher conversion  ·  blank = no campaign that month",
           columns=[(C("Date", "Month"), "Month")], gradients=[(K("Conversion Rate"), "#FFFFFF", "#4F8A80")], font=11)
    combo(p, "cps", x0, by + hm_h + GUT, big, bh - hm_h - GUT, YM, [(K("Calls per Subscription"), "Calls per subscription")],
          [(K("Avg Euribor 3M"), "Euribor 3M (%)")], "The cost of a sale collapsed as rates fell",
          "Columns: calls needed per subscription  ·  Line: 3-month Euribor (%)", col_colors=("#9DBDB6",), line_colors=(S.accent,), sort_cat=YMS)
    rx = x0 + big + GUT
    sc_h = 440
    p.add("scatter", "scatterChart", rx, by, small, sc_h,
          query={"Category": [P(YM)], "X": [P((K("Avg Euribor 3M"), "Euribor 3M (%)"))],
                 "Y": [P((K("Conversion Rate"), "Conversion"))], "Size": [P((K("Clients Contacted"), "Clients"))]},
          objects={"dataPoint": [{"properties": {"fill": color(S.primary)}}]},
          vco=_title("Each month: Euribor vs conversion", "Bubble size: clients contacted that month"), alt="Scatter of monthly Euribor against conversion")
    bar(p, "wd", rx, by + sc_h + GUT, small, bh - sc_h - GUT, (C("Weekday", "Weekday"), "Weekday"), (K("Conversion Rate"), "Conversion"),
        "Conversion by weekday", "A 2-point spread, small next to the macro effect", horizontal=False, sort_desc=False, sort_cat=C("Weekday", "WeekdayNum"))

    # ---------------------------------------------------------------- 5 About
    p = r.page("about", "About & method", bg=S.page)
    chrome(p, TITLE, "How this report is built", "Questions, data, model and definitions behind every number", "5 / 5", SRC)
    about_page(p, [
        ("Business questions", ["Which clients are worth calling, and how much better do they convert?", "How many calls per client make sense before returns collapse?",
                                "How much of conversion is the economy rather than the campaign?"]),
        ("Data", ["UCI Bank Marketing, bank-additional-full (CC BY 4.0).", "41,188 real telemarketing contacts of a Portuguese retail bank, May 2008 - Nov 2010.",
                  "The file has no year: the ETL rebuilds it from the date-ordered rows and asserts the documented range.",
                  "Call duration is reported as an outcome only, never as a targeting input."]),
        ("Model & DAX", ["Star schema: Contacts and monthly Macro facts with Client, Date, Weekday, Attempt Band and Prior Outcome dimensions.",
                         "Field parameter 'Segment by' (8 dimensions) and a what-if 'Attempt Cap' table driving an exact simulation.",
                         "DAX user-defined function Lift; documented measures in display folders."]),
    ], [("Conversion rate", "clients who subscribed / clients contacted."), ("Lift", "segment conversion / campaign average (1.00x = average)."),
        ("Calls per subscription", "all calls made / subscriptions: the cost side of the campaign."),
        ("Calls saved at cap", "sum of calls beyond the cap; exact because every client's call count is known."),
        ("Subscriptions lost at cap", "subscribers whose converting (last) call came after the cap."), ("Euribor 3M", "3-month interbank rate, monthly average (%).")],
        "Built by Mohamed Ayman, Senior Data Scientist  ·  github.com/ayman-stat  ·  mohamedaymanstat.vercel.app")

    r.write()
    write_pbip(os.path.join(ROOT, f"{NAME}.pbip"), f"{NAME}.Report")


if __name__ == "__main__":
    build()
    print("report written")
