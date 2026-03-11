from __future__ import annotations
import pandas as pd

FIXED_DATE_HOLIDAYS = [
    ("New Year's Day", 1, 1),
    ("Juneteenth", 6, 19),
    ("Independence Day", 7, 4),
    ("Veterans Day", 11, 11),
    ("Christmas Day", 12, 25),
]

# Weekday: Monday=0 ... Sunday=6 (pandas convention)
RULE_HOLIDAYS = [
    ("MLK Day (3rd Mon Jan)",      "nth_weekday", {"month": 1,  "weekday": 0, "n": 3}),
    ("Presidents Day (3rd Mon Feb)","nth_weekday", {"month": 2,  "weekday": 0, "n": 3}),
    ("Memorial Day (last Mon May)", "last_weekday",{"month": 5,  "weekday": 0}),
    ("Labor Day (1st Mon Sep)",     "nth_weekday", {"month": 9,  "weekday": 0, "n": 1}),
    ("Indigenous Peoples Day (2nd Mon Oct)", "nth_weekday", {"month": 10, "weekday": 0, "n": 2}),
    ("Thanksgiving (4th Thu Nov)",  "nth_weekday", {"month": 11, "weekday": 3, "n": 4}),
]

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    # shift forward to desired weekday
    delta = (weekday - first.weekday()) % 7
    day = 1 + delta + (n - 1) * 7
    return pd.Timestamp(year=year, month=month, day=day)

def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    last = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1))
    delta = (last.weekday() - weekday) % 7
    return last - pd.Timedelta(days=delta)

def build_free_days(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Returns a dataframe of free parking days for SDOT paid parking:
      - all Sundays
      - listed holidays
      - if a FIXED-DATE holiday falls on Sunday => also add Monday after (observed)
    """
    rows = []

    for year in range(start_year, end_year + 1):
        # All Sundays
        d0 = pd.Timestamp(year=year, month=1, day=1)
        d1 = pd.Timestamp(year=year, month=12, day=31)
        sundays = pd.date_range(d0, d1, freq="W-SUN")
        for d in sundays:
            rows.append({"date": d.date().isoformat(), "name": "Sunday", "kind": "sunday", "observed_of": ""})

        # Fixed-date holidays (*)
        for name, m, day in FIXED_DATE_HOLIDAYS:
            d = pd.Timestamp(year=year, month=m, day=day)
            rows.append({"date": d.date().isoformat(), "name": name, "kind": "holiday_fixed", "observed_of": ""})

            # SDOT rule: if the date falls on Sunday, Monday that follows is also free
            if d.weekday() == 6:  # Sunday
                obs = d + pd.Timedelta(days=1)  # Monday
                rows.append({
                    "date": obs.date().isoformat(),
                    "name": f"{name} (Observed)",
                    "kind": "holiday_observed",
                    "observed_of": d.date().isoformat(),
                })

        # Rule-based holidays (already land on Mon/Thu by definition)
        for name, rule, params in RULE_HOLIDAYS:
            if rule == "nth_weekday":
                d = _nth_weekday(year, params["month"], params["weekday"], params["n"])
            else:
                d = _last_weekday(year, params["month"], params["weekday"])
            rows.append({"date": d.date().isoformat(), "name": name, "kind": "holiday_rule", "observed_of": ""})

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "name"]).sort_values("date").reset_index(drop=True)
    return df

def write_free_days_csv(path: str, start_year: int, end_year: int) -> None:
    df = build_free_days(start_year, end_year)
    df.to_csv(path, index=False)