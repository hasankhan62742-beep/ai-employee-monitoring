"""Reporting: productivity scores, smart alerts, charts, Excel export."""
import pandas as pd


def _format_employee_id(x):
    """Format one employee_id value for display, or None to skip it.

    The engine writes employee_id as a string: "1", "2", "" for ABSENT
    rows, and "None" when ByteTrack has not assigned an ID yet.
    The old code called int(x) directly, which crashed with
    ValueError on "" / "None" (verified live on Streamlit Cloud,
    2026-11-11) and wiped out the whole dashboard. This helper
    skips blank values and passes non-numeric values through as-is.
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    if s in ("", "nan", "NaN", "None", "none"):
        return None
    try:
        return f"ID {int(float(s))}"
    except (TypeError, ValueError):
        return s


def productivity_by_desk(df):
    """One row per desk. Score = working% minus a phone-usage penalty."""
    rows = []
    for desk, group in df.groupby("desk"):
        total = len(group)
        working = (group["state"] == "WORKING").sum()
        phone = (group["state"] == "PHONE USAGE").sum()
        absent = (group["state"] == "ABSENT").sum()
        score = round(working / total * 100 - phone / total * 50, 1)
        score = max(0, min(100, score))
        if score >= 80:
            grade = "A+ Excellent"
        elif score >= 60:
            grade = "A Good"
        elif score >= 40:
            grade = "B Average"
        else:
            grade = "C Needs Attention"
        employees = group["employee_id"].dropna().unique()
        shown = ", ".join(
            name for name in (_format_employee_id(x) for x in employees)
            if name
        )
        rows.append(
            {
                "Desk": desk,
                "Employee": shown or "-",
                "Total_Sec": total,
                "Working_Sec": int(working),
                "Phone_Sec": int(phone),
                "Absent_Sec": int(absent),
                "Working_%": round(working / total * 100, 1),
                "Productivity_Score": score,
                "Grade": grade,
            }
        )
    return pd.DataFrame(rows)


def productivity_by_employee(df):
    """One row per named employee.

    Note: this score covers only seconds when the employee was
    visible on camera. Absent seconds have no employee name, so
    they are counted in the desk score, not here.
    """
    named = df[df["employee_name"].notna() & (df["employee_name"] != "")]
    rows = []
    for name, group in named.groupby("employee_name"):
        total = len(group)
        working = (group["state"] == "WORKING").sum()
        phone = (group["state"] == "PHONE USAGE").sum()
        score = round(working / total * 100 - phone / total * 50, 1)
        rows.append(
            {
                "Employee": name,
                "Total_Sec": total,
                "Working_Sec": int(working),
                "Phone_Sec": int(phone),
                "Working_%": round(working / total * 100, 1),
                "Score": max(0, min(100, score)),
            }
        )
    return pd.DataFrame(rows)


def find_alerts(df, phone_min_sec=3, absent_min_sec=8):
    """Continuous PHONE USAGE >= phone_min_sec, or ABSENT >= absent_min_sec."""
    alerts = []
    for desk, group in df.groupby("desk"):
        group = group.sort_values("time_sec").reset_index(drop=True)
        current, start, count = None, 0, 0
        for _, row in group.iterrows():
            if row["state"] == current:
                count += 1
            else:
                if current == "PHONE USAGE" and count >= phone_min_sec:
                    alerts.append(
                        [desk, start, start + count - 1,
                         f"PHONE USAGE {count} sec", "HIGH"]
                    )
                if current == "ABSENT" and count >= absent_min_sec:
                    alerts.append(
                        [desk, start, start + count - 1,
                         f"ABSENT {count} sec", "MEDIUM"]
                    )
                current, start, count = row["state"], row["time_sec"], 1
        if current == "PHONE USAGE" and count >= phone_min_sec:
            alerts.append(
                [desk, start, start + count - 1,
                 f"PHONE USAGE {count} sec", "HIGH"]
            )
        if current == "ABSENT" and count >= absent_min_sec:
            alerts.append(
                [desk, start, start + count - 1,
                 f"ABSENT {count} sec", "MEDIUM"]
            )
    return pd.DataFrame(
        alerts, columns=["Desk", "Start_Sec", "End_Sec", "Alert", "Level"]
    )


def save_excel_report(df, output_xlsx):
    """Three sheets: Summary, Detailed_Log, Alerts."""
    summary = productivity_by_desk(df)
    alerts = find_alerts(df)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Detailed_Log", index=False)
        alerts.to_excel(writer, sheet_name="Alerts", index=False)
    return summary, alerts
