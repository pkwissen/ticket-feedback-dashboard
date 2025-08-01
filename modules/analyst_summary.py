import pandas as pd

def generate_analyst_summary(df):
    analyst_col = "Analyst who closed the ticket"
    if analyst_col not in df.columns:
        return pd.DataFrame()

    # Determine month label source: prefer existing "Month" column, else derive from "Date"
    if "Month" in df.columns and df["Month"].notna().any():
        # Assume Month is already in format like "Jan,25" or something close; normalize if needed
        df["month_label"] = df["Month"].astype(str)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["month_label"] = df["Date"].dt.strftime("%b,%y")
    else:
        # Cannot get monthly breakdown; fallback to total counts
        summary = (
            df[analyst_col]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Analyst who closed the ticket", analyst_col: "Grand Total"})
        )
        return summary

    # Pivot count of tickets per analyst per month
    pivot = pd.pivot_table(
        df,
        index=analyst_col,
        columns="month_label",
        values="Ticket No.",  # any column that exists per ticket; counts unique occurrences
        aggfunc="count",
        fill_value=0,
    )

    # Chronologically sort the month columns if possible
    def _parse_label(lbl):
        try:
            return pd.to_datetime(lbl, format="%b,%y")
        except Exception:
            return pd.Timestamp.max

    ordered_months = sorted(pivot.columns.tolist(), key=_parse_label)
    pivot = pivot[ordered_months]

    # Grand Total
    pivot["Grand Total"] = pivot.sum(axis=1)

    # Replace zeros with empty strings except Grand Total
    display = pivot.copy()
    for col in display.columns:
        if col != "Grand Total":
            display[col] = display[col].replace(0, "")

    # Reset index and ensure analyst column name
    display = display.reset_index().rename(columns={analyst_col: "Analyst who closed the ticket"})

    return display
