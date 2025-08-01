def category_wise_count(df):
    if "Category" not in df.columns:
        return df
    return df["Category"].value_counts().reset_index().rename(columns={"index": "Category", "Category": "Ticket Count"})
