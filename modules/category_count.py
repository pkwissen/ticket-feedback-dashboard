def category_wise_count(df):
    if "Category" not in df.columns:
        return df
    # Count tickets per category, keep original column name "Category"
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Category", "Ticket Count"]  # "Category" stays, add "Ticket Count"
    return counts
