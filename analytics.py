import pandas as pd


def section_avg_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Return average sentiment score by section."""
    if "section" not in df.columns:
        return pd.DataFrame(columns=["section", "avg_sentiment"])
    return (
        df.groupby("section")["sentiment_score"]
        .mean()
        .reset_index(name="avg_sentiment")
        .sort_values("avg_sentiment", ascending=False)
    )


def section_label_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return count of sentiment labels per section."""
    if "section" not in df.columns or "sentiment_label" not in df.columns:
        return pd.DataFrame(columns=["section", "sentiment_label", "count"])
    return (
        df.groupby(["section", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
