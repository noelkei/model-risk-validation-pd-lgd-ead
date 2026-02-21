# src/data/split.py
from src.config import DATE_COL, TRAIN_START, TRAIN_END, OOT_START, OOT_END

def time_split(df):
    df = df.dropna(subset=[DATE_COL]).copy()

    train = df[(df[DATE_COL] >= TRAIN_START) & (df[DATE_COL] <= TRAIN_END)].copy()
    oot   = df[(df[DATE_COL] >= OOT_START) & (df[DATE_COL] <= OOT_END)].copy()

    return train, oot