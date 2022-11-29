from sklearn.model_selection import train_test_split
import pandas as pd


def train_test_file():
    CSV_FILE = 'trip.csv'
    df = pd.read_csv(CSV_FILE)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df
