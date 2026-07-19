import pandas as pd


def load_data(file_path="data/spotify_tracks.csv"):
    df = pd.read_csv(file_path)

    # Rename dataset columns
    df.columns = [
        "rank",
        "track_name",
        "artist_name",
        "genre",
        "year",
        "bpm",
        "energy",
        "danceability",
        "loudness",
        "speechiness"
    ]

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    return df


if __name__ == "__main__":
    df = load_data()

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nNew Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)