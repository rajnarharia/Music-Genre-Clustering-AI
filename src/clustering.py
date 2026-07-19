from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_data


def perform_clustering(n_clusters=2):

    # Load dataset
    df = load_data()

    # Features used for clustering
    features = [
        "bpm",
        "energy",
        "danceability",
        "loudness",
        "speechiness"
    ]

    X = df[features]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(X_scaled)

    return df, X_scaled, model


if __name__ == "__main__":

    df, X_scaled, model = perform_clustering()

    print("Clustering completed successfully!")

    print("\nCluster Distribution:")
    print(df["cluster"].value_counts().sort_index())

    print("\nSample Results:")
    print(
        df[
            [
                "track_name",
                "artist_name",
                "genre",
                "cluster"
            ]
        ].head(15)
    )