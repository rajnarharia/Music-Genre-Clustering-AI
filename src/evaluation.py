from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_data


def find_best_clusters():

    df = load_data()

    features = [
        "bpm",
        "energy",
        "danceability",
        "loudness",
        "speechiness"
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = 2
    best_score = -1

    print("\nK-Means Clustering Evaluation")
    print("--------------------------------")

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        print(f"K = {k} | Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print("\nBest Clustering Result")
    print("--------------------------------")
    print(f"Best K: {best_k}")
    print(f"Best Silhouette Score: {best_score:.4f}")

    return best_k, best_score


if __name__ == "__main__":
    find_best_clusters()