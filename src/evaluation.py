from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_preprocessing import load_data
from sklearn.preprocessing import StandardScaler


# Features used for clustering
FEATURES = [
    "bpm",
    "energy",
    "danceability",
    "loudness",
    "speechiness"
]


def prepare_data():
    """
    Load and prepare the dataset for clustering.
    """

    df = load_data()

    X = df[FEATURES]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler


def evaluate_k_range(min_k=2, max_k=10):
    """
    Evaluate different values of K using
    Silhouette Score and Inertia.
    """

    df, X_scaled, scaler = prepare_data()

    results = []

    for k in range(min_k, max_k + 1):

        # K cannot be greater than or equal
        # to the number of samples
        if k >= len(df):
            break

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        results.append(
            {
                "k": k,
                "silhouette_score": silhouette,
                "inertia": model.inertia_
            }
        )

    return results


def find_best_k(min_k=2, max_k=10):
    """
    Automatically find the best number
    of clusters using Silhouette Score.
    """

    results = evaluate_k_range(
        min_k=min_k,
        max_k=max_k
    )

    if not results:
        raise ValueError(
            "Unable to evaluate clustering. "
            "Check the dataset size."
        )

    best_result = max(
        results,
        key=lambda result:
        result["silhouette_score"]
    )

    best_k = best_result["k"]

    best_score = best_result[
        "silhouette_score"
    ]

    return best_k, best_score, results


def print_evaluation():
    """
    Display complete clustering evaluation
    results in the terminal.
    """

    best_k, best_score, results = find_best_k()

    print()
    print("=" * 55)
    print("       MUSIC CLUSTERING - K EVALUATION")
    print("=" * 55)

    print()

    for result in results:

        print(
            f"K = {result['k']:2d}"
            f" | Silhouette Score = "
            f"{result['silhouette_score']:.4f}"
            f" | Inertia = "
            f"{result['inertia']:.2f}"
        )

    print()
    print("=" * 55)
    print("           BEST CLUSTERING RESULT")
    print("=" * 55)

    print(
        f"Recommended Number of Clusters : {best_k}"
    )

    print(
        f"Best Silhouette Score          : "
        f"{best_score:.4f}"
    )

    print("=" * 55)

    return best_k, best_score, results


if __name__ == "__main__":

    print_evaluation()