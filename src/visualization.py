import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from clustering import perform_clustering


def visualize_clusters():

    df, X_scaled, model = perform_clustering()

    # Reduce 5 features into 2 dimensions
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    # Plot clusters
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        df["PCA1"],
        df["PCA2"],
        c=df["cluster"],
        cmap="viridis",
        s=80,
        alpha=0.8
    )

    plt.title("Music Genre Clustering using K-Means")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.colorbar(scatter, label="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    visualize_clusters()