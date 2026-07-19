from sklearn.metrics.pairwise import cosine_similarity

from clustering import perform_clustering


def find_similar_songs(song_name, top_n=5):

    df, X_scaled, model = perform_clustering()

    # Case-insensitive song search
    matches = df[
        df["track_name"].str.lower() == song_name.lower()
    ]

    if matches.empty:
        print(f'Song "{song_name}" not found.')
        return None

    song_index = matches.index[0]
    song_position = df.index.get_loc(song_index)

    # Get selected song's cluster
    selected_cluster = df.loc[song_index, "cluster"]

    # Calculate similarity
    similarities = cosine_similarity(
        [X_scaled[song_position]],
        X_scaled
    )[0]

    result = df.copy()
    result["similarity_score"] = similarities

    # Keep songs from same cluster
    result = result[
        result["cluster"] == selected_cluster
    ]

    # Remove selected song
    result = result[
        result.index != song_index
    ]

    # Get most similar songs
    result = result.sort_values(
        "similarity_score",
        ascending=False
    ).head(top_n)

    return result


if __name__ == "__main__":

    song_name = input("Enter song name: ")

    recommendations = find_similar_songs(
        song_name,
        top_n=5
    )

    if recommendations is not None:

        print("\nTop 5 Similar Songs")
        print("-----------------------------")

        print(
            recommendations[
                [
                    "track_name",
                    "artist_name",
                    "genre",
                    "cluster",
                    "similarity_score"
                ]
            ].to_string(index=False)
        )