import pandas as pd


FEATURES = [
    "bpm",
    "energy",
    "danceability",
    "loudness",
    "speechiness"
]


def generate_cluster_profiles(df):
    """
    Generate automatic descriptive profiles
    for each music cluster.
    """

    if "cluster" not in df.columns:
        raise ValueError(
            "The dataframe must contain a 'cluster' column."
        )

    # Calculate overall dataset averages
    overall_means = df[FEATURES].mean()

    profiles = {}

    for cluster_id in sorted(df["cluster"].unique()):

        cluster_data = df[
            df["cluster"] == cluster_id
        ]

        cluster_means = cluster_data[
            FEATURES
        ].mean()

        characteristics = []

        # BPM Profile
        if cluster_means["bpm"] > overall_means["bpm"]:
            characteristics.append("Fast Tempo")
        else:
            characteristics.append("Slow Tempo")

        # Energy Profile
        if cluster_means["energy"] > overall_means["energy"]:
            characteristics.append("High Energy")
        else:
            characteristics.append("Low Energy")

        # Danceability Profile
        if (
            cluster_means["danceability"]
            > overall_means["danceability"]
        ):
            characteristics.append("Highly Danceable")
        else:
            characteristics.append("Less Danceable")

        # Loudness Profile
        if (
            cluster_means["loudness"]
            > overall_means["loudness"]
        ):
            characteristics.append("Louder Sound")
        else:
            characteristics.append("Softer Sound")

        # Speechiness Profile
        if (
            cluster_means["speechiness"]
            > overall_means["speechiness"]
        ):
            characteristics.append("More Vocal / Speech")
        else:
            characteristics.append("More Musical")

        profile_name = " • ".join(
            characteristics[:3]
        )

        profiles[int(cluster_id)] = {
            "cluster": int(cluster_id),
            "name": profile_name,
            "song_count": len(cluster_data),
            "bpm": round(
                cluster_means["bpm"], 2
            ),
            "energy": round(
                cluster_means["energy"], 2
            ),
            "danceability": round(
                cluster_means["danceability"], 2
            ),
            "loudness": round(
                cluster_means["loudness"], 2
            ),
            "speechiness": round(
                cluster_means["speechiness"], 2
            )
        }

    return profiles


def profiles_to_dataframe(profiles):
    """
    Convert cluster profiles dictionary
    into a Pandas DataFrame.
    """

    return pd.DataFrame(
        profiles.values()
    )


def print_cluster_profiles(df):
    """
    Print cluster profiles in the terminal.
    """

    profiles = generate_cluster_profiles(df)

    print()
    print("=" * 65)
    print("                MUSIC CLUSTER PROFILES")
    print("=" * 65)

    for cluster_id, profile in profiles.items():

        print()
        print(
            f"Cluster {cluster_id}: "
            f"{profile['name']}"
        )

        print(
            f"Songs        : {profile['song_count']}"
        )

        print(
            f"Average BPM  : {profile['bpm']}"
        )

        print(
            f"Energy       : {profile['energy']}"
        )

        print(
            f"Danceability : {profile['danceability']}"
        )

        print(
            f"Loudness     : {profile['loudness']}"
        )

        print(
            f"Speechiness  : {profile['speechiness']}"
        )

        print("-" * 65)

    return profiles