import os
import sys

import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# =========================================================
# ADD SRC FOLDER TO PATH
# =========================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)


# =========================================================
# PROJECT IMPORTS
# =========================================================

from clustering import perform_clustering
from similarity import find_similar_songs
from navigation import show_navigation
from evaluation import find_best_k
from cluster_profiles import (
    generate_cluster_profiles,
    profiles_to_dataframe
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Music Genre Clustering AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CSS
# =========================================================

def load_css(file_path):

    try:

        with open(
            file_path,
            encoding="utf-8"
        ) as file:

            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )

    except FileNotFoundError:

        pass


load_css(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "style.css"
    )
)


# =========================================================
# NAVIGATION
# =========================================================

page = show_navigation()


# =========================================================
# LOAD CLUSTERED DATA
# =========================================================

@st.cache_data
def load_clustered_data(n_clusters=2):

    df, X_scaled, model = perform_clustering(
        n_clusters=n_clusters
    )

    return df, X_scaled, model


# =========================================================
# FIND BEST K
# =========================================================

@st.cache_data
def get_best_k():

    best_k, best_score, results = find_best_k(
        min_k=2,
        max_k=10
    )

    return best_k, best_score, results


# =========================================================
# PCA
# =========================================================

def add_pca_columns(df, X_scaled):

    df = df.copy()

    pca = PCA(
        n_components=2
    )

    pca_result = pca.fit_transform(
        X_scaled
    )

    df["PCA1"] = pca_result[:, 0]

    df["PCA2"] = pca_result[:, 1]

    df["cluster_label"] = (
        "Cluster "
        + df["cluster"].astype(str)
    )

    return df


# =========================================================
# DEFAULT DATA
# =========================================================

best_k, best_score, evaluation_results = (
    get_best_k()
)

df, X_scaled, model = load_clustered_data(
    best_k
)

df = add_pca_columns(
    df,
    X_scaled
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title(
        "🎵 Music Genre Clustering AI"
    )

    st.subheader(
        "Discover Hidden Patterns in Music "
        "Using Machine Learning"
    )

    st.markdown(
        """
        This intelligent music analytics platform uses
        **K-Means Clustering** to automatically discover
        groups of similar songs.

        Songs are grouped according to their numerical
        audio characteristics instead of simply relying
        on predefined genre labels.

        The machine learning model analyzes:

        **BPM • Energy • Danceability • Loudness • Speechiness**
        """
    )

    st.write("")

    # =====================================================
    # MAIN METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎵 Total Songs",
            len(df)
        )

    with col2:

        st.metric(
            "🤖 Recommended Clusters",
            best_k
        )

    with col3:

        st.metric(
            "🎼 Genres",
            df["genre"].nunique()
        )

    with col4:

        st.metric(
            "🎤 Artists",
            df["artist_name"].nunique()
        )

    st.success(
        f"🧠 AI Recommendation: K = {best_k} "
        f"achieved the highest Silhouette Score "
        f"of {best_score:.4f}."
    )

    st.divider()

    # =====================================================
    # FEATURES
    # =====================================================

    st.header(
        "✨ Platform Features"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 📂 Dataset Explorer

            Explore songs, artists,
            genres and audio features
            from the music dataset.
            """
        )

    with col2:

        st.info(
            """
            ### 🤖 Dynamic Clustering

            Experiment with different
            values of K and discover
            hidden music groups.
            """
        )

    with col3:

        st.info(
            """
            ### 🎧 Similar Songs

            Discover songs with similar
            musical characteristics using
            machine learning.
            """
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 🧠 Best-K Detection

            Automatically evaluates
            multiple cluster values using
            the Silhouette Score.
            """
        )

    with col2:

        st.info(
            """
            ### 🎼 Cluster Profiles

            Automatically describes the
            musical characteristics of
            discovered clusters.
            """
        )

    with col3:

        st.info(
            """
            ### 📥 Export Results

            Download the clustered music
            dataset directly as a CSV file.
            """
        )

    st.divider()

    # =====================================================
    # ML PIPELINE
    # =====================================================

    st.header(
        "⚙️ Machine Learning Pipeline"
    )

    st.markdown(
        """
        ### 1️⃣ Music Dataset

        ↓

        ### 2️⃣ Audio Feature Selection

        ↓

        ### 3️⃣ Feature Scaling with StandardScaler

        ↓

        ### 4️⃣ Automatic Best-K Evaluation

        ↓

        ### 5️⃣ K-Means Clustering

        ↓

        ### 6️⃣ Cluster Profile Generation

        ↓

        ### 7️⃣ PCA Visualization

        ↓

        ### 8️⃣ Similar Song Discovery
        """
    )


# =========================================================
# DATASET PAGE
# =========================================================

elif page == "📂 Dataset":

    st.title(
        "📂 Music Dataset Explorer"
    )

    st.markdown(
        """
        Explore, search and download the music
        dataset used by the clustering model.
        """
    )

    # =====================================================
    # DATASET METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎵 Songs",
            len(df)
        )

    with col2:

        st.metric(
            "🎤 Artists",
            df["artist_name"].nunique()
        )

    with col3:

        st.metric(
            "🎼 Genres",
            df["genre"].nunique()
        )

    with col4:

        st.metric(
            "🤖 Clusters",
            df["cluster"].nunique()
        )

    st.divider()

    # =====================================================
    # SEARCH
    # =====================================================

    search = st.text_input(
        "🔍 Search Song or Artist",
        placeholder=(
            "Enter a song or artist name..."
        )
    )

    filtered_df = df.copy()

    if search:

        filtered_df = df[
            df["track_name"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            df["artist_name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.caption(
        f"Showing {len(filtered_df)} songs"
    )

    display_columns = [
        "track_name",
        "artist_name",
        "genre",
        "year",
        "bpm",
        "energy",
        "danceability",
        "loudness",
        "speechiness",
        "cluster"
    ]

    st.dataframe(
        filtered_df[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # CSV DOWNLOAD
    # =====================================================

    csv_data = filtered_df[
        display_columns
    ].to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="📥 Download Dataset as CSV",
        data=csv_data,
        file_name=(
            "music_genre_clustered_dataset.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# CLUSTERING PAGE
# =========================================================

elif page == "🤖 Clustering":

    st.title(
        "🤖 Interactive Music Clustering Lab"
    )

    st.markdown(
        """
        Experiment with K-Means clustering and
        discover hidden patterns within the music
        dataset.
        """
    )

    # =====================================================
    # AI RECOMMENDATION
    # =====================================================

    st.success(
        f"🧠 Recommended K: **{best_k}** | "
        f"Best Silhouette Score: "
        f"**{best_score:.4f}**"
    )

    st.divider()

    # =====================================================
    # CONFIGURATION
    # =====================================================

    st.subheader(
        "⚙️ Configure Clustering"
    )

    selected_k = st.slider(
        "Select Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=best_k,
        step=1
    )

    if selected_k == best_k:

        st.info(
            "✨ You are currently using the "
            "AI-recommended number of clusters."
        )

    run_clustering = st.button(
        "🚀 Run AI Clustering",
        use_container_width=True
    )

    if run_clustering:

        with st.spinner(
            "Analyzing songs and creating "
            "music clusters..."
        ):

            dynamic_df, dynamic_X, dynamic_model = (
                load_clustered_data(
                    selected_k
                )
            )

            dynamic_df = add_pca_columns(
                dynamic_df,
                dynamic_X
            )

            score = silhouette_score(
                dynamic_X,
                dynamic_df["cluster"]
            )

            st.session_state[
                "clustering_df"
            ] = dynamic_df

            st.session_state[
                "clustering_X"
            ] = dynamic_X

            st.session_state[
                "clustering_model"
            ] = dynamic_model

            st.session_state[
                "selected_k"
            ] = selected_k

            st.session_state[
                "silhouette_score"
            ] = score

        st.success(
            "Clustering completed successfully!"
        )

    # =====================================================
    # CURRENT RESULTS
    # =====================================================

    if "clustering_df" in st.session_state:

        cluster_df = st.session_state[
            "clustering_df"
        ]

        cluster_X = st.session_state[
            "clustering_X"
        ]

        cluster_model = st.session_state[
            "clustering_model"
        ]

        current_k = st.session_state[
            "selected_k"
        ]

        current_score = st.session_state[
            "silhouette_score"
        ]

    else:

        cluster_df = df

        cluster_X = X_scaled

        cluster_model = model

        current_k = best_k

        current_score = best_score

    st.divider()

    # =====================================================
    # PERFORMANCE
    # =====================================================

    st.subheader(
        "📈 Clustering Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🤖 Algorithm",
            "K-Means"
        )

    with col2:

        st.metric(
            "🔵 Clusters",
            current_k
        )

    with col3:

        st.metric(
            "📊 Silhouette Score",
            f"{current_score:.4f}"
        )

    with col4:

        st.metric(
            "⚙️ Inertia",
            f"{cluster_model.inertia_:.2f}"
        )

    if current_k == best_k:

        st.success(
            "🏆 This configuration matches "
            "the recommended Best-K result."
        )

    else:

        score_difference = (
            best_score
            - current_score
        )

        if score_difference > 0:

            st.warning(
                f"The recommended K = {best_k} "
                f"has a higher Silhouette Score "
                f"by {score_difference:.4f}."
            )

    st.divider()

    # =====================================================
    # K EVALUATION CHART
    # =====================================================

    st.header(
        "📈 Automatic Best-K Evaluation"
    )

    evaluation_df = {
        "K": [
            result["k"]
            for result
            in evaluation_results
        ],
        "Silhouette Score": [
            result["silhouette_score"]
            for result
            in evaluation_results
        ]
    }

    evaluation_fig = px.line(
        evaluation_df,
        x="K",
        y="Silhouette Score",
        markers=True,
        title=(
            "Silhouette Score for "
            "Different Values of K"
        )
    )

    st.plotly_chart(
        evaluation_fig,
        use_container_width=True
    )

    st.caption(
        f"🏆 Best result: K = {best_k} "
        f"with Silhouette Score "
        f"{best_score:.4f}"
    )

    st.divider()

    # =====================================================
    # CLUSTER DISTRIBUTION
    # =====================================================

    st.header(
        "📊 Cluster Distribution"
    )

    cluster_counts = (
        cluster_df[
            "cluster"
        ]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Songs"
    ]

    cluster_counts[
        "Cluster"
    ] = (
        "Cluster "
        + cluster_counts[
            "Cluster"
        ].astype(str)
    )

    cluster_fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Songs",
        text="Songs",
        title=(
            "Number of Songs "
            "in Each Cluster"
        )
    )

    cluster_fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        cluster_fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # CLUSTER PROFILES
    # =====================================================

    st.header(
        "🎼 AI-Generated Cluster Profiles"
    )

    st.markdown(
        """
        Each profile is automatically generated
        by comparing the average audio features
        of each cluster with the overall dataset.
        """
    )

    profiles = generate_cluster_profiles(
        cluster_df
    )

    for cluster_id, profile in profiles.items():

        with st.expander(
            f"🎵 Cluster {cluster_id} — "
            f"{profile['name']}",
            expanded=True
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🎵 Songs",
                    profile[
                        "song_count"
                    ]
                )

                st.metric(
                    "🎚️ Average BPM",
                    profile[
                        "bpm"
                    ]
                )

            with col2:

                st.metric(
                    "⚡ Energy",
                    profile[
                        "energy"
                    ]
                )

                st.metric(
                    "💃 Danceability",
                    profile[
                        "danceability"
                    ]
                )

            with col3:

                st.metric(
                    "🔊 Loudness",
                    profile[
                        "loudness"
                    ]
                )

                st.metric(
                    "🎙️ Speechiness",
                    profile[
                        "speechiness"
                    ]
                )

    st.divider()

    # =====================================================
    # PCA VISUALIZATION
    # =====================================================

    st.header(
        "🧠 Interactive PCA Visualization"
    )

    st.markdown(
        """
        PCA converts the five audio features into
        two dimensions so the discovered music
        clusters can be visualized interactively.
        """
    )

    pca_fig = px.scatter(
        cluster_df,
        x="PCA1",
        y="PCA2",
        color="cluster_label",
        hover_name="track_name",
        hover_data={
            "artist_name": True,
            "genre": True,
            "bpm": True,
            "energy": True,
            "danceability": True,
            "loudness": True,
            "speechiness": True,
            "cluster": True,
            "PCA1": False,
            "PCA2": False
        },
        title=(
            f"K-Means Music Clusters "
            f"(K = {current_k})"
        )
    )

    pca_fig.update_layout(
        height=650,
        legend_title="Music Cluster"
    )

    st.plotly_chart(
        pca_fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # CLUSTER EXPLORER
    # =====================================================

    st.header(
        "🎵 Explore Music Clusters"
    )

    selected_cluster = st.selectbox(
        "Select a Cluster",
        sorted(
            cluster_df[
                "cluster"
            ].unique()
        )
    )

    selected_profile = profiles[
        int(selected_cluster)
    ]

    st.info(
        f"🎼 Cluster {selected_cluster}: "
        f"**{selected_profile['name']}**"
    )

    cluster_songs = cluster_df[
        cluster_df[
            "cluster"
        ] == selected_cluster
    ]

    st.caption(
        f"This cluster contains "
        f"{len(cluster_songs)} songs."
    )

    cluster_display_columns = [
        "track_name",
        "artist_name",
        "genre",
        "bpm",
        "energy",
        "danceability",
        "loudness",
        "speechiness"
    ]

    st.dataframe(
        cluster_songs[
            cluster_display_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # DOWNLOAD CLUSTER
    # =====================================================

    cluster_csv = cluster_songs[
        cluster_display_columns
    ].to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label=(
            f"📥 Download Cluster "
            f"{selected_cluster} as CSV"
        ),
        data=cluster_csv,
        file_name=(
            f"music_cluster_"
            f"{selected_cluster}.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # DOWNLOAD COMPLETE CLUSTERING RESULTS
    # =====================================================

    st.header(
        "📥 Export Clustering Results"
    )

    export_columns = [
        "track_name",
        "artist_name",
        "genre",
        "year",
        "bpm",
        "energy",
        "danceability",
        "loudness",
        "speechiness",
        "cluster"
    ]

    complete_csv = cluster_df[
        export_columns
    ].to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label=(
            "📥 Download Complete "
            "Clustered Dataset"
        ),
        data=complete_csv,
        file_name=(
            f"music_clustering_k"
            f"{current_k}.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    st.title(
        "📊 Music Analytics Dashboard"
    )

    st.markdown(
        """
        Explore patterns and relationships between
        different musical characteristics.
        """
    )

    # =====================================================
    # FILTER
    # =====================================================

    st.subheader(
        "🔍 Analytics Filters"
    )

    genre_options = [
        "All Genres"
    ] + sorted(
        df["genre"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_genre = st.selectbox(
        "Select Genre",
        genre_options
    )

    analytics_df = df.copy()

    if selected_genre != "All Genres":

        analytics_df = df[
            df["genre"]
            == selected_genre
        ]

    st.divider()

    # =====================================================
    # TOP GENRES
    # =====================================================

    st.header(
        "🎼 Top Music Genres"
    )

    genre_counts = (
        analytics_df[
            "genre"
        ]
        .value_counts()
        .head(10)
        .reset_index()
    )

    genre_counts.columns = [
        "Genre",
        "Songs"
    ]

    genre_fig = px.bar(
        genre_counts,
        x="Songs",
        y="Genre",
        orientation="h",
        title="Top Music Genres"
    )

    st.plotly_chart(
        genre_fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # ENERGY VS DANCEABILITY
    # =====================================================

    st.header(
        "⚡ Energy vs Danceability"
    )

    energy_fig = px.scatter(
        analytics_df,
        x="energy",
        y="danceability",
        color="cluster_label",
        hover_name="track_name",
        hover_data=[
            "artist_name",
            "genre",
            "bpm"
        ],
        title=(
            "Energy vs Danceability "
            "by Music Cluster"
        )
    )

    st.plotly_chart(
        energy_fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # BPM DISTRIBUTION
    # =====================================================

    st.header(
        "🎚️ BPM Distribution"
    )

    bpm_fig = px.histogram(
        analytics_df,
        x="bpm",
        color="cluster_label",
        title=(
            "Tempo Distribution Across "
            "Music Clusters"
        )
    )

    st.plotly_chart(
        bpm_fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # ENERGY DISTRIBUTION
    # =====================================================

    st.header(
        "⚡ Energy Distribution"
    )

    energy_hist = px.histogram(
        analytics_df,
        x="energy",
        color="cluster_label",
        title=(
            "Energy Distribution Across "
            "Music Clusters"
        )
    )

    st.plotly_chart(
        energy_hist,
        use_container_width=True
    )


# =========================================================
# SIMILAR SONGS PAGE
# =========================================================

elif page == "🎧 Similar Songs":

    st.title(
        "🎧 AI Similar Song Finder"
    )

    st.markdown(
        """
        Select a song and discover other songs
        with similar audio characteristics.
        """
    )

    st.divider()

    # =====================================================
    # SELECT SONG
    # =====================================================

    selected_song = st.selectbox(
        "🎵 Select Your Song",
        sorted(
            df[
                "track_name"
            ].unique()
        )
    )

    song_info = df[
        df[
            "track_name"
        ] == selected_song
    ].iloc[0]

    st.subheader(
        "🎶 Selected Song"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎤 Artist",
            song_info[
                "artist_name"
            ]
        )

    with col2:

        st.metric(
            "🎼 Genre",
            song_info[
                "genre"
            ]
        )

    with col3:

        st.metric(
            "🤖 Cluster",
            int(
                song_info[
                    "cluster"
                ]
            )
        )

    with col4:

        st.metric(
            "🎚️ BPM",
            song_info[
                "bpm"
            ]
        )

    st.divider()

    # =====================================================
    # AUDIO FEATURES
    # =====================================================

    st.subheader(
        "🎛️ Audio Characteristics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "⚡ Energy",
            song_info[
                "energy"
            ]
        )

    with col2:

        st.metric(
            "💃 Danceability",
            song_info[
                "danceability"
            ]
        )

    with col3:

        st.metric(
            "🔊 Loudness",
            song_info[
                "loudness"
            ]
        )

    with col4:

        st.metric(
            "🎙️ Speechiness",
            song_info[
                "speechiness"
            ]
        )

    st.write("")

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    if st.button(
        "✨ Find Similar Songs",
        use_container_width=True
    ):

        with st.spinner(
            "Finding similar songs..."
        ):

            recommendations = (
                find_similar_songs(
                    selected_song,
                    top_n=5
                )
            )

        if (
            recommendations
            is not None
            and not recommendations.empty
        ):

            st.success(
                f"Top 5 songs similar to "
                f"{selected_song}"
            )

            recommendations = (
                recommendations.copy()
            )

            recommendations[
                "similarity_percentage"
            ] = (
                recommendations[
                    "similarity_score"
                ]
                * 100
            ).round(2)

            result_table = (
                recommendations[
                    [
                        "track_name",
                        "artist_name",
                        "genre",
                        "cluster",
                        "similarity_percentage"
                    ]
                ]
                .rename(
                    columns={
                        "track_name":
                            "Song",

                        "artist_name":
                            "Artist",

                        "genre":
                            "Genre",

                        "cluster":
                            "Cluster",

                        "similarity_percentage":
                            "Similarity %"
                    }
                )
            )

            st.dataframe(
                result_table,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No similar songs found."
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("### 🎵 Music Genre Clustering AI")

st.write(
    "Intelligent Music Discovery Using "
    "Unsupervised Machine Learning"
)

st.caption(
    "Python • Streamlit • K-Means • "
    "Scikit-Learn • PCA • Plotly"
)

st.write("Developed by **Raj Narharia**")