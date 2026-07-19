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
        with open(file_path, encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        pass


load_css("assets/style.css")


# =========================================================
# NAVIGATION
# =========================================================

page = show_navigation()


# =========================================================
# LOAD DEFAULT CLUSTERING
# =========================================================

@st.cache_data
def load_clustered_data(n_clusters=2):

    df, X_scaled, model = perform_clustering(
        n_clusters=n_clusters
    )

    return df, X_scaled, model


# =========================================================
# CREATE PCA DATA
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

df, X_scaled, model = load_clustered_data(2)

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
        "Discover Hidden Patterns in Music Using Machine Learning"
    )

    st.markdown(
        """
        This intelligent music analytics platform uses
        **K-Means Clustering** to automatically discover groups
        of similar songs.

        Songs are grouped according to their audio characteristics
        instead of simply relying on their existing genre labels.

        The machine learning model analyzes:

        **BPM • Energy • Danceability • Loudness • Speechiness**
        """
    )

    st.write("")

    # ================= METRICS =================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎵 Total Songs",
            len(df)
        )

    with col2:

        st.metric(
            "🤖 Music Clusters",
            df["cluster"].nunique()
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

    st.divider()

    # ================= FEATURES =================

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

            Choose the number of clusters
            and run K-Means clustering
            interactively.
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

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 📊 Interactive Analytics

            Explore relationships between
            energy, danceability, BPM
            and music clusters.
            """
        )

    with col2:

        st.info(
            """
            ### 🧠 PCA Visualization

            Visualize high-dimensional
            music features in an interactive
            two-dimensional space.
            """
        )

    with col3:

        st.info(
            """
            ### 📈 Cluster Evaluation

            Evaluate clustering quality
            using the Silhouette Score
            and model inertia.
            """
        )

    st.divider()

    # ================= PIPELINE =================

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

        ### 4️⃣ K-Means Clustering

        ↓

        ### 5️⃣ Silhouette Score Evaluation

        ↓

        ### 6️⃣ PCA Visualization

        ↓

        ### 7️⃣ Similar Song Discovery
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
        Explore and search the complete dataset
        used for music clustering.
        """
    )

    # ================= METRICS =================

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
            "📊 Features",
            5
        )

    st.divider()

    # ================= SEARCH =================

    search = st.text_input(
        "🔍 Search Song or Artist",
        placeholder="Enter song or artist name..."
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

    st.dataframe(
        filtered_df[
            [
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
        ],
        use_container_width=True,
        hide_index=True
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
        Configure and run the K-Means algorithm
        to discover different groups of similar songs.
        """
    )

    st.divider()

    # ================= CONFIGURATION =================

    st.subheader(
        "⚙️ Configure Clustering"
    )

    selected_k = st.slider(
        "Select Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=2,
        step=1
    )

    st.caption(
        "Based on previous evaluation, K = 2 achieved the highest Silhouette Score."
    )

    run_clustering = st.button(
        "🚀 Run AI Clustering",
        use_container_width=True
    )

    if run_clustering:

        with st.spinner(
            "Analyzing songs and creating music clusters..."
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
        st.balloons()

    # ================= GET CURRENT RESULT =================

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

        current_k = 2

        current_score = silhouette_score(
            X_scaled,
            df["cluster"]
        )

    st.divider()

    # ================= MODEL METRICS =================

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

    st.divider()

    # ================= CLUSTER DISTRIBUTION =================

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
        title="Number of Songs in Each Cluster"
    )

    cluster_fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        cluster_fig,
        use_container_width=True
    )

    st.divider()

    # ================= PCA VISUALIZATION =================

    st.header(
        "🧠 Interactive PCA Visualization"
    )

    st.markdown(
        """
        PCA converts the five audio features into
        two dimensions so the discovered music
        clusters can be visualized.
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

    # ================= CLUSTER EXPLORER =================

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

    cluster_songs = cluster_df[
        cluster_df[
            "cluster"
        ] == selected_cluster
    ]

    st.info(
        f"Cluster {selected_cluster} contains "
        f"{len(cluster_songs)} songs."
    )

    st.dataframe(
        cluster_songs[
            [
                "track_name",
                "artist_name",
                "genre",
                "bpm",
                "energy",
                "danceability",
                "loudness",
                "speechiness"
            ]
        ],
        use_container_width=True,
        hide_index=True
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
        Explore patterns and relationships
        between different music characteristics.
        """
    )

    # ================= FILTER =================

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

    # ================= GENRES =================

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

    # ================= ENERGY VS DANCEABILITY =================

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

    # ================= BPM =================

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

    # ================= ENERGY DISTRIBUTION =================

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

    # ================= SELECT SONG =================

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

    st.write("")

    # ================= SONG INFORMATION =================

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

    # ================= AUDIO FEATURES =================

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

    # ================= RECOMMENDATIONS =================

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
            st.snow()

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

st.markdown(
    """
<div style="text-align:center; padding:30px; margin-top:20px; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 20px; backdrop-filter: blur(10px); color:#94A3B8; line-height:2; font-family: 'Outfit', sans-serif;">
    <h3 style="color: #00e5ff; margin-bottom: 5px; font-weight: 800;">🎵 Music Genre Clustering AI</h3>
    <span style="font-size: 1.1em; color: #f8fafc;">Intelligent Music Discovery Using Unsupervised Machine Learning</span><br><br>
    <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 10px;">
        <span style="background: rgba(0, 229, 255, 0.1); color: #00e5ff; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600;">Python</span>
        <span style="background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600;">Streamlit</span>
        <span style="background: rgba(0, 229, 255, 0.1); color: #00e5ff; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600;">K-Means</span>
        <span style="background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600;">Scikit-Learn</span>
        <span style="background: rgba(0, 229, 255, 0.1); color: #00e5ff; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600;">PCA</span>
        <span style="background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600;">Plotly</span>
    </div>
</div>
    """,
    unsafe_allow_html=True
)