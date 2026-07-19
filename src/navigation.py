import streamlit as st


def show_navigation():
    with st.sidebar:
        st.markdown("# 🎵 MusicCluster AI")
        st.caption("Discover hidden patterns in music")

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "📂 Dataset",
                "🤖 Clustering",
                "📊 Analytics",
                "🎧 Similar Songs"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        st.caption("🎵 K-Means Music Clustering")
        st.caption("Built with Python & Machine Learning")

    return page