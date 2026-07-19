# 🎵 Music Genre Clustering AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly)

## 🌟 Overview

Intelligent Music Discovery Using Unsupervised Machine Learning.

This project is a premium, interactive web application built with **Streamlit** that utilizes **K-Means Clustering** to automatically discover hidden patterns and groups of similar songs based on their audio characteristics (BPM, Energy, Danceability, Loudness, Speechiness), rather than relying solely on existing genre labels.

Featuring a beautiful, modern **Glassmorphism** design with vibrant animated gradients and micro-animations, this platform offers a deeply engaging user experience for data exploration.

## ✨ Key Features

- **📂 Dataset Explorer**: Search and filter through the entire music dataset.
- **🤖 Interactive Clustering**: Dynamically configure the number of clusters (K) and run K-Means interactively. Enjoy celebratory visual effects upon successful clustering!
- **🧠 PCA Visualization**: Visualize high-dimensional audio features in an interactive 2D space using Principal Component Analysis.
- **🎧 AI Similar Song Finder**: Select any track and let the AI find the top 5 most similar songs based on acoustic properties.
- **📊 Interactive Analytics Dashboard**: Explore the distribution of tempos, energy levels, and genres across your generated clusters using beautiful Plotly charts.
- **💎 Premium UI/UX**: Stunning frosted glass panels, animated gradient backgrounds, custom 'Outfit' typography, and delightful micro-interactions (like balloons and snow effects!).

## ⚙️ Tech Stack

- **Frontend**: Streamlit, HTML/CSS (Glassmorphism theme)
- **Machine Learning**: Scikit-Learn (K-Means, PCA, StandardScaler)
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly Express

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tarunkrsaini/Music-Genre-Clustering-AI.git
   cd Music-Genre-Clustering-AI
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Enjoy!**
   The application should open automatically in your web browser at `http://localhost:8501`.

## 🧠 Machine Learning Pipeline

1. **Audio Feature Selection**: Extract relevant audio properties.
2. **Feature Scaling**: Apply `StandardScaler` to normalize the data.
3. **K-Means Clustering**: Unsupervised grouping of similar tracks.
4. **Silhouette Score Evaluation**: Assess the quality of the clusters.
5. **Dimensionality Reduction**: Use `PCA` to project features into 2D space for visualization.
6. **Similar Song Discovery**: Compute similarity metrics to recommend tracks.

---
*Created with ❤️ by Tarun Kr Saini*
