<div align="center">

# 🎵 Music Genre Clustering AI

### Intelligent Music Discovery Using Unsupervised Machine Learning

**Discover hidden patterns in music and group similar songs based on their audio characteristics.**

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)
![K-Means](https://img.shields.io/badge/K--Means-Clustering-purple?style=for-the-badge)
![PCA](https://img.shields.io/badge/PCA-Visualization-green?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-blueviolet?style=for-the-badge&logo=plotly&logoColor=white)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 🌐 Live Demo

Music Genre Clustering AI is deployed and available online on **Streamlit Community Cloud**.

<div align="center">

### 🚀 [Launch Music Genre Clustering AI](https://music-genre-clustering-ai-yoqfsawhtmpk8ar4sdffoc.streamlit.app)

**No installation required — open the application directly in your browser.**

</div>

---

## 📌 About the Project

**Music Genre Clustering AI** is an interactive machine learning application that discovers natural groups of similar songs using **unsupervised machine learning**.

Instead of simply relying on predefined genre labels, the system analyzes the numerical audio characteristics of songs and uses **K-Means Clustering** to identify hidden patterns within the music dataset.

The project also includes interactive data analytics, PCA-based cluster visualization, dynamic clustering configuration, and a similar-song discovery system.

---

## 🎯 Project Objective

The main objective of this project is to demonstrate how **unsupervised machine learning** can be used to discover similarities between songs without requiring predefined target labels.

The system follows this workflow:

```text
Music Dataset
      ↓
Data Preprocessing
      ↓
Audio Feature Selection
      ↓
Feature Scaling
      ↓
K-Means Clustering
      ↓
Cluster Evaluation
      ↓
PCA Visualization
      ↓
Similar Song Discovery
```

---

## ✨ Key Features

### 🤖 Dynamic Music Clustering

- Groups similar songs using **K-Means Clustering**
- Users can dynamically select the number of clusters
- Supports cluster values from **K = 2 to K = 10**
- Displays real-time clustering results

### 📈 Clustering Evaluation

- Calculates the **Silhouette Score**
- Displays model **Inertia**
- Helps compare clustering quality for different values of K

### 🧠 Interactive PCA Visualization

- Uses **Principal Component Analysis (PCA)**
- Converts multiple audio features into a 2D representation
- Displays interactive music clusters using Plotly
- Hover over individual points to explore song information

### 🎧 Similar Song Finder

- Select any available song from the dataset
- Finds songs from the same musical cluster
- Uses audio-feature similarity
- Calculates similarity using **Cosine Similarity**
- Returns the Top 5 most similar songs

### 📂 Dataset Explorer

- Explore the complete music dataset
- Search songs by track name
- Search songs by artist
- View genres and audio characteristics

### 📊 Interactive Analytics Dashboard

- Genre distribution analysis
- Energy vs Danceability visualization
- BPM distribution
- Energy distribution
- Cluster-based interactive charts

### 🎨 Modern Streamlit Interface

- Premium dark dashboard design
- Responsive metric cards
- Sidebar navigation
- Interactive Plotly visualizations
- Clean and user-friendly interface

---

## 🧠 Machine Learning Approach

The project uses **K-Means Clustering**, an unsupervised machine learning algorithm that groups data points based on their similarity.

The following audio features are currently used for clustering:

| Feature | Description |
|---|---|
| 🎚️ BPM | Tempo of the song |
| ⚡ Energy | Intensity and activity level |
| 💃 Danceability | How suitable the song is for dancing |
| 🔊 Loudness | Overall loudness level |
| 🎙️ Speechiness | Presence of spoken words |

Before clustering, these features are normalized using **StandardScaler** so that features with different numerical ranges contribute fairly to the clustering process.

---

## 📊 Clustering Evaluation Results

The project evaluated multiple values of **K** using the Silhouette Score.

| Number of Clusters (K) | Silhouette Score |
|---:|---:|
| 2 | **0.3764** |
| 3 | 0.2581 |
| 4 | 0.2563 |
| 5 | 0.1855 |
| 6 | 0.1855 |
| 7 | 0.2020 |
| 8 | 0.2099 |
| 9 | 0.2068 |
| 10 | 0.2143 |

Based on the current dataset and selected audio features, **K = 2** achieved the highest Silhouette Score.

> The application still allows users to dynamically experiment with different cluster values from 2 to 10.

---

## 🎧 How Similar Song Discovery Works

When a user selects a song:

1. The selected song is identified in the dataset.
2. Its K-Means cluster is determined.
3. Songs belonging to the same cluster are selected.
4. Audio-feature similarity is calculated using **Cosine Similarity**.
5. The most similar songs are ranked.
6. The Top 5 recommendations are displayed.

This provides a simple machine-learning-based approach to music discovery.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| Scikit-Learn | Machine learning |
| K-Means | Music clustering |
| StandardScaler | Feature normalization |
| PCA | Dimensionality reduction |
| Cosine Similarity | Similar song discovery |
| Plotly | Interactive data visualization |
| Matplotlib | ML visualization |
| Streamlit Community Cloud | Application deployment |
| GitHub | Version control and source hosting |

---

## 📁 Project Structure

```text
MusicGenreClustering/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── spotify_tracks.csv
│
├── assets/
│   └── style.css
│
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── clustering.py
    ├── evaluation.py
    ├── visualization.py
    ├── similarity.py
    └── navigation.py
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rajnarharia/Music-Genre-Clustering-AI.git
```

### 2. Navigate to the Project

```bash
cd Music-Genre-Clustering-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will typically be available locally at:

```text
http://localhost:8501
```

---

## 🖥️ Application Modules

The application contains five main sections:

- 🏠 **Home** — Project overview and ML workflow
- 📂 **Dataset** — Explore and search music data
- 🤖 **Clustering** — Configure and run K-Means clustering
- 📊 **Analytics** — Explore interactive music visualizations
- 🎧 **Similar Songs** — Discover musically similar tracks

---

## 🌐 Deployment

Music Genre Clustering AI is deployed on **Streamlit Community Cloud** and can be accessed directly from any modern web browser.

### 🚀 Live Application

**[Launch Music Genre Clustering AI](https://music-genre-clustering-ai-yoqfsawhtmpk8ar4sdffoc.streamlit.app)**

The deployed application allows users to explore the music dataset, experiment with K-Means clustering, visualize clusters using PCA, analyze music characteristics, and discover similar songs directly from their browser.

---

## 🔮 Future Improvements

Future versions of the project can include:

- 🎵 Integration with the Spotify API
- 🔍 Automatic optimal-K detection using the Elbow Method
- 🧠 DBSCAN and Hierarchical Clustering comparison
- 🎧 Larger real-world Spotify datasets
- 📊 Advanced cluster profiling
- 🤖 AI-generated explanations of music clusters
- ❤️ Favourite song functionality
- 🎼 Personalized playlist generation

---

## 👨‍💻 Developer

<div align="center">

### Raj Narharia

**B.Tech Artificial Intelligence Student | AI & Machine Learning Enthusiast**

Passionate about building practical projects using  
**Artificial Intelligence, Machine Learning and Data Science.**

<br>

[![GitHub](https://img.shields.io/badge/GitHub-rajnarharia-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rajnarharia)

<br>

**Developed with ❤️ by Raj Narharia**

</div>

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a **⭐ Star**.

Contributions, suggestions, feedback, and improvements are always welcome.

---

<div align="center">

# 🎵 Music Genre Clustering AI

### Intelligent Music Discovery Using Unsupervised Machine Learning

**Python • Streamlit • K-Means • Scikit-Learn • PCA • Plotly**

<br>

**© 2026 Raj Narharia**

</div>