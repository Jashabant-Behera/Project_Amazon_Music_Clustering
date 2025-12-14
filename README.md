# Amazon Music Clustering Project

## Overview
This project applies unsupervised machine learning techniques to cluster Amazon Music songs based on their audio characteristics. The goal is to identify patterns and group similar songs without prior labels, which can be useful for recommendation systems and playlist curation.

## Project Structure
```
Project_Amazon_Music_Clustering/
│
├── src/
│   ├── models/               # Saved models (Scaler, KMeans, PCA)
│   ├── notebooks/            # Jupyter Notebooks for analysis
│   │   ├── EDA.ipynb
│   │   ├── preprocessing.ipynb
│   │   ├── clustering.ipynb
│   │   ├── evaluation.ipynb
│   │   └── visualization.ipynb
│   ├── data/
│   │   ├── raw/              # Original dataset
│   │   └── processed/        # Processed datasets
│
├── app.py                    # Interactive Streamlit Dashboard
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Analysis (Notebooks):**
    You can run the Jupyter notebooks in `src/notebooks/` in the following order:
    1.  `EDA.ipynb`: Explore the dataset.
    2.  `preprocessing.ipynb`: Clean and scale the data.
    3.  `clustering.ipynb`: Train the K-Means model and assign clusters.
    4.  `visualization.ipynb`: Visualize the results using PCA.

3.  **Run the Dashboard:**
    Launch the interactive Streamlit app to explore clusters dynamically.
    ```bash
    streamlit run app.py
    ```

## Features Used
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration_ms

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn (KMeans, PCA, StandardScaler)
- Matplotlib, Seaborn, Plotly
- Streamlit
