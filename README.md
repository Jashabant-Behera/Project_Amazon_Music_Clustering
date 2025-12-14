# Amazon Music Clustering Project

A machine learning project that analyzes and clusters music tracks based on audio features using unsupervised learning techniques.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Visualizations](#visualizations)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a complete machine learning pipeline to cluster music tracks from Amazon Music based on their audio characteristics. By analyzing features such as danceability, energy, tempo, and acousticness, the system groups similar tracks together to discover patterns and musical similarities.

**Key Objectives:**
- Analyze audio features of music tracks
- Apply dimensionality reduction techniques
- Cluster similar songs using unsupervised learning
- Visualize and interpret clustering results

---

## Project Structure

```
amazon-music-clustering/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed and clustered data
│       ├── clustered_data.csv
│       └── scaled_features.csv
│
├── models/                     # Saved models
│   ├── pca_model.pkl
│   └── clustering_model.pkl
│
├── src/                        # Source code
│   ├── visualization.ipynb     # Visualization notebook
│   ├── clustering.py           # Clustering implementation
│   └── preprocessing.py        # Data preprocessing
│
├── notebooks/                  # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Features

The project analyzes the following audio features:

| Feature | Description |
|---------|-------------|
| **Danceability** | How suitable a track is for dancing |
| **Energy** | Intensity and activity measure |
| **Loudness** | Overall volume in decibels |
| **Speechiness** | Presence of spoken words |
| **Acousticness** | Confidence measure of acoustic nature |
| **Instrumentalness** | Predicts whether a track contains vocals |
| **Liveness** | Detects presence of an audience |
| **Valence** | Musical positiveness/happiness |
| **Tempo** | Beats per minute (BPM) |
| **Duration** | Track length in milliseconds |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Jashabant-Behera/Project_Amazon_Music_Clustering.git
cd amazon-music-clustering
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Pipeline

1. **Data Preprocessing:**
```bash
python src/preprocessing.py
```

2. **Clustering Analysis:**
```bash
python src/clustering.py
```

3. **Visualization:**
```bash
jupyter notebook src/visualization.ipynb
```

### Quick Start Example

```python
import pandas as pd
from sklearn.decomposition import PCA
import pickle

# Load processed data
df = pd.read_csv('data/processed/clustered_data.csv')

# Load saved models
with open('models/pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# View cluster distribution
print(df['cluster'].value_counts())
```

---

## Methodology

### 1. Data Preprocessing
- Load raw music feature data
- Handle missing values and outliers
- Normalize features using StandardScaler
- Save processed data for reproducibility

### 2. Dimensionality Reduction
- Apply PCA (Principal Component Analysis)
- Reduce features to 2 components for visualization
- Preserve variance while simplifying data structure

### 3. Clustering
- Implement clustering algorithm (K-Means/DBSCAN/Hierarchical)
- Determine optimal number of clusters
- Assign cluster labels to each track
- Save trained models

### 4. Visualization & Analysis
- Create 2D scatter plots using PCA components
- Generate feature distribution plots by cluster
- Analyze cluster characteristics
- Interpret musical patterns

---

## Visualizations

The project generates several visualization types:

### Cluster Visualization
- **PCA Scatter Plot**: 2D representation of clusters in principal component space
- Color-coded clusters for easy interpretation
- Shows separation and overlap between music groups

### Feature Analysis
- **Box Plots**: Distribution of each audio feature across clusters
- **Violin Plots**: Density distribution within clusters
- **Heatmaps**: Feature correlation matrices

### Sample Output
```
Clusters Visualized using PCA
├── Cluster 0: High-energy, danceable tracks
├── Cluster 1: Acoustic, calm music
├── Cluster 2: Instrumental, ambient sounds
└── Cluster 3: Speech-heavy, podcast-style content
```

---

## Results

### Cluster Characteristics

**Cluster Insights:**
- Each cluster represents distinct musical characteristics
- Clear separation based on energy, danceability, and acousticness
- Potential applications in music recommendation systems

**Performance Metrics:**
- Silhouette Score: Measure of cluster quality
- Inertia: Within-cluster sum of squares
- Variance Explained: PCA component contribution

---

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

For a complete list, see `requirements.txt`

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Amazon Music for providing audio feature data
- scikit-learn community for excellent ML tools
- Contributors and maintainers

---

## Contact

For questions or feedback, please open an issue on GitHub or contact the project maintainer.

**Project Link:** [https://github.com/Jashabant-Behera/Project_Amazon_Music_Clustering.git](https://github.com/Jashabant-Behera/Project_Amazon_Music_Clustering.git)

---

**Last Updated:** December 2025