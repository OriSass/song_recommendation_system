# 🎵 Needle in a Data Haystack: Hybrid Spotify Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57)

A highly scalable, multi-layered music recommendation engine built to solve the "Popularity Bias" problem in Music Information Retrieval (MIR). By combining acoustic audio features, artist collaboration networks, and community playlist behaviors, this system surfaces high-quality niche tracks that traditional global-popularity algorithms bury.

Developed as the final project for **A Needle in a Data Haystack (67978)** at The Hebrew University of Jerusalem.

## 👥 Team Members
* **Ori Sass** - ori.sass@mail.huji.ac.il | 206789182
* **Ziv Ben Gigi** - ziv.bengigi@mail.huji.ac.il | 314750795
* **Lian Nissim** - Lian.nissim@mail.huji.ac.il | 318804572

## 🎥 Video Demonstration
**Watch our system in action:** [INSERT YOUR YOUTUBE/DRIVE DEMO LINK HERE]

---

## 🚀 The Architecture
The engine utilizes a 3-Layer Weighted Accumulator Strategy:
1. **Layer 1: Audio Features (25% Weight):** Vectorized Euclidean distance finding tracks mathematically similar to the seed based on 11 acoustic features (tempo, energy, acousticness, etc.).
2. **Layer 2: Artist Connections (40% Weight):** A collaboration network prioritizing direct-artist matches before blending frequent collaborators, governed by a strict anti-flood quota cap.
3. **Layer 3: Shared Playlists (35% Weight):** Collaborative filtering querying junction tables to surface frequently co-occurring songs from 1,000,000 user-curated playlists.

### ⚙️ Key System Optimizations
* **Min-Max Layer Normalization:** Mathematically rebalances voting power so highly popular global hits do not drown out niche, obscure tracks.
* **Dynamic Backoff Strategy:** When obscure tracks lack community playlist data, the system dynamically drops Layer 3 weight to 0% and splits voting power 50/50 between Audio and Artist layers to prevent zero-division errors.
* **RAM Caching Matrix:** Caching the full feature matrix directly into memory during system initialization yields a **14.2x execution speedup**.

---
## 📊 The Data
This project processes and synthesizes two distinct datasets, joined via a rigorous deduplication and URI-stripping pipeline to create a "Golden Dataset" intersection of over 15,000 tracks.
* **Kaggle Spotify Tracks DB:** 114,000 tracks with audio features and metadata (20MB).
* **AIcrowd Playlist Dataset:** 1,000,000 user-curated JSON playlists (35GB).

* **(Note: The fully processed, deduplicated SQLite database is 9.6GB. Due to GitHub and submission file size limits, it is hosted externally on Google Drive).**
---

## 💻 Installation & Running the Dashboard

We built a fully interactive Streamlit dashboard so users can input seed songs, generate recommendations, and visually audit the system's performance, PCA space, and layer weights in real-time.

**1. Clone the repository and navigate to the folder:**

`git clone https://github.com/YourUsername/Spotify-Hybrid-Recommender.git`

`cd Spotify-Hybrid-Recommender`

**2. Download and Extract the Database:** 

Download our compressed SQLite database from Google Drive: 👉 **[Click here to download the database](https://drive.google.com/file/d/1pKlbAaqCZv6zVNy9DehJL5dzf7HDO2tJ/view?usp=sharing)** 

*(Note: Shared directly with dafna.shahaf@mail.huji.ac.il as per submission guidelines).*
Once downloaded, extract the `.zip` file and place the resulting `.db` file directly into the root directory of this project.

**3. Install dependencies:**
`pip install -r requirements.txt`

**4. Run the Streamlit App:**
`streamlit run app.py`