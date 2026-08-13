import sqlite3
import pandas as pd
import numpy as np
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HybridRecommender import HybridRecommender


def get_audio_features(cur, track_id):
    cur.execute("""
        SELECT danceability, energy, key, loudness, mode, speechiness, 
               acousticness, instrumentalness, liveness, valence, tempo 
        FROM kaggle_audio_features WHERE track_id = ?
    """, (track_id,))
    res = cur.fetchone()
    return np.array(res) if res else None


def evaluate_seed_drowning():
    print("Evaluating Problem 2: Seed Drowning (Normalization ON vs OFF)...\n")
    print("Fetching Popular and Niche tracks from the Golden Dataset intersection...")

    conn = sqlite3.connect('spotify_recommender.db')
    cur = conn.cursor()
    recommender = HybridRecommender(db_path='spotify_recommender.db')

    # Find Top 15 Most Popular Tracks strictly inside the Kaggle dataset
    cur.execute("""
        SELECT p.track_uri 
        FROM playlists p
        INNER JOIN kaggle_audio_features k ON p.track_uri = k.track_id
        GROUP BY p.track_uri 
        ORDER BY COUNT(p.playlist_id) DESC 
        LIMIT 15
    """)
    pop_tracks = [row[0] for row in cur.fetchall()]

    # Find 15 Niche Tracks strictly inside the Kaggle dataset
    cur.execute("""
        SELECT p.track_uri 
        FROM playlists p
        INNER JOIN kaggle_audio_features k ON p.track_uri = k.track_id
        GROUP BY p.track_uri 
        HAVING COUNT(p.playlist_id) BETWEEN 1 AND 2 
        ORDER BY RANDOM() 
        LIMIT 15
    """)
    niche_tracks = [row[0] for row in cur.fetchall()]

    seed_pairs = list(zip(pop_tracks, niche_tracks))
    results = []

    print(f"Testing {len(seed_pairs)} Extreme Pairs...\n")

    for pop_seed, niche_seed in seed_pairs:
        pop_vec = get_audio_features(cur, pop_seed)
        niche_vec = get_audio_features(cur, niche_seed)

        if pop_vec is None or niche_vec is None:
            continue

        for is_normalized in [True, False]:
            recs, _ = recommender.hybrid_recommend([pop_seed, niche_seed], top_n=20, normalize=is_normalized)
            niche_attributions = 0

            for _, rec in recs.iterrows():
                rec_vec = get_audio_features(cur, rec['track_id'])
                if rec_vec is not None:
                    dist_to_pop = np.linalg.norm(rec_vec - pop_vec)
                    dist_to_niche = np.linalg.norm(rec_vec - niche_vec)

                    if dist_to_niche < dist_to_pop:
                        niche_attributions += 1

            niche_pct = (niche_attributions / len(recs)) * 100 if not recs.empty else 0

            results.append({
                'normalized': "ON" if is_normalized else "OFF",
                'niche_percentage': niche_pct
            })

    conn.close()
    recommender.close()

    df = pd.DataFrame(results)

    if df.empty:
        print("❌ Error: No valid pairs were tested. Check database intersection.")
        return

    summary = df.groupby('normalized')['niche_percentage'].agg(['mean', 'std']).reset_index()

    print("=== PROBLEM 2 RESULTS ===")
    print("Mean % of Top-20 attributable to the Niche Seed:")
    print(f"Normalization OFF (Naive): {summary[summary['normalized'] == 'OFF']['mean'].values[0]:.2f}%")
    print(f"Normalization ON (Fixed):  {summary[summary['normalized'] == 'ON']['mean'].values[0]:.2f}%")


if __name__ == "__main__":
    evaluate_seed_drowning()