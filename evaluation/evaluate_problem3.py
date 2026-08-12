import sqlite3
import pandas as pd
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HybridRecommender import HybridRecommender


def evaluate_backoff_strategy(total_playlists=100, top_n=20):
    print(f"Evaluating Problem 3: Backoff Strategy (Simulating Missing Data on {total_playlists} playlists)...\n")

    conn = sqlite3.connect('spotify_recommender.db')
    cur = conn.cursor()
    recommender = HybridRecommender(db_path='spotify_recommender.db')

    cur.execute("""
        SELECT playlist_id FROM playlists
        GROUP BY playlist_id HAVING COUNT(track_uri) BETWEEN 10 AND 50
        ORDER BY RANDOM() LIMIT ?
    """, (total_playlists,))
    test_playlists = [row[0] for row in cur.fetchall()]

    backoff_recalls = []
    full_signal_recalls = []
    baseline_recalls = []

    baseline_df = recommender.get_popularity_baseline(limit=top_n)
    baseline_recs = set(baseline_df['track_id'].tolist()) if not baseline_df.empty else set()

    for index, pid in enumerate(test_playlists):
        cur.execute("SELECT track_uri FROM playlists WHERE playlist_id = ?", (pid,))
        tracks = [row[0] for row in cur.fetchall()]

        split_idx = len(tracks) // 2
        seeds = tracks[:split_idx]
        targets = set(tracks[split_idx:])
        if not targets: continue

        # 1. Full Signal (Normal Run)
        recs_full = recommender.hybrid_recommend(seeds, top_n=top_n)
        rec_ids_full = set(recs_full['track_id'].tolist()) if not recs_full.empty else set()
        full_recall = (len(rec_ids_full.intersection(targets)) / len(targets)) * 100
        full_signal_recalls.append(full_recall)

        # 2. Simulated Backoff (Force Layer 3 to return empty)
        original_l3 = recommender.layer_3_shared_playlists
        recommender.layer_3_shared_playlists = lambda *args, **kwargs: pd.DataFrame(columns=['track_id', 'score'])

        recs_backoff = recommender.hybrid_recommend(seeds, top_n=top_n)
        rec_ids_backoff = set(recs_backoff['track_id'].tolist()) if not recs_backoff.empty else set()
        backoff_recall = (len(rec_ids_backoff.intersection(targets)) / len(targets)) * 100
        backoff_recalls.append(backoff_recall)

        # Restore Layer 3 for the next loop
        recommender.layer_3_shared_playlists = original_l3

        # 3. Baseline
        base_recall = (len(baseline_recs.intersection(targets)) / len(targets)) * 100
        baseline_recalls.append(base_recall)

        if (index + 1) % 10 == 0:
            print(f"  Processed {index + 1}/{total_playlists} playlists...")

    conn.close()
    recommender.close()

    print("\n=== PROBLEM 3 RESULTS ===")
    print(f"Global Baseline Mean Recall:   {sum(baseline_recalls) / len(baseline_recalls):.2f}%")
    print(f"Simulated Backoff Mean Recall: {sum(backoff_recalls) / len(backoff_recalls):.2f}%")
    print(f"Full Signal Mean Recall:       {sum(full_signal_recalls) / len(full_signal_recalls):.2f}%")


if __name__ == "__main__":
    evaluate_backoff_strategy()