import time

import pandas as pd
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HybridRecommender import HybridRecommender


def run_experiment(num_trials=3, playlists_per_trial=15, top_n=20, db_path='spotify_recommender.db'):
    """
    Runs a multi-trial Leave-K-Out evaluation to compare the Baseline model
    against two variants of the Hybrid Recommendation engine.
    """
    # --- START THE CLOCK ---
    experiment_start = time.time()

    rec = HybridRecommender(db_path)
    conn = rec.conn

    # Pre-compute baseline recommendations (global popularity is static)
    baseline_df = rec.get_popularity_baseline(limit=top_n)
    baseline_recs = set(baseline_df['track_id'].tolist())

    trial_records = []

    print(f"🚀 Starting {num_trials} Evaluation Trials ({playlists_per_trial} playlists/trial)...\n")

    for trial in range(1, num_trials + 1):
        # Fetch random test playlists for this trial
        playlist_query = """
        SELECT playlist_id, COUNT(track_uri) as track_count
        FROM playlists
        GROUP BY playlist_id
        HAVING track_count BETWEEN 10 AND 50
        ORDER BY RANDOM()
        LIMIT ?
        """
        test_playlists = pd.read_sql_query(playlist_query, conn, params=(playlists_per_trial,))

        base_hits, model_hits = 0, 0
        total_targets = 0

        for index, pid in enumerate(test_playlists['playlist_id']):
            tracks_query = "SELECT track_uri FROM playlists WHERE playlist_id = ?"
            tracks = pd.read_sql_query(tracks_query, conn, params=(pid,))['track_uri'].tolist()

            # Split the playlist into seeds and targets
            split_idx = len(tracks) // 2
            seeds = tracks[:split_idx]
            targets = set(tracks[split_idx:])
            total_targets += len(targets)

            # Model B: Collaborator Network
            rec_b = set(rec.hybrid_recommend(seeds, top_n=top_n)['track_id'].tolist())

            # Calculate Intersections (Hits)
            base_hits += len(baseline_recs.intersection(targets))
            model_hits += len(rec_b.intersection(targets))

            print(f"  [Trial {trial}] Processed {index + 1}/{playlists_per_trial} playlists...")

        # Calculate Recall Metric
        base_recall = (base_hits / total_targets) * 100 if total_targets > 0 else 0
        rec_recall = (model_hits / total_targets) * 100 if total_targets > 0 else 0

        trial_records.append({
            'Trial': trial,
            'Total_Targets': total_targets,
            'Baseline_Recall_%': round(base_recall, 2),
            'Model_Recall_%': round(rec_recall, 2)
        })

        print(
            f"✅ Trial {trial}/{num_trials} Completed | Baseline: {base_recall:.2f}% | Model: {rec_recall:.2f}%\n")

    results_df = pd.DataFrame(trial_records)

    # Save raw trial data to CSV
    csv_filename = "evaluation_results.csv"
    results_df.to_csv(csv_filename, index=False)

    # --- STOP THE CLOCK AND CALCULATE ---
    experiment_end = time.time()
    total_seconds = experiment_end - experiment_start
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)

    # Summary Statistics
    print("=" * 50)
    print(" 📊 AGGREGATED EVALUATION SUMMARY")
    print("=" * 50)
    print(
        f"Baseline Mean Recall:  {results_df['Baseline_Recall_%'].mean():.2f}% (±{results_df['Baseline_Recall_%'].std():.2f})")
    print(
        f"Model Recall:   {results_df['Model_Recall_%'].mean():.2f}% (±{results_df['Model_Recall_%'].std():.2f})")
    print("-" * 50)
    print(f"⏱️ Total Execution Time: {mins}m {secs}s")  # <-- Printed here
    print("=" * 50)
    print(f"📁 Full trial log saved to: {csv_filename}")

    rec.close()

if __name__ == "__main__":
    # Ensure full output is visible for pandas DataFrames
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Run 3 trials of 15 playlists
    run_experiment(num_trials=5, playlists_per_trial=30, top_n=20)