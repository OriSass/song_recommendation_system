import pandas as pd
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HybridRecommender import HybridRecommender


def run_offline_evaluation(trials=10):
    print(f"🚀 Starting Offline Evaluation (Targeting {trials} successful trials)...")
    rec = HybridRecommender()

    results_data = []
    successful_trials = 0
    attempts = 0
    max_attempts = 50  # Failsafe to prevent infinite loops

    while successful_trials < trials and attempts < max_attempts:
        attempts += 1
        try:
            # 1. Grab 2 random seeds
            random_seeds = rec.get_random_tracks(limit=2)
            if random_seeds.empty:
                continue

            seed_ids = random_seeds['track_id'].tolist()

            # Get true seed artists safely
            seed_artists_set = set()
            for art_str in random_seeds['artists'].dropna():
                for a in re.split(r'[;,]', str(art_str)):
                    if a.strip():
                        seed_artists_set.add(a.strip())

            # 2. Run the engine
            smart_results, candidate_pool = rec.hybrid_recommend(seed_ids, top_n=10, normalize=False)

            # Skip if the algorithm couldn't find enough data for a fair comparison
            if smart_results.empty or candidate_pool.empty or len(candidate_pool) < 10:
                continue

                # 3. The "Naive" output is the top 10 raw scores before the smart cap kicks in
            naive_results = candidate_pool.head(10)

            # 4. Calculate Retention for both
            def calc_retention(df):
                count = 0
                for rec_artists in df['artists'].dropna():
                    rec_arts = [a.strip() for a in re.split(r'[;,]', str(rec_artists)) if a.strip()]
                    if any(ra in seed_artists_set for ra in rec_arts):
                        count += 1
                return (count / len(df)) * 100 if len(df) > 0 else 0

            naive_retention = calc_retention(naive_results)
            smart_retention = calc_retention(smart_results)

            successful_trials += 1

            # Append to our data list
            results_data.append({"Trial": f"Trial {successful_trials}", "Retention Rate (%)": naive_retention,
                                 "Model": "Naive (No Cap)"})
            results_data.append({"Trial": f"Trial {successful_trials}", "Retention Rate (%)": smart_retention,
                                 "Model": "Smart Anti-Flood Cap"})

            print(f"Trial {successful_trials} Complete | Naive: {naive_retention:.1f}% | Smart: {smart_retention:.1f}%")

        except Exception as e:
            # Catch any unexpected database/pandas errors quietly and retry
            print(f"Skipping attempt {attempts} due to data anomaly...")
            continue

    # 5. Save to CSV
    if results_data:
        eval_df = pd.DataFrame(results_data)
        eval_df.to_csv("artist_retention_metrics.csv", index=False)
        print(f"\n✅ Success: Saved {successful_trials} real evaluation trials to 'artist_retention_metrics.csv'")
    else:
        print("\n❌ Failed to generate evaluation data. Check your database connections.")

    rec.close()

if __name__ == "__main__":
    run_offline_evaluation()