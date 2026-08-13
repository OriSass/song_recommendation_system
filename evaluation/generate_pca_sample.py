import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HybridRecommender import HybridRecommender


def generate_background_sample():
    print("🔄 Fetching the ENTIRE global audio dataset for PCA background...")
    rec = HybridRecommender()

    # Grab all tracks with no limit
    global_df = rec.get_global_audio()

    global_df.to_csv("global_audio_sample.csv", index=False)
    print(f"✅ Success: Saved {len(global_df)} tracks to 'global_audio_sample.csv'")

    rec.close()


if __name__ == "__main__":
    generate_background_sample()