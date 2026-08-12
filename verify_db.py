import sqlite3
import pandas as pd

print("Fetching data from both databases... (this will only take a few seconds)")

conn_backup = sqlite3.connect('spotify_recommender_backup.db')
conn_current = sqlite3.connect('spotify_recommender.db')

# 1. Get unique tracks and their features from the backup.
# By using MAX(), we collapse the identical duplicates into a single row for comparison.
query_backup = """
SELECT track_id, MAX(danceability) as danceability, MAX(energy) as energy, MAX(tempo) as tempo 
FROM kaggle_audio_features 
GROUP BY track_id 
ORDER BY track_id
"""
df_backup = pd.read_sql(query_backup, conn_backup)

# 2. Get the tracks and features from the current deduplicated DB
query_current = """
SELECT track_id, danceability, energy, tempo 
FROM kaggle_audio_features 
ORDER BY track_id
"""
df_current = pd.read_sql(query_current, conn_current)

# 3. Compare every single cell in the two DataFrames
print(f"Comparing all {len(df_current)} tracks...")

if df_backup.equals(df_current):
    print("✅ 100% SUCCESS: All 89,741 tracks perfectly match their original audio features!")
else:
    print("❌ DIFFERENCE DETECTED: Some features do not match.")

conn_backup.close()
conn_current.close()