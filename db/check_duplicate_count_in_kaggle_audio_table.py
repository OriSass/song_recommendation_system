import sqlite3, pandas as pd
conn = sqlite3.connect('../spotify_recommender.db')
dupes = pd.read_sql_query("""
    SELECT track_id, COUNT(*) as cnt
    FROM kaggle_audio_features
    GROUP BY track_id
    HAVING cnt > 1
""", conn)
print(f"{len(dupes)} track_ids appear more than once")
print(dupes.sort_values('cnt', ascending=False).head(10))