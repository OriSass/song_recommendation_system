import sqlite3

conn = sqlite3.connect('../spotify_recommender.db')
cur = conn.cursor()

before = cur.execute("SELECT COUNT(*) FROM kaggle_audio_features").fetchone()[0]

# Keep one row per track_id (lowest rowid = first inserted), drop the rest
cur.execute("""
    DELETE FROM kaggle_audio_features
    WHERE rowid NOT IN (
        SELECT MIN(rowid)
        FROM kaggle_audio_features
        GROUP BY track_id
    )
""")
conn.commit()

after = cur.execute("SELECT COUNT(*) FROM kaggle_audio_features").fetchone()[0]
print(f"Removed {before - after} duplicate rows ({before} -> {after})")

conn.close()