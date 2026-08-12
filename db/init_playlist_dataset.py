import sqlite3
import json
import os
import glob
from tqdm import tqdm

# 1. Connect to the SQLite database
conn = sqlite3.connect('../spotify_recommender.db')
cursor = conn.cursor()

# 2. Create Table 2 for the Playlist Data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS playlists (
        playlist_id TEXT,
        track_uri TEXT
    )
''')

# 3. Define the path to your data directory (update this to your actual path)
# Use a raw string (r'...') in Windows to handle backslashes correctly
data_dir = r'../million_playlist_dataset/data'

# Find all .json files in that directory
json_files = glob.glob(os.path.join(data_dir, '*.json'))

batch = []
batch_size = 100000  # Insert 100,000 rows at a time for speed

print(f"Found {len(json_files)} JSON files. Starting extraction...")

# 4. Loop through every single JSON file one by one
for file_path in tqdm(json_files, desc="Parsing Playlists", unit="file"):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Standard json.load is safe here because each file is small
        data = json.load(f)

        # Parse the playlists within the current file
        for playlist in data['playlists']:
            pid = playlist['pid']

            for track in playlist['tracks']:
                track_uri = track['track_uri'].split(':')[-1]
                batch.append((pid, track_uri))

                # When the batch is full, push to the database and clear the batch
                if len(batch) >= batch_size:
                    cursor.executemany('INSERT INTO playlists (playlist_id, track_uri) VALUES (?, ?)', batch)
                    conn.commit()
                    batch = []

# 5. Push any remaining rows in the final batch after all files are read
if batch:
    cursor.executemany('INSERT INTO playlists (playlist_id, track_uri) VALUES (?, ?)', batch)
    conn.commit()

print("Data extraction complete! Database is ready.")
conn.close()