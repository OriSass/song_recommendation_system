import sqlite3
import pandas as pd
import itertools
from collections import Counter
import re

print("Mining SQLite Database for ALL artist collaborations...")

conn = sqlite3.connect('../spotify_recommender.db')

# Fetch all tracks with multiple artists
query = """
SELECT artists 
FROM kaggle_audio_features 
WHERE artists LIKE '%;%' OR artists LIKE '%,%'
"""
df = pd.read_sql_query(query, conn)
conn.close()

collaboration_counter = Counter()

# Parse artists and count pairs
for artists_str in df['artists'].dropna():
    artists_list = [a.strip() for a in re.split(r'[;,]', str(artists_str)) if a.strip()]

    if len(artists_list) > 1:
        artists_list.sort()
        for pair in itertools.combinations(artists_list, 2):
            collaboration_counter[pair] += 1

edges = []
for (artist1, artist2), count in collaboration_counter.items():
    edges.append({'source': artist1, 'target': artist2, 'weight': count})

network_df = pd.DataFrame(edges)

# Sort by weight (highest collaborations first) but keep ALL of them
network_df = network_df.sort_values(by='weight', ascending=False)

# Save the full dataset
network_df.to_csv("global_artist_network.csv", index=False)
print(f"Success! Exported all {len(network_df)} unique collaboration edges to evaluation/global_artist_network.csv")