import sqlite3
import pandas as pd
import numpy as np


class HybridRecommender:

    def __init__(self, db_path='spotify_recommender.db'):
        """Initialize the database connection."""
        self.conn = sqlite3.connect(db_path)

    def get_track_info(self, track_id):
        """Helper function to get the name and artist of a track ID."""
        query = "SELECT track_id, track_name, artists FROM kaggle_audio_features WHERE track_id = ?"
        return pd.read_sql_query(query, self.conn, params=(track_id,))

    def layer_1_audio_features(self, seed_track_id, limit=10):
        """Layer 1: Content-Based Filtering (Euclidean distance on audio features)."""
        seed_query = """
        SELECT danceability, energy, key, loudness, mode, speechiness, 
               acousticness, instrumentalness, liveness, valence, tempo
        FROM kaggle_audio_features 
        WHERE track_id = ?
        """
        seed_df = pd.read_sql_query(seed_query, self.conn, params=(seed_track_id,))
        if seed_df.empty:
            return pd.DataFrame(columns=['track_id', 'score'])

        seed_features = seed_df.iloc[0].values.astype(float)

        all_query = """
        SELECT track_id, danceability, energy, key, loudness, mode, speechiness, 
               acousticness, instrumentalness, liveness, valence, tempo
        FROM kaggle_audio_features
        WHERE track_id != ?
        """
        tracks_df = pd.read_sql_query(all_query, self.conn, params=(seed_track_id,))
        if tracks_df.empty:
            return pd.DataFrame(columns=['track_id', 'score'])

        feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

        feature_matrix = tracks_df[feature_cols].values.astype(float)

        distances = np.linalg.norm(feature_matrix - seed_features, axis=1)
        scores = 1 / (1 + distances)

        tracks_df['score'] = scores
        return tracks_df[['track_id', 'score']].sort_values(by='score', ascending=False).head(limit)

    def layer_2_artist_connections(self, seed_track_id, limit=10):
        """Layer 2: Graph/Metadata Connections (Exact artist matches)."""
        seed_info = self.get_track_info(seed_track_id)
        if seed_info.empty:
            return pd.DataFrame(columns=['track_id', 'score'])

        artist_name = seed_info.iloc[0]['artists']

        query = """
        SELECT track_id, 1.0 as score
        FROM kaggle_audio_features
        WHERE artists = ? AND track_id != ?
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(artist_name, seed_track_id, limit))

    def layer_3_shared_playlists(self, seed_track_id, limit=10):
        """Layer 3: Collaborative Filtering (Playlist co-occurrence)."""
        query = """
        SELECT 
            p2.track_uri as track_id, 
            CAST(COUNT(p2.playlist_id) AS REAL) as score
        FROM playlists p1
        JOIN playlists p2 ON p1.playlist_id = p2.playlist_id
        WHERE p1.track_uri = ? AND p2.track_uri != ?
        GROUP BY p2.track_uri
        ORDER BY score DESC
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(seed_track_id, seed_track_id, limit))

    def hybrid_recommend(self, seed_track_ids, top_n=10):
        """
        Combines layers using dynamic weighting. Handles edge cases where
        a track might only exist in one of the two datasets.
        """
        if isinstance(seed_track_ids, str):
            seed_track_ids = [seed_track_ids]

        scores_map = {}
        valid_seeds_processed = 0

        def accumulate(df, weight):
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    tid = row['track_id']
                    if tid in seed_track_ids:
                        continue
                    sc = row['score']
                    scores_map[tid] = scores_map.get(tid, 0.0) + (sc * weight)

        # Process each song independently
        for seed_id in seed_track_ids:
            l1 = self.layer_1_audio_features(seed_id, limit=50)
            l2 = self.layer_2_artist_connections(seed_id, limit=50)
            l3 = self.layer_3_shared_playlists(seed_id, limit=50)

            # EDGE CASE 1: Song is a complete ghost (missing from both datasets)
            if l1.empty and l3.empty:
                continue

            valid_seeds_processed += 1

            # DYNAMIC WEIGHTING ALGORITHM
            w1_dyn, w2_dyn, w3_dyn = 0.3, 0.3, 0.4

            if l1.empty:
                # Missing from Kaggle: Shift all weight to Alcrowd Collaborative Filtering
                w1_dyn, w2_dyn = 0.0, 0.0
                w3_dyn = 1.0
            elif l3.empty:
                # Missing from Alcrowd: Shift weight to Kaggle Audio/Artist data
                w3_dyn = 0.0
                w1_dyn, w2_dyn = 0.5, 0.5

            # Normalize scores within each layer to a 0-1 scale
            for df in [l1, l2, l3]:
                if not df.empty and df['score'].max() > 0:
                    df['score'] = df['score'] / df['score'].max()

            accumulate(l1, w1_dyn)
            accumulate(l2, w2_dyn)
            accumulate(l3, w3_dyn)

        # EDGE CASE 2: Base Case Fallback (None of the inputs existed)
        if valid_seeds_processed == 0:
            return self.get_popularity_baseline(limit=top_n)

        # Sort combined results, grabbing extra to account for duplicates
        sorted_tracks = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:top_n * 3]

        if not sorted_tracks:
            return pd.DataFrame(columns=['track_id', 'hybrid_score', 'track_name', 'artists', 'all_genres'])

        rec_df = pd.DataFrame(sorted_tracks, columns=['track_id', 'hybrid_score'])

        # Attach track metadata (names and artists) for readability
        track_ids = rec_df['track_id'].tolist()
        placeholders = ','.join(['?'] * len(track_ids))

        meta_query = f"""
                SELECT 
                    track_id, 
                    track_name, 
                    artists,
                    GROUP_CONCAT(track_genre, ', ') as all_genres
                FROM kaggle_audio_features 
                WHERE track_id IN ({placeholders})
                GROUP BY track_id, track_name, artists
                """

        meta_df = pd.read_sql_query(meta_query, self.conn, params=track_ids)

        final_df = pd.merge(rec_df, meta_df, on='track_id', how='left')

        # Drop songs with the exact same name and artist, keeping the one with the highest score
        final_df = final_df.drop_duplicates(subset=['track_name', 'artists'], keep='first')

        # Return exactly the number of tracks requested
        return final_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

    def close(self):
        self.conn.close()

    def get_popularity_baseline(self, limit=10):
        """
        Base Case: Returns the most globally popular tracks from the Alcrowd dataset.
        Triggered when user seed songs are completely missing from all databases.
        """
        query = """
        SELECT track_uri as track_id, CAST(COUNT(playlist_id) AS REAL) as popularity_score
        FROM playlists
        GROUP BY track_uri
        ORDER BY popularity_score DESC
        LIMIT ?
        """
        pop_df = pd.read_sql_query(query, self.conn, params=(limit,))

        if pop_df.empty:
            return pd.DataFrame(columns=['track_id', 'hybrid_score', 'track_name', 'artists', 'all_genres'])

        # Attach Kaggle metadata if the popular track happens to exist in the Golden Set
        track_ids = pop_df['track_id'].tolist()
        placeholders = ','.join(['?'] * len(track_ids))

        meta_query = f"""
                SELECT 
                    track_id, 
                    track_name, 
                    artists,
                    GROUP_CONCAT(track_genre, ', ') as all_genres
                FROM kaggle_audio_features 
                WHERE track_id IN ({placeholders})
                GROUP BY track_id, track_name, artists
                """

        meta_df = pd.read_sql_query(meta_query, self.conn, params=track_ids)

        final_df = pd.merge(pop_df, meta_df, on='track_id', how='left')

        # Rename column to match the expected output format
        final_df = final_df.rename(columns={'popularity_score': 'hybrid_score'})
        return final_df


# --- Test Script ---
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    rec = HybridRecommender()

    sample_query = "SELECT track_id, track_name, artists FROM kaggle_audio_features LIMIT 3"
    sample = pd.read_sql_query(sample_query, rec.conn)

    if len(sample) >= 2:
        test_ids = sample['track_id'].tolist()
        names = sample['track_name'].tolist()
        artists = sample['artists'].tolist()

        print(f"Testing Strategy B (Accumulator) with:\n1. {names[0]} by {artists[0]}\n2. {names[1]} by {artists[1]}\n3. {names[2]} by {artists[2]}\n")

        recommendations = rec.hybrid_recommend(test_ids, top_n=10)
        print("Top 10 Recommendations:")
        print(recommendations[['track_name', 'artists', 'hybrid_score']])
    else:
        print("Not enough tracks found in database.")

    rec.close()