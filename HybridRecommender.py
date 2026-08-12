import sqlite3
import pandas as pd
import numpy as np
import os
import re

class HybridRecommender:
    """
    A music recommendation engine utilizing a hybrid accumulator strategy.
    Synthesizes Content-Based Filtering (audio features), Metadata Graphing (artists),
    and Collaborative Filtering (playlist co-occurrence) across multiple SQLite datasets.
    """

    def __init__(self, db_path='spotify_recommender.db'):
        """
        Initializes the database connection.
        Note: check_same_thread=False is added to safely support multi-threaded
        frameworks like Streamlit.

        Args:
            db_path (str): The relative or absolute path to the SQLite database file.
        """
        # Get the exact directory where HybridRecommender.py lives
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # If the provided db_path is just a file name (relative), make it absolute
        if not os.path.isabs(db_path):
            db_path = os.path.join(base_dir, db_path)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        all_tracks_query = f"SELECT track_id, {', '.join(feature_cols)} FROM kaggle_audio_features"
        self._audio_df = pd.read_sql_query(all_tracks_query, self.conn)
        self._audio_df = self._audio_df.set_index('track_id')
        self._feature_matrix = self._audio_df[feature_cols].values.astype(float)
        self._feature_track_ids = self._audio_df.index.to_numpy()

    def get_track_info(self, track_id):
        """
        Retrieves standard metadata (name and artist) for a specific track ID.

        Args:
            track_id (str): The unique Spotify/Kaggle track identifier.

        Returns:
            pd.DataFrame: A single-row DataFrame containing track_id, track_name, and artists.
        """
        query = "SELECT track_id, track_name, artists FROM kaggle_audio_features WHERE track_id = ?"
        return pd.read_sql_query(query, self.conn, params=(track_id,))

    def layer_1_audio_features(self, seed_track_id, limit=10):
        if seed_track_id not in self._audio_df.index:
            return pd.DataFrame(columns=['track_id', 'score'])

        seed_features = self._audio_df.loc[seed_track_id].values.astype(float)

        mask = self._feature_track_ids != seed_track_id
        candidate_ids = self._feature_track_ids[mask]
        candidate_matrix = self._feature_matrix[mask]

        distances = np.linalg.norm(candidate_matrix - seed_features, axis=1)
        scores = 1 / (1 + distances)

        result = pd.DataFrame({'track_id': candidate_ids, 'score': scores})
        return result.sort_values(by='score', ascending=False).head(limit)
    def layer_2_collaborations(self, seed_track_id, limit=10):
        """
        Layer 2 (Model B): Combined Direct Artist + Collaborator Network.
        Explicitly splits the candidate quota between direct seed artist tracks
        and frequent collaborator tracks so both are represented without crowding.
        """
        seed_info = self.get_track_info(seed_track_id)
        if seed_info.empty:
            return pd.DataFrame(columns=['track_id', 'score'])

        raw_artist_str = seed_info.iloc[0]['artists']
        if not raw_artist_str:
            return pd.DataFrame(columns=['track_id', 'score'])

        # seed_artists = [a.strip() for a in str(raw_artist_str).split(';') if a.strip()]
        seed_artists = [a.strip() for a in re.split(r'[;,]', str(raw_artist_str)) if a.strip()]
        if not seed_artists:
            return pd.DataFrame(columns=['track_id', 'score'])

        # 1. Get Direct Artist Tracks (Quota: Hard cap to prevent artist flooding)
        direct_limit = 2  # <-- Hard cap: Only allow 2 direct tracks per seed song!
        direct_clauses = " OR ".join(["artists = ?"] * len(seed_artists))
        direct_params = seed_artists + [seed_track_id, direct_limit]

        direct_query = f"""
                SELECT track_id, artists
                FROM kaggle_audio_features
                WHERE ({direct_clauses}) AND track_id != ?
                LIMIT ?
                """
        direct_df = pd.read_sql_query(direct_query, self.conn, params=direct_params)
        direct_df['score'] = 0.75  # <-- Lowered from 0.9 so strong audio/playlist matches can beat it

        # 2. Find Collaborator Artists from the dataset
        like_clauses = " OR ".join(["artists LIKE ?"] * len(seed_artists))
        collab_params = [f"%{artist}%" for artist in seed_artists]
        collab_query = f"SELECT artists FROM kaggle_audio_features WHERE ({like_clauses})"
        collab_df = pd.read_sql_query(collab_query, self.conn, params=collab_params)

        collaborator_counts = {}
        for artist_str in collab_df['artists'].dropna():
            for artist in str(artist_str).split(';'):
                artist = artist.strip()
                if artist and artist not in seed_artists:
                    collaborator_counts[artist] = collaborator_counts.get(artist, 0) + 1

        # 3. Get Collaborator Tracks (Quota: remaining limit slots)
        collab_dfs = []
        if collaborator_counts:
            sorted_collaborators = sorted(collaborator_counts.items(), key=lambda x: x[1], reverse=True)
            top_collaborators = [artist for artist, count in sorted_collaborators[:5]]

            collab_limit = limit - len(direct_df)
            if collab_limit > 0:
                collab_clauses = " OR ".join(["artists LIKE ?"] * len(top_collaborators))
                collab_target_params = [f"%{ca}%" for ca in top_collaborators] + [seed_track_id, collab_limit]

                collab_query_db = f"""
                SELECT track_id, artists
                FROM kaggle_audio_features
                WHERE ({collab_clauses}) AND track_id != ?
                LIMIT ?
                """
                collab_candidates = pd.read_sql_query(collab_query_db, self.conn, params=collab_target_params)

                # Assign dynamic scores based on collaboration frequency
                scores = []
                for _, row in collab_candidates.iterrows():
                    row_artists = [a.strip() for a in str(row['artists']).split(';')]
                    c_score = 0.85  # Default strong collaborator score
                    for ca in top_collaborators:
                        if ca in row_artists or ca in str(row['artists']):
                            freq = collaborator_counts.get(ca, 1)
                            c_score = min(1.0, 0.85 + (0.03 * freq))
                            break
                    scores.append(c_score)
                collab_candidates['score'] = scores
                collab_dfs.append(collab_candidates)

        # 4. Combine both pools cleanly
        combined_dfs = [direct_df] + collab_dfs
        final_df = pd.concat(combined_dfs, ignore_index=True).drop_duplicates(subset=['track_id'])

        return final_df[['track_id', 'score']].sort_values(by='score', ascending=False).head(limit)


    def layer_3_shared_playlists(self, seed_track_id, limit=10, max_playlists=100):
        """
        Layer 3: Collaborative Filtering via Shared Playlists.
        Bulletproof Optimization: Uses a two-step Python process to absolutely force
        SQLite to use indexes, bypassing its broken subquery optimizer.
        """
        # Step 1: Explicitly grab up to 100 playlist IDs
        p_query = "SELECT playlist_id FROM playlists WHERE track_uri = ? LIMIT ?"
        p_ids_df = pd.read_sql_query(p_query, self.conn, params=(seed_track_id, max_playlists))

        if p_ids_df.empty:
            return pd.DataFrame(columns=['track_id', 'score'])

        p_ids = p_ids_df['playlist_id'].tolist()

        # Step 2: Dynamically build an exact IN clause (e.g., IN (?, ?, ?))
        placeholders = ','.join('?' * len(p_ids))
        t_query = f"""
        SELECT track_uri as track_id, CAST(COUNT(playlist_id) AS REAL) as score
        FROM playlists
        WHERE playlist_id IN ({placeholders}) AND track_uri != ?
        GROUP BY track_uri
        ORDER BY score DESC
        LIMIT ?
        """

        # Combine the playlist IDs with the final parameters
        params = p_ids + [seed_track_id, limit]
        df = pd.read_sql_query(t_query, self.conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=['track_id', 'score'])

        return df[['track_id', 'score']]

    def hybrid_recommend(self, seed_track_ids, top_n=10, normalize=True):
        """
        Executes the Accumulator Strategy. Loops through an array of seed tracks,
        calculates candidates independently across all 3 layers, and blends them
        into a unified scoring pool using dynamic weights to handle missing dataset entries.

        Args:
            seed_track_ids (list): A list of input track IDs to base recommendations on.
            top_n (int): The final number of unique recommendations to return.

        Returns:
            pd.DataFrame: The final sorted recommendations with full track metadata.
            :param normalize:
        """
        if isinstance(seed_track_ids, str):
            seed_track_ids = [seed_track_ids]

        scores_map = {}
        valid_seeds_processed = 0

        def accumulate(df, weight):
            """Helper function to incrementally add weighted scores to the global pool."""
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    tid = row['track_id']
                    # Prevent recommending the seed songs back to the user
                    if tid in seed_track_ids:
                        continue
                    sc = row['score']
                    scores_map[tid] = scores_map.get(tid, 0.0) + (sc * weight)

        # Iterate through each user-provided seed song
        for seed_id in seed_track_ids:
            l1 = self.layer_1_audio_features(seed_id, limit=50)
            l2 = self.layer_2_collaborations(seed_id, limit=50)
            l3 = self.layer_3_shared_playlists(seed_id, limit=50)
            w1_dyn, w2_dyn, w3_dyn = 0.25, 0.4, 0.35

            # Edge Case 1: Track does not exist in any dataset
            if l1.empty and l3.empty:
                continue

            valid_seeds_processed += 1

            # # Dynamic Weighting Algorithm: Balances the formula if data is missing
            # w1_dyn, w2_dyn, w3_dyn = 0.3, 0.3, 0.4

            if l1.empty:
                # Track is missing from Kaggle; rely entirely on Alcrowd playlists
                w1_dyn, w2_dyn = 0.0, 0.0
                w3_dyn = 1.0
            elif l3.empty:
                # Track is missing from Alcrowd; rely entirely on Kaggle audio/metadata
                w3_dyn = 0.0
                w1_dyn, w2_dyn = 0.5, 0.5

            if normalize:
                # Min-Max Scaling: Normalize layer scores to a 0.0 - 1.0 scale before applying weights
                for df in [l1, l2, l3]:
                    if not df.empty and df['score'].max() > 0:
                        df['score'] = df['score'] / df['score'].max()

            accumulate(l1, w1_dyn)
            accumulate(l2, w2_dyn)
            accumulate(l3, w3_dyn)

        # Edge Case 2: Base Case Fallback (None of the user inputs were valid)
        if valid_seeds_processed == 0:
            return self.get_popularity_baseline(limit=top_n)

        # Sort the global pool. Request top_n * 3 to create a buffer for deduplication.
        sorted_tracks = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:top_n * 20]

        if not sorted_tracks:
            return pd.DataFrame(columns=['track_id', 'hybrid_score', 'track_name', 'artists', 'all_genres'])

        rec_df = pd.DataFrame(sorted_tracks, columns=['track_id', 'hybrid_score'])

        # Format track IDs for the SQL IN clause
        track_ids = rec_df['track_id'].tolist()
        placeholders = ','.join(['?'] * len(track_ids))

        # Query metadata and use GROUP_CONCAT to prevent Cartesian Product row explosion on multi-genre tracks
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

        # Deduplication: Remove different track_ids that share the exact same title and artist (e.g. Single vs Album edits)
        final_df = final_df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
        # Drop any tracks that were used as seed songs
        final_df = final_df[~final_df['track_id'].isin(seed_track_ids)]

        # --- NEW: PHANTOM DUPLICATE FILTER ---
        # Query the database to get the exact track names of the seed IDs
        seed_placeholders = ','.join(['?'] * len(seed_track_ids))
        seed_names_query = f"SELECT DISTINCT track_name FROM kaggle_audio_features WHERE track_id IN ({seed_placeholders})"
        seed_names_df = pd.read_sql_query(seed_names_query, self.conn, params=seed_track_ids)
        seed_names_list = seed_names_df['track_name'].tolist()

        # Drop any recommendations where the track_name matches a seed song name
        final_df = final_df[~final_df['track_name'].isin(seed_names_list)]

        # --- SMART ANTI-FLOOD ARTIST CAP ---
        # --- SMART ANTI-FLOOD ARTIST CAP (COMMAS & SEMICOLONS) ---
        final_df = final_df.sort_values(by='hybrid_score', ascending=False)

        # Extract true primary artist (the very first artist before any comma or semicolon)
        final_df['primary_artist'] = final_df['artists'].apply(lambda x: re.split(r'[;,]', str(x))[0].strip())

        # Get all individual seed artists (splitting by comma or semicolon)
        seed_info_df = pd.read_sql_query(
            f"SELECT DISTINCT artists FROM kaggle_audio_features WHERE track_id IN ({','.join(['?'] * len(seed_track_ids))})",
            self.conn, params=seed_track_ids
        )
        seed_artists_set = set()
        for art_str in seed_info_df['artists'].dropna():
            for a in re.split(r'[;,]', str(art_str)):
                if a.strip():
                    seed_artists_set.add(a.strip())

        # Separate recommendations into "Known Seed Artists" vs "New Discoveries"
        known_recs = final_df[final_df['primary_artist'].isin(seed_artists_set)]
        new_recs = final_df[~final_df['primary_artist'].isin(seed_artists_set)]

        # Hard Cap: Allow a MAXIMUM of 2 total tracks from the user's seed artists
        max_known_allowed = 2
        final_known = known_recs.head(max_known_allowed)

        remaining_slots = top_n - len(final_known)
        final_new = new_recs.head(remaining_slots)

        final_df = pd.concat([final_known, final_new], ignore_index=True)
        return final_df.head(top_n)

    def get_popularity_baseline(self, limit=10):
        """
        Base Case Strategy.
        Returns the most globally popular tracks aggregated from the playlist dataset.

        Args:
            limit (int): The number of popular tracks to return.

        Returns:
            pd.DataFrame: A DataFrame of tracks scored purely by global playlist frequency.
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

        # Standardize column naming convention for the Streamlit frontend
        final_df = final_df.rename(columns={'popularity_score': 'hybrid_score'})
        return final_df

    def close(self):
        """Safely closes the SQLite connection."""
        self.conn.close()

    def get_random_tracks(self, limit=5):
        """
        Helper method for the frontend UI.
        Fetches a random selection of tracks for the discovery panel.
        """
        query = "SELECT track_id, track_name, artists FROM kaggle_audio_features ORDER BY RANDOM() LIMIT ?"
        return pd.read_sql_query(query, self.conn, params=(limit,))

    def get_audio_features(self, track_ids):
        """
        Helper method for the frontend UI.
        Fetches the 6 core audio features for a list of track IDs to use in the vibe analysis chart.
        """
        if isinstance(track_ids, str):
            track_ids = [track_ids]

        if not track_ids:
            return pd.DataFrame()

        placeholders = ','.join(['?'] * len(track_ids))
        query = f"""
            SELECT danceability, energy, valence, acousticness, instrumentalness, liveness 
            FROM kaggle_audio_features WHERE track_id IN ({placeholders})
        """
        return pd.read_sql_query(query, self.conn, params=track_ids)


# --- Test Script ---
if __name__ == "__main__":
    # Configure Pandas display limits for full terminal visibility
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    rec = HybridRecommender()

    # Deterministic test query (pulls the first 3 rows consistently)
    sample_query = "SELECT track_id, track_name, artists FROM kaggle_audio_features LIMIT 3"
    sample = pd.read_sql_query(sample_query, rec.conn)

    if len(sample) >= 2:
        test_ids = sample['track_id'].tolist()
        names = sample['track_name'].tolist()

        print(f"Testing Strategy B (Accumulator) with:\n1. {names[0]}\n2. {names[1]}\n3. {names[2]}\n")

        recommendations = rec.hybrid_recommend(test_ids, top_n=10)
        print("Top 10 Recommendations:")
        print(recommendations[['track_name', 'artists', 'hybrid_score']])
    else:
        print("Not enough tracks found in database.")

    rec.close()