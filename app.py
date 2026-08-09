import streamlit as st
import pandas as pd
from HybridRecommender import HybridRecommender

# MUST be the very first Streamlit command in your file
st.set_page_config(page_title="Spotify Hybrid Recommender", layout="wide")

# 1. Initialize the "Catch Area" in Streamlit's session state
if 'seed_bank' not in st.session_state:
    st.session_state['seed_bank'] = []

st.title("🎵 Spotify Hybrid Recommender")


# Initialize your backend engine
# st.cache_resource ensures the database connection isn't rebuilt on every interaction
@st.cache_resource
def get_engine():
    return HybridRecommender()


rec = get_engine()

# --- THE CATCH AREA (Seed Bank) ---
st.subheader("📥 Your Seed Songs")
if st.session_state['seed_bank']:
    for i, song in enumerate(st.session_state['seed_bank']):
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{song['track_name']}** by {song['artists']}")
        # Allow users to remove songs from their catch area
        if col2.button("❌ Remove", key=f"remove_{i}"):
            st.session_state['seed_bank'].pop(i)
            st.rerun()
else:
    st.info("Your seed bank is empty! Generate and add some songs below.")

st.markdown("---")

# --- RANDOM SONG GENERATOR ---
st.subheader("🎲 Discover Random Songs")
if st.button("Generate New Random Songs"):
    # Fetch 5 random songs from the database
    query = "SELECT track_id, track_name, artists FROM kaggle_audio_features ORDER BY RANDOM() LIMIT 5"
    st.session_state['random_songs'] = pd.read_sql_query(query, rec.conn)

if 'random_songs' in st.session_state:
    for index, row in st.session_state['random_songs'].iterrows():
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{row['track_name']}** by {row['artists']}")

        # The "Catch" button
        if col2.button("➕ Add to Seed", key=f"add_{row['track_id']}"):
            # Check if it's already in the bank to prevent duplicates
            if not any(s['track_id'] == row['track_id'] for s in st.session_state['seed_bank']):
                st.session_state['seed_bank'].append({
                    'track_id': row['track_id'],
                    'track_name': row['track_name'],
                    'artists': row['artists']
                })
                st.rerun()

st.markdown("---")

# --- RUN THE ENGINE ---
if len(st.session_state['seed_bank']) > 0:
    if st.button("🚀 Run Recommendation Engine", type="primary"):
        seed_ids = [song['track_id'] for song in st.session_state['seed_bank']]
        with st.spinner("Calculating hybrid scores..."):
            results = rec.hybrid_recommend(seed_ids, top_n=10)

            # Display the final results cleanly
            st.subheader("Top 10 Recommendations")
            st.dataframe(
                results[['track_name', 'artists', 'all_genres', 'hybrid_score']],
                use_container_width=True,
                hide_index=True
            )