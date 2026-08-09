import streamlit as st
import pandas as pd
from HybridRecommender import HybridRecommender

# MUST be the very first Streamlit command in your file
st.set_page_config(page_title="Spotify Hybrid Recommender", layout="wide")

# --- UI & UX CSS STYLING ---
st.markdown("""
    <style>
    /* Center all headers */
    h1, h2, h3 {
        text-align: center !important;
    }

    /* Apply percentage padding to the main block container and reduce top space */
    .block-container {
        padding-top: 1rem; 
        padding-bottom: 5rem;
        padding-left: 10%;
        padding-right: 10%;
        max-width: 100%;
        min-height: 101vh; /* This makes the page 1% taller than the screen */
    }

    /* Target only the primary button to make it green */
    div.stButton > button[kind="primary"] {
        background-color: #28a745;
        color: white;
        border-color: #28a745;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }

    /* Center, enlarge, and bold table headers */
    thead tr th {
        text-align: center !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
    
    </style>
""", unsafe_allow_html=True)

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

# Create two main columns for a side-by-side layout
col_seed, col_discover = st.columns(2)

# --- LEFT COLUMN: THE CATCH AREA (Seed Bank) ---
with col_seed:
    st.subheader("📥 Your Seed Songs")

    # Add a matching button layout to perfectly align with the right side
    btn_clear1, btn_clear2, btn_clear3 = st.columns([1, 1.5, 1])
    with btn_clear2:
        # The button disables itself automatically if the bank is already empty!
        if st.button("🗑️ Clear All Seeds", use_container_width=True, disabled=len(st.session_state['seed_bank']) == 0):
            st.session_state['seed_bank'] = []
            st.rerun()

    if st.session_state['seed_bank']:
        # Fixed height container for scrolling
        with st.container(height=300):
            for i, song in enumerate(st.session_state['seed_bank']):
                c1, c2 = st.columns([5, 1])
                c1.write(f"**{song['track_name']}** by {song['artists'].replace(';', ', ')}")

                # Allow users to remove songs
                if c2.button("❌ Remove", key=f"remove_{i}"):
                    st.session_state['seed_bank'].pop(i)
                    st.rerun()
    else:
        # Wrap the empty state in a 300px container so the UI doesn't collapse!
        with st.container(height=300):
            st.info("Your seed bank is empty! Generate and add some songs from the right.")

# --- RIGHT COLUMN: DISCOVER RANDOM SONGS ---
with col_discover:
    st.subheader("🎲 Discover Random Songs")

    # Center the button within this half of the screen
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1.5, 1])
    with btn_col2:
        generate_clicked = st.button("Generate New Random Songs", use_container_width=True)

    if generate_clicked:
        # Fetch 5 random songs from the database
        query = "SELECT track_id, track_name, artists FROM kaggle_audio_features ORDER BY RANDOM() LIMIT 5"
        st.session_state['random_songs'] = pd.read_sql_query(query, rec.conn)

    if 'random_songs' in st.session_state:
        # Putting this in a fixed-height container keeps both sides visually balanced
        with st.container(height=300):
            for index, row in st.session_state['random_songs'].iterrows():
                c1, c2 = st.columns([5, 1])
                c1.write(f"**{row['track_name']}** by {row['artists'].replace(';', ', ')}")

                # The "Catch" button
                if c2.button("➕ Add", key=f"add_{row['track_id']}"):
                    if not any(s['track_id'] == row['track_id'] for s in st.session_state['seed_bank']):
                        st.session_state['seed_bank'].append({
                            'track_id': row['track_id'],
                            'track_name': row['track_name'],
                            'artists': row['artists']
                        })
                        st.rerun()
                    else:
                        st.toast(f"**{row['track_name']}** is already in your seed bank!", icon="⚠️")

st.markdown("---")

# --- RUN THE ENGINE ---
if len(st.session_state['seed_bank']) > 0:

    # Create 3 equal columns to center the button
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        # Place the button in the center column and stretch it
        run_clicked = st.button("🚀 Run Recommendation Engine", type="primary", use_container_width=True)

    if run_clicked:
        seed_ids = [song['track_id'] for song in st.session_state['seed_bank']]
        with st.spinner("Calculating hybrid scores..."):
            results = rec.hybrid_recommend(seed_ids, top_n=10)

            # Rename columns to hide database naming conventions
            display_df = results.rename(columns={
                'track_name': 'Song Title',
                'artists': 'Artist(s)',
                'all_genres': 'Genres',
                'hybrid_score': 'Match Score'
            })
            # Replace semicolons with commas in the Artist(s) column
            display_df['Artist(s)'] = display_df['Artist(s)'].str.replace(';', ', ')

            # Display the final results cleanly
            st.subheader("Top 10 Recommendations")

            st.dataframe(
                display_df[['Song Title', 'Artist(s)', 'Genres', 'Match Score']],
                use_container_width=True,
                hide_index=True
            )
            # Add some empty space below the table
            st.markdown("<br><br><br>", unsafe_allow_html=True)