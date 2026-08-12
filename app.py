import streamlit as st
import pandas as pd
import altair as alt
import re
from HybridRecommender import HybridRecommender

# MUST be the very first Streamlit command in your file
st.set_page_config(page_title="Spotify Hybrid Recommender", layout="wide")

# --- UI & UX CSS STYLING ---
st.markdown("""
    <style>
    h1, h2, h3 { text-align: center !important; }
    .block-container {
        padding-top: 1rem; 
        padding-bottom: 5rem;
        padding-left: 10%;
        padding-right: 10%;
        max-width: 100%;
        min-height: 101vh; 
    }
    div.stButton > button[kind="primary"] {
        background-color: #28a745;
        color: white;
        border-color: #28a745;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
    thead tr th {
        text-align: center !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

if 'seed_bank' not in st.session_state:
    st.session_state['seed_bank'] = []

st.title("🎵 Spotify Hybrid Recommender")


@st.cache_resource
def get_engine():
    return HybridRecommender()


rec = get_engine()

col_seed, col_discover = st.columns(2)

# --- LEFT COLUMN: THE CATCH AREA (Seed Bank) ---
with col_seed:
    st.subheader("📥 Your Seed Songs")

    btn_clear1, btn_clear2, btn_clear3 = st.columns([1, 1.5, 1])
    with btn_clear2:
        if st.button("🗑️ Clear All Seeds", use_container_width=True, disabled=len(st.session_state['seed_bank']) == 0):
            st.session_state['seed_bank'] = []
            st.rerun()

    if st.session_state['seed_bank']:
        with st.container(height=300):
            for i, song in enumerate(st.session_state['seed_bank']):
                c1, c2 = st.columns([5, 1])
                c1.write(f"**{song['track_name']}** by {song['artists'].replace(';', ', ')}")
                if c2.button("❌ Remove", key=f"remove_{i}"):
                    st.session_state['seed_bank'].pop(i)
                    st.rerun()
    else:
        with st.container(height=300):
            st.info("Your seed bank is empty! Generate and add some songs from the right.")

# --- RIGHT COLUMN: DISCOVER RANDOM SONGS ---
with col_discover:
    st.subheader("🎲 Discover Random Songs")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1.5, 1])
    with btn_col2:
        generate_clicked = st.button("Generate New Random Songs", use_container_width=True)

    if generate_clicked:
        st.session_state['random_songs'] = rec.get_random_tracks(limit=5)

    if 'random_songs' in st.session_state:
        with st.container(height=300):
            for index, row in st.session_state['random_songs'].iterrows():
                c1, c2 = st.columns([5, 1])
                c1.write(f"**{row['track_name']}** by {row['artists'].replace(';', ', ')}")
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

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_clicked = st.button("🚀 Run Recommendation Engine", type="primary", use_container_width=True)

    if run_clicked:
        # Trigger popup toast telling the user to scroll down
        st.toast("Scroll down to view your results, stats, and audio visuals! 👇", icon="🚀")

        seed_ids = [song['track_id'] for song in st.session_state['seed_bank']]
        with st.spinner("Calculating hybrid scores..."):
            results = rec.hybrid_recommend(seed_ids, top_n=10, normalize=False)

            display_df = results.rename(columns={
                'track_name': 'Song Title',
                'artists': 'Artist(s)',
                'all_genres': 'Genres',
                'hybrid_score': 'Match Score'
            }).sort_values(by='Match Score', ascending=False)
            display_df['Artist(s)'] = display_df['Artist(s)'].str.replace(';', ', ')

            st.subheader("Top 10 Recommendations")
            st.dataframe(
                display_df[['Song Title', 'Artist(s)', 'Genres', 'Match Score']],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # --- FETCH AUDIO FEATURES FOR STATS & CHARTS ---
            seed_feat_df = rec.get_audio_features(seed_ids)
            rec_feat_df = rec.get_audio_features(results['track_id'].tolist())

            # --- FUN QUICK STATISTICS ---
            st.markdown("### ⚡ Quick Vibe Check")

            stat1, stat2, stat3 = st.columns(3)

            # Stat 1: Artist Retention (Fixed for commas & semicolons)
            seed_artists = set()
            for song in st.session_state['seed_bank']:
                for a in re.split(r'[;,]', str(song['artists'])):
                    if a.strip():
                        seed_artists.add(a.strip())

            same_artist_count = 0
            for rec_artists in display_df['Artist(s)']:
                rec_arts = [a.strip() for a in re.split(r'[;,]', str(rec_artists)) if a.strip()]
                if any(ra in seed_artists for ra in rec_arts):
                    same_artist_count += 1

            artist_overlap_pct = int((same_artist_count / len(display_df)) * 100)

            stat1.metric(label="Artist Retention", value=f"{artist_overlap_pct}%",
                         help="Percentage of recommendations by the exact same artists as your seeds.")

            # Stat 2: Genre Exploration
            rec_genres = set(
                [g.strip() for genres in display_df['Genres'].dropna() for g in str(genres).split(',') if g.strip()])
            stat2.metric(label="Unique Genres Explored", value=len(rec_genres),
                         help="The number of different musical genres blended into your Top 10.")

            # Stat 3: Energy Shift
            if not seed_feat_df.empty and not rec_feat_df.empty:
                seed_energy = seed_feat_df['energy'].mean()
                rec_energy = rec_feat_df['energy'].mean()
                energy_diff = rec_energy - seed_energy

                vibe_label = "Matching Vibe 🎧"
                if energy_diff > 0.05:
                    vibe_label = "More Energetic ⚡"
                elif energy_diff < -0.05:
                    vibe_label = "More Chill 🛋️"

                stat3.metric(label="Energy Shift", value=vibe_label, delta=f"{energy_diff * 100:+.1f}%",
                             help="Did the engine dial the energy up or down compared to your seeds?")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- DYNAMIC VISUAL: AUDIO VIBE ANALYSIS ---
            st.markdown("### 🎛️ Audio Vibe Analysis")
            st.caption("Comparing the average audio features of your Seed Songs vs. the Engine's Recommendations.")

            if not seed_feat_df.empty and not rec_feat_df.empty:
                compare_df = pd.DataFrame({
                    'Your Seeds': seed_feat_df.mean(),
                    'Recommendations': rec_feat_df.mean()
                })

                # Filter out empty or near-zero features so they don't clutter the chart
                compare_df = compare_df[(compare_df['Your Seeds'] > 0.005) | (compare_df['Recommendations'] > 0.005)]
                compare_df = compare_df.dropna()

                plot_df = compare_df.reset_index().melt(id_vars='index', var_name='Source', value_name='Score')
                plot_df.rename(columns={'index': 'Feature'}, inplace=True)

                # Build the chart with larger font sizes and filtered features
                chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X('Feature:N', axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=14)),
                    y=alt.Y('Score:Q', axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
                    color=alt.Color('Source:N', scale=alt.Scale(range=["#0068c9", "#83c9ff"])),
                    xOffset='Source:N'
                ).properties(height=350)

                st.altair_chart(chart, use_container_width=True)

# --- GENERIC EVALUATION METRICS (FOR WRITEUP SCREENSHOTS) ---
st.markdown("<br><br><br><br>", unsafe_allow_html=True)
with st.expander("📊 View System Evaluation Metrics (Writeup Data)"):
    st.write("Explore the interactive metrics generated during our algorithmic testing.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Problem 1: Model vs. Baseline Recall (5 Trials)**")
        st.caption("Tracking Recall@20 consistency across 5 independent test runs.")

        prob1_df = pd.DataFrame({
            "Trial": ["Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5"] * 2,
            "Recall": [1.86, 0.23, 0.88, 1.78, 1.51, 8.60, 8.43, 5.25, 7.87, 8.84],
            "Model": ["Global Baseline"] * 5 + ["Hybrid Model"] * 5
        })
        chart1 = alt.Chart(prob1_df).mark_line(point=True).encode(
            x=alt.X('Trial:N', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Recall:Q'),
            color=alt.Color('Model:N',
                            scale=alt.Scale(domain=['Global Baseline', 'Hybrid Model'], range=['#ff4b4b', '#28a745']))
        ).properties(height=250)
        st.altair_chart(chart1, use_container_width=True)

        st.markdown("**Problem 2: Mitigating Popularity Bias**")
        st.caption("Percentage of Top-20 attributable to the Niche Seed.")

        prob2_df = pd.DataFrame({
            "Configuration": ["Normalization OFF (Naive)", "Normalization ON (Fixed)"],
            "Attribution": [37.67, 41.00]
        })
        chart2 = alt.Chart(prob2_df).mark_bar(color='#0068c9').encode(
            x=alt.X('Configuration:N', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Attribution:Q')
        ).properties(height=250)
        st.altair_chart(chart2, use_container_width=True)

    with col_b:
        st.markdown("**Problem 3: Dynamic Backoff Strategy**")
        st.caption("Performance when Layer 3 playlist data is missing.")

        prob3_df = pd.DataFrame({
            "Strategy": ["Global Baseline", "Simulated Backoff", "Full Signal"],
            "Mean Recall": [0.82, 1.37, 10.78]
        })
        chart3 = alt.Chart(prob3_df).mark_bar(color='#29b5e8').encode(
            x=alt.X('Strategy:N', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Mean Recall:Q')
        ).properties(height=250)
        st.altair_chart(chart3, use_container_width=True)