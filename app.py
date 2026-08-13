import streamlit as st
import pandas as pd
import altair as alt
import re
import plotly.express as px
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

# --- Session state for persisting execution results ---
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = None
if 'full_candidate_pool' not in st.session_state:
    st.session_state['full_candidate_pool'] = None
if 'seed_feat_df' not in st.session_state:
    st.session_state['seed_feat_df'] = None
if 'rec_feat_df' not in st.session_state:
    st.session_state['rec_feat_df'] = None

st.title("🎵 Spotify Hybrid Recommender")


@st.cache_resource
def get_engine():
    return HybridRecommender()


rec = get_engine()


# --- Load static global PCA background sample ---
@st.cache_data
def load_global_sample():
    try:
        return pd.read_csv("evaluation/global_audio_sample.csv")
    except FileNotFoundError:
        return pd.DataFrame()


col_seed, col_discover = st.columns(2)

# --- LEFT COLUMN: THE CATCH AREA (Seed Bank) ---
with col_seed:
    st.subheader("📥 Your Seed Songs")

    btn_clear1, btn_clear2, btn_clear3 = st.columns([1, 1.5, 1])
    with btn_clear2:
        if st.button("🗑️ Clear All Seeds", use_container_width=True, disabled=len(st.session_state['seed_bank']) == 0):
            st.session_state['seed_bank'] = []
            st.session_state['recommendations'] = None  # Clear results on reset
            st.rerun()

    if st.session_state['seed_bank']:
        with st.container(height=300):
            for i, song in enumerate(st.session_state['seed_bank']):
                c1, c2 = st.columns([5, 1])
                c1.write(f"**{song['track_name']}** by {song['artists'].replace(';', ', ')}")
                if c2.button("❌ Remove", key=f"remove_{i}"):
                    st.session_state['seed_bank'].pop(i)
                    st.session_state['recommendations'] = None
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

# --- RUN THE ENGINE BUTTON ---
if len(st.session_state['seed_bank']) > 0:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_clicked = st.button("🚀 Run Recommendation Engine", type="primary", use_container_width=True)

    if run_clicked:
        st.toast("Scroll down to view your results, stats, and audio visuals! 👇", icon="🚀")
        seed_ids = [song['track_id'] for song in st.session_state['seed_bank']]

        with st.spinner("Calculating hybrid scores..."):
            results, full_candidate_pool = rec.hybrid_recommend(seed_ids, top_n=10, normalize=False)
            seed_feat_df = rec.get_audio_features(seed_ids)
            rec_feat_df = rec.get_audio_features(results['track_id'].tolist())

            # Store in session state so results persist cleanly
            st.session_state['recommendations'] = results
            st.session_state['full_candidate_pool'] = full_candidate_pool
            st.session_state['seed_feat_df'] = seed_feat_df
            st.session_state['rec_feat_df'] = rec_feat_df

# --- DISPLAY RESULTS ONLY AFTER RUNNING ---
if st.session_state['recommendations'] is not None:
    results = st.session_state['recommendations']
    full_candidate_pool = st.session_state['full_candidate_pool']
    seed_feat_df = st.session_state['seed_feat_df']
    rec_feat_df = st.session_state['rec_feat_df']
    seed_ids = [song['track_id'] for song in st.session_state['seed_bank']]

    display_df = results.rename(columns={
        'track_name': 'Song Title',
        'artists': 'Artist(s)',
        'all_genres': 'Genres',
        'hybrid_score': 'Match Score'
    }).sort_values(by='Match Score', ascending=False)
    display_df['Artist(s)'] = display_df['Artist(s)'].str.replace(';', ', ')
    display_df['Match Score'] = display_df['Match Score'].round(3)

    st.subheader("Top 10 Recommendations")
    st.dataframe(
        display_df[['Song Title', 'Artist(s)', 'Genres', 'Match Score']],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FUN QUICK STATISTICS ---
    st.markdown("### ⚡ Quick Vibe Check")

    stat1, stat2, stat3 = st.columns(3)

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

    rec_genres = set(
        [g.strip() for genres in display_df['Genres'].dropna() for g in str(genres).split(',') if g.strip()])
    stat2.metric(label="Unique Genres Explored", value=len(rec_genres),
                 help="The number of different musical genres blended into your Top 10.")

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

    st.markdown("---")
    st.caption(
        "Explore how the hybrid engine solves core recommendation challenges based on your current seeds and global dataset.")

    tab1, tab2, tab3 = st.tabs([
        "🎯 Sub-Problem 1: Model Recall & Performance",
        "🌐 Sub-Problem 2: Mitigating Popularity Bias",
        "⚙️ Sub-Problem 3: Dynamic Backoff Strategy"
    ])

    # ==========================================
    # TAB 1: MODEL PERFORMANCE & RECALL
    # ==========================================
    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Separating Signal from Noise")
            st.caption("Score decay across all candidates, highlighting the 'long tail' distribution.")

            decay_df = full_candidate_pool.reset_index(drop=True)
            decay_df['Rank'] = decay_df.index + 1

            fig1 = px.line(
                decay_df,
                x="Rank",
                y="hybrid_score",
                color_discrete_sequence=["#0068c9"],
                labels={"hybrid_score": "Match Score", "Rank": "Candidate Rank"}
            )
            fig1.add_vline(
                x=10, line_dash="dash", line_color="red", line_width=2,
                annotation_text="Top 10 Cutoff",
                annotation_position="top right"
            )
            fig1.update_traces(fill='tozeroy', fillcolor='rgba(0, 104, 201, 0.2)')
            fig1.update_layout(height=420, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("### Why Popularity Fails")
            st.caption("Offline Evaluation: Mean Recall (Baseline vs. Hybrid Model)")

            try:
                eval_df = pd.read_csv("evaluation/evaluation_results.csv")
                eval_melted = eval_df.melt(
                    id_vars='Trial',
                    value_vars=['Baseline_Recall_%', 'Model_Recall_%'],
                    var_name='Model',
                    value_name='Recall (%)'
                )
                eval_melted['Model'] = eval_melted['Model'].map({
                    'Baseline_Recall_%': 'Popularity Baseline',
                    'Model_Recall_%': 'Hybrid Engine'
                })

                fig2 = px.bar(
                    eval_melted,
                    x="Trial",
                    y="Recall (%)",
                    color="Model",
                    barmode="group",
                    color_discrete_map={"Popularity Baseline": "#ff4b4b", "Hybrid Engine": "#28a745"}
                )
                fig2.update_layout(
                    height=420,
                    margin=dict(t=30, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, title_text="")
                )
                st.plotly_chart(fig2, use_container_width=True)
            except FileNotFoundError:
                st.info("⚠️ Ensure 'evaluation_results.csv' is in your project directory.")

        with col3:
            st.markdown("### 🎛️ Audio Vibe Analysis")
            st.caption("Average audio features of your Seeds vs. Recommendations.")

            if not seed_feat_df.empty and not rec_feat_df.empty:
                compare_df = pd.DataFrame({
                    'Your Seeds': seed_feat_df.mean(),
                    'Recommendations': rec_feat_df.mean()
                })
                compare_df = compare_df[(compare_df['Your Seeds'] > 0.005) | (compare_df['Recommendations'] > 0.005)]
                compare_df = compare_df.dropna()

                plot_df = compare_df.reset_index().melt(id_vars='index', var_name='Source', value_name='Score')
                plot_df.rename(columns={'index': 'Feature'}, inplace=True)

                chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X('Feature:N',
                            axis=alt.Axis(labelAngle=-45, labelFontSize=11, titleFontSize=12, labelOverlap=False)),
                    y=alt.Y('Score:Q', axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
                    color=alt.Color('Source:N', scale=alt.Scale(range=["#0068c9", "#83c9ff"]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    xOffset='Source:N'
                ).properties(height=420)

                st.altair_chart(chart, use_container_width=True)

    # ==========================================
    # TAB 2: MITIGATING POPULARITY BIAS (Side-by-Side at height=420)
    # ==========================================
    with tab2:
        st.markdown("### 🌌 The Global Audio Universe & Feature Space")
        st.caption(
            "Validating the acoustic feature space and visualizing how the engine explores niche outer edges rather than the popular center.")

        col_pca, col_corr = st.columns([1.3, 1])

        # --- 1. Global PCA Scatter Plot ---
        with col_pca:
            st.markdown("### 🔭 Finding Outliers (PCA Space)",
                        help="Compressed core audio features into 2 dimensions via PCA. The gray background cloud represents the full global dataset, showing that recommendations successfully explore sparse outer boundaries.")

            if not seed_feat_df.empty and not rec_feat_df.empty:
                from sklearn.decomposition import PCA
                import numpy as np

                global_df = load_global_sample().copy()
                if not global_df.empty:
                    global_df['Type'] = '🌍 Global Dataset (All Tracks)'

                seed_plot_df = seed_feat_df.copy()
                if 'track_id' not in seed_plot_df.columns:
                    seed_plot_df = seed_plot_df.reset_index()
                if 'track_id' not in seed_plot_df.columns:
                    seed_plot_df['track_id'] = seed_ids[:len(seed_plot_df)]

                seed_plot_df['Type'] = '🎵 Current Run: Your Seeds'
                seed_id_to_name = {s['track_id']: s['track_name'] for s in st.session_state['seed_bank']}
                seed_plot_df['track_name'] = seed_plot_df['track_id'].map(seed_id_to_name)

                rec_plot_df = rec_feat_df.copy()
                rec_tracks = results['track_id'].tolist()
                if 'track_id' not in rec_plot_df.columns:
                    rec_plot_df = rec_plot_df.reset_index()
                if 'track_id' not in rec_plot_df.columns:
                    rec_plot_df['track_id'] = rec_tracks[:len(rec_plot_df)]

                rec_plot_df['Type'] = '🎸 Current Run: Recommendations'
                rec_id_to_name = dict(zip(rec_tracks, display_df['Song Title']))
                rec_plot_df['track_name'] = rec_plot_df['track_id'].map(rec_id_to_name)

                desired_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                                    'speechiness', 'valence']
                features = [f for f in desired_features if
                            f in global_df.columns and f in seed_plot_df.columns and f in rec_plot_df.columns]
                if not features:
                    features = [f for f in desired_features if f in seed_plot_df.columns]

                combined_df = pd.concat([global_df, seed_plot_df, rec_plot_df], ignore_index=True)
                combined_df = combined_df.dropna(subset=features)

                pca = PCA(n_components=2)
                pca_results = pca.fit_transform(combined_df[features])
                combined_df['PCA1'] = pca_results[:, 0]
                combined_df['PCA2'] = pca_results[:, 1]

                fig_pca = px.scatter(
                    combined_df,
                    x='PCA1', y='PCA2',
                    color='Type',
                    hover_data=['track_name', 'Type'],
                    color_discrete_map={
                        '🌍 Global Dataset (All Tracks)': 'rgba(200, 200, 200, 0.35)',
                        '🎵 Current Run: Your Seeds': '#0068c9',
                        '🎸 Current Run: Recommendations': '#ff4b4b'
                    }
                )

                fig_pca.update_traces(marker=dict(size=13, line=dict(width=1, color='black')),
                                      selector=dict(name='🎵 Current Run: Your Seeds'))
                fig_pca.update_traces(marker=dict(size=11, line=dict(width=1, color='black')),
                                      selector=dict(name='🎸 Current Run: Recommendations'))
                fig_pca.update_traces(marker=dict(size=4), selector=dict(name='🌍 Global Dataset (All Tracks)'))

                fig_pca.update_layout(
                    height=420,
                    margin=dict(t=20, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, title_text=""),
                    xaxis=dict(showticklabels=False, title=""),
                    yaxis=dict(showticklabels=False, title="")
                )

                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("Not enough audio data to generate the PCA space.")

        # --- 2. Acoustic Feature Correlation Heatmap (Diverging Scale & Precise Framing) ---
        with col_corr:
            st.markdown("### 🌡️ Global Dataset: Feature Correlation",
                        help="Calculated across all Kaggle tracks. While most features hover near independence, key pairs show strong linear dependence (e.g., acousticness vs. energy at −0.73), justifying dimensionality reduction via PCA to remove redundancy.")

            global_df_cache = load_global_sample()
            if not global_df_cache.empty:
                desired_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                                    'speechiness', 'valence']
                valid_corr_features = [f for f in desired_features if f in global_df_cache.columns]

                if len(valid_corr_features) > 1:
                    corr_matrix = global_df_cache[valid_corr_features].corr()

                    # FIX: Use a diverging colormap ('RdBu') centered at 0 so positive and negative values pull apart visually
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                        aspect="auto",
                        zmin=-1, zmax=1
                    )

                    fig_corr.update_layout(
                        height=420,
                        margin=dict(t=20, b=10, l=10, r=10),
                        xaxis_tickangle=-30
                    )

                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Insufficient numeric columns for correlation matrix.")
            else:
                st.info("Global background dataset not loaded.")
    # ==========================================
    # TAB 3: DYNAMIC BACKOFF STRATEGY
    # ==========================================
    with tab3:
        st.markdown("### ⚙️ Dynamic Backoff & Weight Redistribution")
        st.caption(
            "Visualizing how the engine automatically re-allocates accumulator weights when collaborative playlist data is missing.")

        col_backoff_info, col_weights_chart = st.columns([1, 1])

        with col_backoff_info:
            st.markdown("### 🔄 Two-State Dynamic Weighting",
                        help="Since seeds originate from the Kaggle dataset, they always have audio/artist metadata. The backoff strategy solely manages the presence or absence of collaborative playlist data.")

            st.markdown("""
            * **State 1: Dual-Source Blend (Optimal)**  
              *Condition:* Track data is found in both Kaggle and Playlist tables.  
              *Weights:* Audio Features (`25%`), Artist Graph (`40%`), Shared Playlists (`35%`).
            * **State 2: Kaggle-Only Fallback (`l3.empty`)**  
              *Condition:* Track has no co-occurrence data in the playlist dataset.  
              *Weights:* Playlist weight drops to **0%**, and weight is automatically redistributed evenly between Audio Features (`50%`) and Artist Collaborations (`50%`).
            """)

            st.info(
                "💡 **Architectural Win:** This guarantees robust scoring and zero division errors even for niche tracks lacking collaborative playlist history.")

        with col_weights_chart:
            st.markdown("### 📊 Layer Weight Shift Comparison",
                        help="Compares weight allocation between normal operation and the fallback state.")

            weight_data = pd.DataFrame({
                'Layer': ['Audio Features (l1)', 'Artist Graph (l2)', 'Shared Playlists (l3)',
                          'Audio Features (l1)', 'Artist Graph (l2)', 'Shared Playlists (l3)'],
                'State': ['State 1: Dual-Source', 'State 1: Dual-Source', 'State 1: Dual-Source',
                          'State 2: Kaggle-Only Fallback', 'State 2: Kaggle-Only Fallback',
                          'State 2: Kaggle-Only Fallback'],
                'Weight': [0.25, 0.40, 0.35, 0.50, 0.50, 0.00]
            })

            fig_weights = px.bar(
                weight_data,
                x='Layer',
                y='Weight',
                color='State',
                barmode='group',
                color_discrete_map={
                    'State 1: Dual-Source': '#28a745',
                    'State 2: Kaggle-Only Fallback': '#ff4b4b'
                },
                range_y=[0, 0.6]
            )

            fig_weights.update_layout(
                height=420,
                margin=dict(t=20, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title_text=""),
                xaxis_tickangle=-15
            )

            st.plotly_chart(fig_weights, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)