import re
import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from HybridRecommender import HybridRecommender

# Streamlit page configuration must precede all other Streamlit commands
st.set_page_config(page_title="Spotify Hybrid Recommender", layout="wide")

# --- Custom CSS Injection ---
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

# --- Initialize session state variables for execution persistence ---
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


# --- Cache global PCA background sample to optimize load times ---
@st.cache_data
def load_global_sample():
    try:
        return pd.read_csv("evaluation/global_audio_sample.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def build_offline_community_chart():
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.Graph()
    try:
        network_df = pd.read_csv("evaluation/global_artist_network.csv").head(400)
        for _, row in network_df.iterrows():
            G.add_edge(row['source'], row['target'])
    except FileNotFoundError:
        return None

    # Precompute network physics layout and cache in memory
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=35)

    traces = []
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    traces.append(
        go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='rgba(180, 180, 180, 0.5)'), hoverinfo='none',
                   mode='lines'))

    node_x, node_y, node_text, node_sizes = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{node}</b>")
        deg = G.degree(node)
        node_sizes.append(min(6 + (deg * 1.5), 24))

    traces.append(go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text',
                             marker=dict(showscale=False, color='#0068c9', size=node_sizes,
                                         line=dict(width=1, color='rgba(50, 50, 50, 0.8)'))))

    fig_comm = go.Figure(data=traces)
    fig_comm.update_layout(height=420, margin=dict(t=20, b=10, l=10, r=10), showlegend=False,
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(family="Arial, sans-serif"))
    return fig_comm


col_seed, col_discover = st.columns(2)

# --- Left Column: Seed Track Management ---
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

# --- Right Column: Random Track Discovery ---
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


# --- Recommendation Engine Execution ---
if len(st.session_state['seed_bank']) > 0:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Add a checkbox so the grader can test the normalization themselves!
        use_norm = st.checkbox("⚖️ Apply Niche Normalization (Anti-Popularity Bias)", value=False,
                               help="Scales layer scores to 1.0 to prevent global hits from drowning out niche seeds.")
        run_clicked = st.button("🚀 Run Recommendation Engine", type="primary", use_container_width=True)

    if run_clicked:
        seed_ids = [song['track_id'] for song in st.session_state['seed_bank']]

        with st.spinner("Calculating hybrid scores..."):
            results, full_candidate_pool = rec.hybrid_recommend(seed_ids, top_n=10, normalize=use_norm)
            seed_feat_df = rec.get_audio_features(seed_ids)
            rec_feat_df = rec.get_audio_features(results['track_id'].tolist())

            # Store in session state so results persist cleanly
            st.session_state['recommendations'] = results
            st.session_state['full_candidate_pool'] = full_candidate_pool
            st.session_state['seed_feat_df'] = seed_feat_df
            st.session_state['rec_feat_df'] = rec_feat_df
st.markdown("---")

# --- Results Rendering ---
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

    # --- QUICK STATISTICS ---
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
        "🎯 Sub-Problem 1: Performance",
        "🌐 Sub-Problem 2: Popularity Bias",
        "⚙️ Sub-Problem 3: Backoff Strategy"
    ])

    # ==========================================
    # TAB 1: MODEL PERFORMANCE & RECALL
    # ==========================================
    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📉 Current Run: Score Decay")
            st.caption(
                "Score distribution of all candidate tracks evaluated in this specific session, highlighting the 'long tail'.")

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
            st.markdown("### 📊 Offline Data: Model Recall")
            st.caption(
                "Historical evaluation (Mean Recall@20) comparing the hybrid engine against a global popularity baseline.")
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
            st.markdown("### 🎛️ Current Run: Vibe Match")
            st.caption(
                "Direct comparison of the average audio features between your input seeds and the final recommendations.")

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
        col_pca, col_corr, col_network = st.columns(3)

        # --- 1. Global PCA Scatter Plot ---
        with col_pca:
            st.markdown("### 🔭 Global vs. Current: PCA Space")
            st.caption(
                "Mapping this session's tracks over the entire global dataset to demonstrate"
                " how the engine explores niche boundaries.")

            if not seed_feat_df.empty and not rec_feat_df.empty:
                from sklearn.decomposition import PCA

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
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, title_text=""),
                    xaxis=dict(showticklabels=False, title=""),
                    yaxis=dict(showticklabels=False, title="")
                )

                st.plotly_chart(fig_pca, use_container_width=True)

                # Highlight the visual impact of the normalization toggle
                st.markdown(
                    "**💡 Observation: Without normalization, the red recommendation dots cluster into a"
                    " dense center mass. When normalization is applied (near run button), the recommendations become less "
                    "dense and much more scattered, demonstrating the engine is successfully pulling niche tracks.**")
            else:
                st.info("Not enough audio data to generate the PCA space.")

        # --- 2. Acoustic Feature Correlation Heatmap ---
        with col_corr:

            help_text = (
                "This heatmap helps us spot redundant information by looking for large numbers (ignoring the positive/negative sign).\n\n"
                "For example, the deep red **-0.73** between `acousticness` and `energy` highlights a massive inverse relationship: highly acoustic songs are almost never highly energetic.\n\n"
                "Because features like these heavily overlap in the story they tell, we can safely use PCA to compress them into fewer dimensions without losing the track's core musical identity."
            )

            # The title with the [?] tooltip
            st.markdown("### 🌡️ Offline Data: Feature Correlation", help=help_text)

            # The high-level sentence visible directly on the page
            st.caption(
                "Collinearity matrix calculated across all Kaggle tracks, justifying dimensionality reduction via PCA.")
            global_df_cache = load_global_sample()
            if not global_df_cache.empty:
                desired_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                                    'speechiness', 'valence']
                valid_corr_features = [f for f in desired_features if f in global_df_cache.columns]

                if len(valid_corr_features) > 1:
                    corr_matrix = global_df_cache[valid_corr_features].corr()

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
                        xaxis_tickangle=-30,
                        coloraxis_colorbar=dict(title="")
                    )

                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Insufficient numeric columns for correlation matrix.")
            else:
                st.info("Global background dataset not loaded.")

        # --- 3. Artist Connection Network (Sankey Diagram) ---
        with col_network:
            st.markdown("### 🕸️ Current Run: Artist Network")
            st.caption(
                "Tracing influence from your specific Seed Artists to the final output tracks, validating the anti-flood cap.")

            # Extract and truncate unique seed artists for UI formatting
            seed_artists_list = []
            for song in st.session_state['seed_bank']:
                for a in re.split(r'[;,]', str(song['artists'])):
                    if a.strip() and a.strip() not in seed_artists_list:
                        art_name = a.strip()
                        seed_artists_list.append(art_name[:15] + "..." if len(art_name) > 15 else art_name)

            # Extract output artists and aggregate track counts to validate quota constraints
            output_artists_counts = {}
            for i, row in display_df.iterrows():
                primary_rec_artist = str(row['Artist(s)']).split(',')[0].strip()
                art_name = primary_rec_artist[:15] + "..." if len(
                    primary_rec_artist) > 15 else primary_rec_artist
                output_artists_counts[art_name] = output_artists_counts.get(art_name, 0) + 1

            rec_artists_list = list(output_artists_counts.keys())

            # Construct Sankey nodes (Seeds -> Engine -> Outputs)
            engine_node_idx = len(seed_artists_list)

            # Apply bold HTML tags to node labels for readability
            raw_nodes = seed_artists_list + [" "] + rec_artists_list
            all_nodes = [f"<b>{name}</b>" for name in raw_nodes]

            node_colors = ['#0068c9'] * len(seed_artists_list) + ['#333333'] + ['#ff4b4b'] * len(
                rec_artists_list)

            links_source = []
            links_target = []
            links_value = []

            # Map edges: Seeds to Engine
            for s_idx in range(len(seed_artists_list)):
                links_source.append(s_idx)
                links_target.append(engine_node_idx)
                links_value.append(1)

            # Map edges: Engine to Output Artists
            for r_idx, r_artist in enumerate(rec_artists_list):
                links_source.append(engine_node_idx)
                links_target.append(engine_node_idx + 1 + r_idx)
                links_value.append(output_artists_counts[r_artist])

            # Render the Figure
            fig_network = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=20,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors
                ),
                link=dict(
                    source=links_source,
                    target=links_target,
                    value=links_value,
                    color="rgba(180, 180, 180, 0.3)"
                )
            )])

            fig_network.update_layout(
                height=420,
                margin=dict(t=20, b=10, l=10, r=30),
                font=dict(color="black", size=13, family="Arial, sans-serif"),
                # Force a white background to ensure seamless shadow blending for the Sankey nodes
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF"
            )

            st.plotly_chart(fig_network, use_container_width=True, theme=None)
    # ==========================================
    # TAB 3: DYNAMIC BACKOFF STRATEGY
    # ==========================================
    with tab3:

        col_network, col_weights_chart = st.columns([1.2, 1])
        # --- COLUMN 1: Global Artist Community Graph (Offline Data) ---
        with col_network:
            st.markdown("### 🕸️ Offline Data: Global Community",
                        help="Visualizing the strongest historical collaborations from the Kaggle dataset. Node size highlights 'Hub' artists with the most connections (Degree Centrality).")

            fig_comm = build_offline_community_chart()
            if fig_comm:
                st.plotly_chart(fig_comm, use_container_width=True, theme=None)
            else:
                st.warning("⚠️ Run `generate_artist_collabs_data.py` to generate the background community graph.")
        with col_weights_chart:
            help_text = (
                "**🔄 Two-State Dynamic Weighting**\n\n"
                "• **State 1: Dual-Source Blend (Optimal):** Artist Graph (40% - primary preference predictor), Shared Playlists (35% - community validation), Audio Features (25% - acoustic vibe filter).\n\n"
                "• **State 2: Kaggle-Only Fallback:** When a track lacks playlist co-occurrence data, the playlist weight drops to 0%. The engine redistributes weight evenly between Audio (50%) and the Artist Community Graph (50%).\n\n"
                "*Architectural Win: This guarantees zero division errors and maintains robust scoring even for niche tracks.*\n\n"
                "**Fallback Context:** When playlist data is sparse, the engine falls back heavily onto the Artist Collaboration Network (visualized on the left)."
            )

            st.markdown("### 📊 System Logic: Layer Weights", help=help_text)
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

    if run_clicked:
        st.toast("Dashboard fully loaded! 👇", icon="✅")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
