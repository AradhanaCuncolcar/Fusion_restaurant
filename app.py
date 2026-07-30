import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. SETUP & THEME CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Dubai Fusion Feasibility Dashboard", layout="wide")

GOLD = "#D4AF37"
TEAL = "#006666"
COPPER = "#B87333"
PLATINUM = "#E5E4E2"
DARK_BG = "#111111"

DISTINCT_COLORS = [GOLD, TEAL, COPPER, "#8B0000", PLATINUM, "#5F9EA0", "#D2691E", "#4682B4"]

# ---------------------------------------------------------
# 2. DATA GENERATION (Simulated 500 Survey Responses)
# ---------------------------------------------------------
@st.cache_data
def load_survey_data():
    np.random.seed(42)
    n = 500
    
    locations = ['Downtown Dubai', 'Dubai Marina', 'Jumeirah', 'DIFC', 'Business Bay']
    base_cuisines = ['Japanese', 'Levantine', 'Italian', 'Indian', 'Mexican']
    fusion_twists = ['Peruvian', 'Korean', 'Emirati', 'Thai', 'French']
    interior_vibes = ['Minimalist & Modern', 'Cozy & Intimate', 'Vibrant & Energetic', 'Premium Fine Dining']
    timings = ['Lunch (12PM-3PM)', 'Evening (5PM-8PM)', 'Dinner (8PM-11PM)', 'Late Night (11PM+)']
    
    data = {
        'Respondent_ID': range(1, n + 1),
        'Age': np.random.normal(loc=32, scale=7, size=n).astype(int),
        'Current_Residence': np.random.choice(locations, n, p=[0.25, 0.30, 0.20, 0.10, 0.15]),
        'Preferred_Restaurant_Location': np.random.choice(locations, n, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        'Base_Cuisine_Preference': np.random.choice(base_cuisines, n),
        'Fusion_Twist_Preference': np.random.choice(fusion_twists, n),
        'Preferred_Interior_Vibe': np.random.choice(interior_vibes, n, p=[0.3, 0.4, 0.2, 0.1]),
        'Preferred_Timing': np.random.choice(timings, n, p=[0.15, 0.25, 0.50, 0.10]),
        'Expected_Spend_Per_Person_AED': np.random.normal(loc=200, scale=60, size=n).clip(50, 600)
    }
    
    df = pd.DataFrame(data)
    df['Age'] = df['Age'].clip(18, 70)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '46+'])
    return df

df_raw = load_survey_data()

# ---------------------------------------------------------
# 3. GLOBAL SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Global Filters")
st.sidebar.markdown("Slice the 500 survey responses to uncover niche trends.")

min_age, max_age = int(df_raw['Age'].min()), int(df_raw['Age'].max())
selected_age = st.sidebar.slider("Respondent Age Range", min_age, max_age, (20, 50))
all_locations = df_raw['Preferred_Restaurant_Location'].unique().tolist()
selected_locations = st.sidebar.multiselect("Filter by Preferred Location", all_locations, default=all_locations)

df = df_raw[(df_raw['Age'] >= selected_age[0]) & (df_raw['Age'] <= selected_age[1]) & (df_raw['Preferred_Restaurant_Location'].isin(selected_locations))]

if df.empty:
    st.error("No survey data available for these filters.")
    st.stop()

# ---------------------------------------------------------
# 4. TAB ROUTING
# ---------------------------------------------------------
st.title("🍽️ Fusion Feasibility: Survey Analysis Dashboard")

tab_bp, tab_loc, tab_menu, tab_ops, tab_ml = st.tabs([
    "📑 Executive Summary",
    "📍 Location Target", 
    "🍣 Menu & Blends", 
    "🕰️ Interior & Timings", 
    "🤖 Spender Personas"
])

# =========================================================
# TAB 1: Executive Summary
# =========================================================
with tab_bp:
    st.subheader("Data-Driven Concept Blueprint")
    
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Survey Responses", f"{len(df)}", "Qualified Leads")
    f2.metric("Avg. Willingness to Pay", f"AED {df['Expected_Spend_Per_Person_AED'].mean():.0f}", "Per Person")
    f3.metric("Top Location Choice", df['Preferred_Restaurant_Location'].mode()[0])
    f4.metric("Peak Demand Window", df['Preferred_Timing'].mode()[0])
    
    bp_html = f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:20px; margin-top:25px;">
        <div style="background:#111; border-left:4px solid {GOLD}; padding:20px; border-radius:8px;">
            <h4 style="color:{GOLD}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">📍 1. Location Strategy</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">Based on the survey density, the optimal real estate targets should be narrowed down to the top selected neighborhoods. Proximity to these hubs will capture both residential foot traffic and corporate dining budgets.</p>
        </div>
        <div style="background:#111; border-left:4px solid {TEAL}; padding:20px; border-radius:8px;">
            <h4 style="color:{TEAL}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🍣 2. Culinary Blending</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">Rather than guessing, the menu development will be strictly guided by the highest intersecting pairs of <b>Base Cuisines</b> and <b>Fusion Twists</b> identified in the heatmap analysis, ensuring immediate market acceptance.</p>
        </div>
        <div style="background:#111; border-left:4px solid {COPPER}; padding:20px; border-radius:8px;">
            <h4 style="color:{COPPER}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🛋️ 3. Interior & Seating</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">The architectural and interior design budget will be allocated to match the dominant 'Vibe' preference of the respondents. Seating capacity models will be optimized for the chosen dining style to maximize table turnover.</p>
        </div>
        <div style="background:#111; border-left:4px solid {PLATINUM}; padding:20px; border-radius:8px;">
            <h4 style="color:{PLATINUM}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🕰️ 4. Operational Timings</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">Staffing rotas, inventory deliveries, and peak marketing spend will be synchronized exactly with the highest-rated dining windows to maximize efficiency and minimize dead operational hours.</p>
        </div>
    </div>
    """
    st.markdown(bp_html, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🚀 Pre-Launch Execution Timeline")
    
    timeline_html = f"""
    <div style="border-left:3px solid {TEAL}; padding-left:20px; margin-top:10px;">
        <div style="margin-bottom:20px; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Phase 1: Real Estate & Menu R&D</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Action:</b> Initiate commercial lease tours exclusively in the top 2 preferred neighborhoods. Commission a consulting chef to develop tasting menus specifically for the top 3 requested Base+Twist fusion pairings from the survey.</div>
        </div>
        <div style="margin-bottom:20px; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Phase 2: Fit-Out & Operational Mapping</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Action:</b> Finalize interior design contracts to match the dominant survey vibe (e.g., Cozy & Intimate). Map out shift scheduling ensuring maximum floor staffing during the identified 'Peak Demand Window'.</div>
        </div>
        <div style="margin-bottom:0; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Phase 3: Targeted Soft Launch</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Action:</b> Invite the original survey respondents to an exclusive soft launch. Test the average spend per head against the projected survey baseline (AED 200) to calibrate final menu pricing.</div>
        </div>
    </div>
    """
    st.markdown(timeline_html, unsafe_allow_html=True)

# =========================================================
# TAB 2: Location Target
# =========================================================
with tab_loc:
    st.subheader("Where Should We Open?")
    
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            pref_loc = df['Preferred_Restaurant_Location'].value_counts().reset_index()
            pref_loc.columns = ['Location', 'Votes']
            fig_loc = px.bar(pref_loc, x="Location", y="Votes", color="Location", title="Ideal Restaurant Location", color_discrete_sequence=DISTINCT_COLORS, text_auto=True, template="plotly_dark")
            fig_loc.update_traces(textposition="outside", cliponaxis=False)
            fig_loc.update_layout(margin=dict(t=50, b=20, l=20, r=20), showlegend=False, yaxis_title="Number of Respondents")
            st.plotly_chart(fig_loc, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            res_loc = df['Current_Residence'].value_counts().reset_index()
            res_loc.columns = ['Residence', 'Count']
            fig_res = px.pie(res_loc, names="Residence", values="Count", title="Where Do Our Respondents Live?", color_discrete_sequence=DISTINCT_COLORS, template="plotly_dark", hole=0.4)
            fig_res.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_res, use_container_width=True)

# =========================================================
# TAB 3: Menu & Blends
# =========================================================
with tab_menu:
    st.subheader("What Cuisines Should We Blend?")
    
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            base_counts = df['Base_Cuisine_Preference'].value_counts().reset_index()
            base_counts.columns = ['Base Cuisine', 'Votes']
            fig_base = px.bar(base_counts, x="Votes", y="Base Cuisine", color="Base Cuisine", orientation='h', title="Core Foundation Cuisine", color_discrete_sequence=DISTINCT_COLORS, text_auto=True, template="plotly_dark")
            fig_base.update_traces(textposition="outside", cliponaxis=False)
            fig_base.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=50, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_base, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            twist_counts = df['Fusion_Twist_Preference'].value_counts().reset_index()
            twist_counts.columns = ['Fusion Twist', 'Votes']
            fig_twist = px.bar(twist_counts, x="Votes", y="Fusion Twist", color="Fusion Twist", orientation='h', title="Requested Fusion Twist", color_discrete_sequence=DISTINCT_COLORS[::-1], text_auto=True, template="plotly_dark")
            fig_twist.update_traces(textposition="outside", cliponaxis=False)
            fig_twist.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=50, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_twist, use_container_width=True)
            
    st.markdown("### The Ultimate Fusion Matrix")
    st.markdown("This heatmap shows the exact crossover combinations respondents asked for.")
    with st.container(border=True):
        heatmap_data = pd.crosstab(df['Base_Cuisine_Preference'], df['Fusion_Twist_Preference'])
        fig_heat = px.density_heatmap(heatmap_data, x=heatmap_data.columns, y=heatmap_data.index, title="Base Cuisine vs. Fusion Twist Demand", color_continuous_scale="copper", text_auto=True, template="plotly_dark")
        fig_heat.update_layout(xaxis_title="Fusion Twist", yaxis_title="Base Cuisine", margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB 4: Interior & Timings
# =========================================================
with tab_ops:
    st.subheader("How Should We Design & Operate?")
    
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            vibe_counts = df['Preferred_Interior_Vibe'].value_counts().reset_index()
            vibe_counts.columns = ['Interior Vibe', 'Votes']
            fig_vibe = px.bar(vibe_counts, x="Interior Vibe", y="Votes", color="Interior Vibe", title="Interior & Seating Style", color_discrete_sequence=DISTINCT_COLORS, text_auto=True, template="plotly_dark")
            fig_vibe.update_traces(textposition="outside", cliponaxis=False)
            fig_vibe.update_layout(margin=dict(t=50, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_vibe, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            time_counts = df['Preferred_Timing'].value_counts().reset_index()
            time_counts.columns = ['Timing', 'Votes']
            # Sort chronologically
            time_order = ['Lunch (12PM-3PM)', 'Evening (5PM-8PM)', 'Dinner (8PM-11PM)', 'Late Night (11PM+)']
            fig_time = px.bar(time_counts, x="Timing", y="Votes", color="Timing", title="Peak Operational Timings", color_discrete_sequence=DISTINCT_COLORS, text_auto=True, template="plotly_dark", category_orders={"Timing": time_order})
            fig_time.update_traces(textposition="outside", cliponaxis=False)
            fig_time.update_layout(margin=dict(t=50, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)

# =========================================================
# TAB 5: Spender Personas
# =========================================================
with tab_ml:
    st.subheader("Who Is Our Customer?")
    
    # Simple K-Means on Age and Expected Spend
    features = ['Expected_Spend_Per_Person_AED', 'Age']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
    df['Persona'] = df['Cluster'].map({0: 'Value Seekers', 1: 'Premium Diners', 2: 'Mid-Tier Explorers'})
    
    with st.container(border=True):
        fig_cluster = px.scatter(df, x='Age', y='Expected_Spend_Per_Person_AED', color='Persona', color_discrete_sequence=[GOLD, TEAL, PLATINUM], title="Customer Segmentation (Spend vs. Age)", template="plotly_dark", opacity=0.8)
        fig_cluster.update_layout(margin=dict(t=50, b=20, l=20, r=20), yaxis_title="Expected Spend (AED)")
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.markdown("#### Segment Synthesis")
    cluster_means = df.groupby('Persona')[features].mean()
    
    p_html = f"""<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">"""
    for c_name, row in cluster_means.iterrows():
        if c_name == 'Value Seekers':
            color, icon, desc = PLATINUM, "👥", "Highly cost-conscious. Focus on volume and turning tables quickly during lunch hours."
        elif c_name == 'Premium Diners':
            color, icon, desc = GOLD, "💎", "High willingness to pay. Will demand a 'Premium Fine Dining' interior and intricate fusion dishes."
        else:
            color, icon, desc = TEAL, "🧭", "The reliable core demographic. They expect a solid ambiance and good portions for a fair price."
        
        p_html += f"""<div style="background:#111; border:2px solid {color}; border-radius:10px; padding:20px;">"""
        p_html += f"""<div style="display:flex; align-items:center; gap:10px; margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:10px;"><div style="font-size:30px;">{icon}</div><h4 style="color:{color}; margin:0; font-weight:900; text-transform:uppercase;">{c_name}</h4></div>"""
        p_html += f"""<div style="color:#ccc; font-size:0.95em; line-height:1.5;"><p><b>Avg Age:</b> {int(row['Age'])}</p><p><b>Expected Spend:</b> AED {int(row['Expected_Spend_Per_Person_AED'])}</p><p><strong style="color:{color};">Strategy:</strong> {desc}</p></div></div>"""
    p_html += "</div>"
    st.markdown(p_html, unsafe_allow_html=True)
