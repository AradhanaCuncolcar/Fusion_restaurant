import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import warnings

# ---------------------------------------------------------
# IMPORT MENU DATA FROM menu.py
# ---------------------------------------------------------
try:
    from menu import MENU
except ImportError:
    # Fallback and warning if menu.py is not present in the same directory
    MENU = []
    st.warning("⚠️ Could not find 'menu.py'. Please make sure the file is in the same folder as this script, or your recommendations will be empty.")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# SETUP & THEME CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Dubai Fusion Strategy & Launch Model", page_icon="✦", layout="wide")

GOLD = "#D4AF37"
TEAL = "#006666"
COPPER = "#B87333"
PLATINUM = "#E5E4E2"
DARK_BG = "#111111"
DISTINCT_COLORS = [GOLD, TEAL, COPPER, "#8B0000", PLATINUM, "#5F9EA0", "#D2691E", "#4682B4"]

# AI Concierge Styling
st.markdown("""
<style>
    .eyebrow { color: #a89c86; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; }
    .ticket {
        position: relative; background: #2a2620; border: 1px solid #3a352b;
        border-radius: 10px; padding: 18px 18px 14px; margin-bottom: 14px;
    }
    .ticket::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px;
        background: repeating-linear-gradient(180deg, #c9a227 0 8px, transparent 8px 16px);
    }
    .ticket h3 { margin: 0 0 6px 10px; font-size: 19px; color: #f4efe4; font-family: Georgia, serif; }
    .ticket-price { color: #c9a227; font-weight: 600; }
    .ticket-reason { margin: 0 0 6px 10px; font-size: 14.5px; line-height: 1.5; color: #f4efe4; }
    .ticket-pairing { margin: 0 0 4px 10px; font-size: 13px; color: #4f7d70; font-style: italic; }
    .ticket-source { margin: 6px 0 0 10px; font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: #a89c86; }
    .concierge-note { font-size: 18px; color: #e4c766; line-height: 1.5; margin-bottom: 18px; }
    div[data-testid="stForm"] div.stButton > button { background: #c9a227; color: #201b0e; font-weight: 600; border: none; }
    div[data-testid="stForm"] div.stButton > button:hover { background: #e4c766; color: #201b0e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA GENERATION (Simulating 500 Survey Responses)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500  
    
    areas = ['Downtown Dubai', 'Dubai Marina', 'Jumeirah', 'DIFC', 'Business Bay']
    cuisines_pool = ['Japanese', 'Peruvian', 'Levantine', 'Italian', 'Indian', 'Mexican']
    timings = ['Lunch (12PM-3PM)', 'Dinner (7PM-10PM)', 'Late Night (10PM+)']
    seating = ['Intimate Fine Dining', 'High-Energy Bar/Lounge', 'Casual & Relaxed', 'Outdoor Terrace']
    
    data = {
        'Respondent_ID': range(1, n + 1),
        'Age': np.random.normal(loc=32, scale=7, size=n).astype(int),
        'Area_Preference': np.random.choice(areas, n, p=[0.30, 0.25, 0.15, 0.20, 0.10]),
        'Preferred_Timing': np.random.choice(timings, n, p=[0.20, 0.60, 0.20]),
        'Seating_Vibe': np.random.choice(seating, n, p=[0.30, 0.35, 0.20, 0.15]),
        'Spend_Capacity (AED)': np.random.normal(loc=280, scale=60, size=n).clip(100, 800)
    }
    df = pd.DataFrame(data)
    
    df['Cuisines_Enjoyed'] = [", ".join(np.random.choice(cuisines_pool, np.random.randint(2, 4), replace=False)) for _ in range(n)]
    df['Age'] = df['Age'].clip(18, 70)
    
    return df

df_raw = load_data()

# ---------------------------------------------------------
# MENU CONCIERGE CONSTANTS & FALLBACK LOGIC
# ---------------------------------------------------------
OCCASIONS = ["Solo treat", "Date night", "Family", "Business", "Celebrating"]
SPICE = ["Mild", "Medium", "Spicy", "Very spicy"]
BUDGETS = ["Casual", "Mid-range", "Premium", "Fine dining"]
DIETARY_OPTIONS = ["vegetarian", "vegan", "gluten_free", "no_pork", "no_alcohol"]
CUISINES = ["Levantine", "Japanese", "Indian", "Italian", "Mexican", "Emirati", "Korean", "Peruvian", "French"]
BUDGET_CEILING_AED = {"Casual": 100, "Mid-range": 250, "Premium": 500, "Fine dining": 100000}
SPICE_CAP = {"Mild": 0, "Medium": 1, "Spicy": 2, "Very spicy": 3}

def recommend_fallback(prefs: dict) -> dict:
    ceiling = BUDGET_CEILING_AED.get(prefs["budget_pref"], 250)
    spice_cap = SPICE_CAP.get(prefs["spice_tolerance"], 2)

    candidates = []
    for dish in MENU:
        if dish.get("category", "") == "Beverages":
            continue
        if dish.get("price_aed", 0) > ceiling * 1.3:
            continue
        if dish.get("spice_level", 0) > spice_cap:
            continue
        if any(d not in dish.get("dietary", []) for d in prefs["dietary"]):
            continue
        score = sum(2 for c in prefs["favorite_cuisines"] if c.lower() in dish.get("cuisine_fusion", "").lower())
        score += 1 if dish.get("signature") else 0
        candidates.append((score, dish))

    candidates.sort(key=lambda x: x[0], reverse=True)
    picks = [d for _, d in candidates[:3]] if candidates else MENU[:3]

    return {
        "recommendations": [
            {
                "dish_id": d.get("id", "0"), "name": d.get("name", "Unknown"), "price_aed": d.get("price_aed", 0),
                "reason": f"A {d.get('cuisine_fusion', 'fusion')} {d.get('category', 'dish').lower()[:-1] if d.get('category') else 'dish'} that fits your preferences.",
                "pairing": None,
            } for d in picks
        ],
        "concierge_note": "Here are a few dishes picked for you based on your answers.",
        "source": "fallback",
    }

# ---------------------------------------------------------
# GLOBAL SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Global Filters")
st.sidebar.markdown("Slice the survey data to target specific demographics.")
min_age, max_age = int(df_raw['Age'].min()), int(df_raw['Age'].max())
selected_age = st.sidebar.slider("Age Range", min_age, max_age, (20, 50))
all_areas = df_raw['Area_Preference'].unique().tolist()
selected_areas = st.sidebar.multiselect("Target Location(s)", all_areas, default=all_areas)

df = df_raw[(df_raw['Age'] >= selected_age[0]) & (df_raw['Age'] <= selected_age[1]) & (df_raw['Area_Preference'].isin(selected_areas))]
if df.empty:
    st.error("No survey data matches the selected filters.")
    st.stop()

# ---------------------------------------------------------
# TAB ROUTING
# ---------------------------------------------------------
st.title("🍽️ Dubai Fusion: Pre-Launch Survey Analytics")

tab_bp, tab_demo, tab_menu, tab_ops, tab_fin, tab_whatif, tab_ai = st.tabs([
    "📑 Executive Blueprint",
    "📊 Demographics & Location", 
    "🥢 Cuisine & Menu Engineering", 
    "🛋️ Atmosphere & Operations",
    "📈 Financial Projections",
    "🎛️ What-If & Sensitivity Analysis",
    "🤖 AI Menu Concierge"
])

# =========================================================
# TAB 1: Executive Blueprint
# =========================================================
with tab_bp:
    st.subheader("Data-Driven Launch Strategy (Based on 500 Responses)")
    
    f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1.4])
    f1.metric("Total Survey Validations", f"{len(df)}", "Respondents")
    f2.metric("Market AOV (Spend)", f"AED {df['Spend_Capacity (AED)'].mean():.0f}", "Avg Capacity")
    f3.metric("Peak Demand Window", df['Preferred_Timing'].mode()[0], "Highest Volume")
    f4.metric("Top Location Request", df['Area_Preference'].mode()[0], "Prime Real Estate")
    
    bp_html = f"""<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap:20px; margin-top:25px; align-items:stretch;"><div style="background:#111; border-left:4px solid {GOLD}; padding:20px; border-radius:8px; height:100%; box-sizing:border-box;"><h4 style="color:{GOLD}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">📍 Location Strategy</h4><p style="color:#eee; font-size:0.95em; line-height:1.6; margin-bottom:0;">Survey data heavily indexes toward corporate/premium hubs. Targeting real estate in <b>{df['Area_Preference'].mode()[0]}</b> captures the highest concentration of high-spend individuals. A secondary outpost in DIFC caters to the corporate lunch crowd.</p></div><div style="background:#111; border-left:4px solid {TEAL}; padding:20px; border-radius:8px; height:100%; box-sizing:border-box;"><h4 style="color:{TEAL}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🛋️ Interior & Atmosphere</h4><p style="color:#eee; font-size:0.95em; line-height:1.6; margin-bottom:0;">The audience demands a dual-experience environment. The layout must prioritize a <b>{df['Seating_Vibe'].mode()[0]}</b> for the evening rush, transitioning smoothly from an intimate dining setting into a high-energy lounge to maximize late-night beverage sales.</p></div><div style="background:#111; border-left:4px solid {COPPER}; padding:20px; border-radius:8px; height:100%; box-sizing:border-box;"><h4 style="color:{COPPER}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🥢 Menu & Blending</h4><p style="color:#eee; font-size:0.95em; line-height:1.6; margin-bottom:0;">The data confirms a strong appetite for bold combinations. Menu engineering should focus heavily on the top intersecting cuisine pairs discovered in the survey, allowing for a streamlined supply chain by cross-utilizing premium ingredients.</p></div><div style="background:#111; border-left:4px solid {PLATINUM}; padding:20px; border-radius:8px; height:100%; box-sizing:border-box;"><h4 style="color:{PLATINUM}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">⏱️ Operational Efficiency</h4><p style="color:#eee; font-size:0.95em; line-height:1.6; margin-bottom:0;">Peak staffing and inventory must align with the <b>{df['Preferred_Timing'].mode()[0]}</b> window. Table turnover rates must be tightly managed during this period, whereas lunch operations can operate with a leaner team utilizing automation.</p></div></div>"""
    st.markdown(bp_html, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🚀 Real-World Execution Timeline")
    
    timeline_html = f"""<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:20px; margin-top:20px; align-items:stretch;"><div style="background:#111; border-top:5px solid {GOLD}; padding:25px; border-radius:10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); height:100%; box-sizing:border-box;"><div style="font-size:45px; margin-bottom:15px;">🏗️</div><div style="color:#aaa; font-size:0.85em; font-weight:bold; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:5px;">Days 1 - 30</div><h4 style="color:{GOLD}; margin:0 0 15px 0; font-weight:900; font-size:1.3em;">FOUNDATION & R&D</h4><p style="color:#eee; font-size:0.95em; line-height:1.6; margin:0;"><b style="color:white;">Action:</b> Leverage survey data to negotiate lease terms in the top-voted neighborhood. Finalize R&D for the top 3 fusion blends. Secure supply chain contracts for cross-utilized ingredients.</p></div><div style="background:#111; border-top:5px solid {COPPER}; padding:25px; border-radius:10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); height:100%; box-sizing:border-box;"><div style="font-size:45px; margin-bottom:15px;">⚙️</div><div style="color:#aaa; font-size:0.85em; font-weight:bold; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:5px;">Days 31 - 60</div><h4 style="color:{COPPER}; margin:0 0 15px 0; font-weight:900; font-size:1.3em;">OPS & STAFFING</h4><p style="color:#eee; font-size:0.95em; line-height:1.6; margin:0;"><b style="color:white;">Action:</b> Zone the restaurant seating to match the survey's vibe preferences (e.g., 60% High-Energy Lounge, 40% Intimate Dining). Hire front-of-house staff optimized for the identified peak operational hours.</p></div><div style="background:#111; border-top:5px solid {TEAL}; padding:25px; border-radius:10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); height:100%; box-sizing:border-box;"><div style="font-size:45px; margin-bottom:15px;">🚀</div><div style="color:#aaa; font-size:0.85em; font-weight:bold; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:5px;">Days 61 - 90</div><h4 style="color:{TEAL}; margin:0 0 15px 0; font-weight:900; font-size:1.3em;">SOFT LAUNCH</h4><p style="color:#eee; font-size:0.95em; line-height:1.6; margin:0;"><b style="color:white;">Action:</b> Send exclusive soft-launch invitations to the 500 original survey respondents. Use their initial feedback to calibrate menu pricing and service flow before the official grand opening.</p></div></div>"""
    st.markdown(timeline_html, unsafe_allow_html=True)

# =========================================================
# TAB 2: Demographics & Location
# =========================================================
with tab_demo:
    st.subheader("Where and Who is Your Customer?")
    
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            fig_age = px.histogram(df, x="Age", nbins=15, title="Target Demographic (Age)", color_discrete_sequence=[TEAL], text_auto=True, template="plotly_dark")
            fig_age.update_traces(marker_line_width=1.5, marker_line_color=PLATINUM, textposition="outside", cliponaxis=False)
            fig_age.update_layout(bargap=0, margin=dict(t=60, b=20, l=20, r=20), xaxis_title="Age (Years)", yaxis_title="Customer Count")
            st.plotly_chart(fig_age, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            area_counts = df['Area_Preference'].value_counts().reset_index()
            area_counts.columns = ['Area', 'Count']
            fig_area = px.bar(area_counts, x="Area", y="Count", color="Area", title="Top Requested Locations", color_discrete_sequence=DISTINCT_COLORS, text_auto=True, template="plotly_dark")
            fig_area.update_traces(textposition="outside", cliponaxis=False)
            fig_area.update_layout(margin=dict(t=60, b=20, l=20, r=20), showlegend=False, xaxis_title="Neighborhood", yaxis_title="Votes")
            st.plotly_chart(fig_area, use_container_width=True)

# =========================================================
# TAB 3: Cuisine & Menu Engineering
# =========================================================
with tab_menu:
    st.subheader("Culinary Demand & Fusion Opportunities")
    
    c1, c2 = st.columns(2)
    
    with c1:
        with st.container(border=True):
            cuisines_exploded = df['Cuisines_Enjoyed'].str.split(', ').explode().value_counts().reset_index()
            cuisines_exploded.columns = ['Cuisine', 'Count']
            fig_cuisine = px.bar(cuisines_exploded, x="Count", y="Cuisine", color="Cuisine", orientation='h', title="Base Cuisine Popularity", color_discrete_sequence=DISTINCT_COLORS, text_auto=True, template="plotly_dark")
            fig_cuisine.update_traces(textposition="outside", cliponaxis=False)
            fig_cuisine.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=60, b=20, l=20, r=20), showlegend=False, xaxis_title="Total Votes", yaxis_title="")
            st.plotly_chart(fig_cuisine, use_container_width=True)

    with c2:
        with st.container(border=True):
            pair_counts = {}
            for row in df['Cuisines_Enjoyed']:
                items = sorted(row.split(', '))
                for pair in combinations(items, 2):
                    pair_counts[f"{pair[0]} x {pair[1]}"] = pair_counts.get(f"{pair[0]} x {pair[1]}", 0) + 1
                    
            df_pairs = pd.DataFrame(list(pair_counts.items()), columns=['Fusion Blend', 'Demand Score']).sort_values('Demand Score', ascending=False).head(5)
            
            fig_pairs = px.bar(df_pairs, x="Fusion Blend", y="Demand Score", title="Top 5 Most Requested Fusion Blends", color_discrete_sequence=[GOLD], text_auto=True, template="plotly_dark")
            fig_pairs.update_traces(textposition="outside", cliponaxis=False)
            fig_pairs.update_layout(margin=dict(t=60, b=20, l=20, r=20), xaxis_title="Cuisine Pairings", yaxis_title="Co-occurrence Frequency")
            st.plotly_chart(fig_pairs, use_container_width=True)

# =========================================================
# TAB 4: Atmosphere & Operations
# =========================================================
with tab_ops:
    st.subheader("Optimizing Space and Time")
    
    c1, c2 = st.columns(2)
    
    with c1:
        with st.container(border=True):
            vibe_counts = df['Seating_Vibe'].value_counts().reset_index()
            vibe_counts.columns = ['Vibe', 'Count']
            fig_vibe = px.bar(vibe_counts, x="Vibe", y="Count", color="Vibe", title="Desired Restaurant Atmosphere", color_discrete_sequence=DISTINCT_COLORS, text_auto=True, template="plotly_dark")
            fig_vibe.update_traces(textposition="outside", cliponaxis=False)
            fig_vibe.update_layout(margin=dict(t=60, b=20, l=20, r=20), showlegend=False, xaxis_title="", yaxis_title="Votes")
            st.plotly_chart(fig_vibe, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            time_counts = df['Preferred_Timing'].value_counts().reset_index()
            time_counts.columns = ['Timing', 'Count']
            fig_time = px.pie(time_counts, values='Count', names='Timing', title="Peak Operational Demand", color_discrete_sequence=[TEAL, COPPER, PLATINUM], hole=0.4, template="plotly_dark")
            fig_time.update_traces(textposition='outside', textinfo='percent+label')
            fig_time.update_layout(margin=dict(t=60, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)

# =========================================================
# TAB 5: Financials & Elasticity
# =========================================================
with tab_fin:
    st.subheader("Menu Pricing & Revenue Projection")
    
    total_expected_diners = len(df) * 4 
    base_avg_spend = df['Spend_Capacity (AED)'].mean()
    base_revenue = total_expected_diners * base_avg_spend
    
    st.markdown(f"**Baseline Monthly Revenue Model:** Based on targeting {total_expected_diners:,} monthly covers at the survey's average capacity of AED {base_avg_spend:.0f}.")
    
    price_increase = st.slider("Test Menu Price Increase Premium (%)", min_value=0, max_value=30, value=15, step=1)
    
    scaled_drop_off = (price_increase / 100.0) * -0.65 
    
    new_visits = total_expected_diners * (1 + scaled_drop_off)
    new_spend = base_avg_spend * (1 + (price_increase/100))
    new_revenue = new_visits * new_spend
    
    price_gain = base_revenue * (price_increase / 100)
    volume_loss = new_revenue - (base_revenue + price_gain)

    with st.container(border=True):
        fig_waterfall = go.Figure(go.Waterfall(
            name="Revenue Impact", 
            orientation="v", 
            measure=["absolute", "relative", "relative", "total"], 
            x=["Baseline Projection", "Premium Pricing Gain", "Lost Footfall (Elasticity)", "Net Projected Revenue"], 
            y=[base_revenue, price_gain, volume_loss, new_revenue],
            text=[f"AED {base_revenue/1000:,.0f}k", f"+AED {price_gain/1000:,.0f}k", f"-AED {abs(volume_loss)/1000:,.0f}k", f"AED {new_revenue/1000:,.0f}k"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}}, 
            increasing={"marker": {"color": GOLD}}, 
            decreasing={"marker": {"color": COPPER}}, 
            totals={"marker": {"color": TEAL}}
        ))
        fig_waterfall.update_traces(cliponaxis=False)
        fig_waterfall.update_layout(
            title="Strategic Pricing Elasticity Analysis", 
            template="plotly_dark", 
            margin=dict(t=60, b=40, l=40, r=40),
            yaxis_title="Monthly Revenue (AED)",
            xaxis_title="Financial Drivers"
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

# =========================================================
# TAB 6: What-If & Sensitivity Analysis
# =========================================================
with tab_whatif:
    st.subheader("🎛️ Interactive What-If Scenario Simulator")
    st.markdown("Use the advanced controls below to stress-test your restaurant concept against custom target subsets, marketing premiums, and guest capacity adjustments in real-time.")
    
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        sim_location = st.multiselect("Simulate Locations", df_raw['Area_Preference'].unique().tolist(), default=df_raw['Area_Preference'].unique().tolist())
    with col_w2:
        sim_vibe = st.multiselect("Simulate Atmosphere Vibes", df_raw['Seating_Vibe'].unique().tolist(), default=df_raw['Seating_Vibe'].unique().tolist())
    with col_w3:
        marketing_boost = st.slider("Targeted Marketing Spend Multiplier (%)", 0, 100, 20, step=5)
        
    sim_df = df_raw[(df_raw['Area_Preference'].isin(sim_location)) & (df_raw['Seating_Vibe'].isin(sim_vibe))]
    
    if sim_df.empty:
        st.warning("⚠️ Your current simulation filters are too restrictive. Please select at least one location and vibe.")
    else:
        sim_respondents = len(sim_df)
        sim_aov = sim_df['Spend_Capacity (AED)'].mean() * (1 + (marketing_boost / 300.0)) 
        sim_monthly_rev = sim_respondents * 4 * sim_aov
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Simulated Market Reach", f"{sim_respondents} Profiles", f"{(sim_respondents/len(df_raw))*100:.1f}% of Survey")
        k2.metric("Simulated Average Spend (AOV)", f"AED {sim_aov:.0f}", f"+{marketing_boost}% Marketing Lift")
        k3.metric("Simulated Monthly Potential", f"AED {sim_monthly_rev/1000:,.0f}k", "Projected Yield")
        
        st.divider()
        st.markdown("### 🌐 Concept Alignment Radar Analysis")
        st.markdown("The radar graph below maps how strongly your selected simulation parameters align across core operational pillars compared to the baseline market.")
        
        categories = ['Spend Potential', 'Volume Density', 'Corporate Fit', 'Nightlife Appeal', 'Dining Preference']
        baseline_spend = df_raw['Spend_Capacity (AED)'].mean()
        sim_spend_score = min(100, (sim_aov / baseline_spend) * 50)
        sim_density_score = min(100, (sim_respondents / len(df_raw)) * 100)
        sim_corp_score = 80 if 'Downtown Dubai' in sim_location or 'DIFC' in sim_location else 40
        sim_night_score = 85 if 'High-Energy Bar/Lounge' in sim_vibe else 50
        sim_dining_score = 90 if 'Intimate Fine Dining' in sim_vibe else 60
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[sim_spend_score, sim_density_score, sim_corp_score, sim_night_score, sim_dining_score],
            theta=categories,
            fill='toself',
            name='Simulated Concept',
            line_color=GOLD
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[50, 80, 70, 65, 70], 
            theta=categories,
            fill='toself',
            name='Market Benchmark',
            line_color=TEAL
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template="plotly_dark",
            margin=dict(t=40, b=40, l=40, r=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        with st.container(border=True):
            st.plotly_chart(fig_radar, use_container_width=True)
            
        st.markdown("### 💡 Professional Scenario Insights")
        
        insight_html = f"""
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:20px; margin-top:15px; align-items:stretch;">
            <div style="background:#111; border-top:5px solid {GOLD}; padding:25px; border-radius:10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <h4 style="color:{GOLD}; margin-top:0; font-weight:900;">🎯 Audience Viability</h4>
                <p style="color:#eee; font-size:0.95em; line-height:1.6;">Your selected configuration captures <b>{sim_respondents} active survey respondents</b>. The spend capacity index sits at <b>AED {sim_aov:.0f}</b>, indicating strong unit economic resilience against standard operational overheads.</p>
            </div>
            <div style="background:#111; border-top:5px solid {COPPER}; padding:25px; border-radius:10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <h4 style="color:{COPPER}; margin-top:0; font-weight:900;">⚠️ Sensitivity Warning</h4>
                <p style="color:#eee; font-size:0.95em; line-height:1.6;">Narrowing location parameters too aggressively increases customer acquisition costs (CAC). Ensure marketing expenditure multipliers do not outpace the <b>AED {sim_monthly_rev/1000:,.0f}k monthly revenue ceiling</b> shown above.</p>
            </div>
            <div style="background:#111; border-top:5px solid {TEAL}; padding:25px; border-radius:10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <h4 style="color:{TEAL}; margin-top:0; font-weight:900;">🚀 Strategic Recommendation</h4>
                <p style="color:#eee; font-size:0.95em; line-height:1.6;">Based on the radar symmetry, balancing your vibe selection between lounge and dining elements optimizes both early-evening food sales and late-night beverage margins.</p>
            </div>
        </div>
        """
        st.markdown(insight_html, unsafe_allow_html=True)

# =========================================================
# TAB 7: Menu Concierge (Local Fallback)
# =========================================================
with tab_ai:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<p class="eyebrow">Tonight\'s table</p>', unsafe_allow_html=True)
        st.markdown('<h1 style="color:#f4efe4; font-family:Georgia, serif; margin-top:0;">✦ The Concierge</h1>', unsafe_allow_html=True)
        st.caption("Answer a few quick questions to receive 3 curated dishes from tonight's menu.")

        with st.form("quiz"):
            occasion = st.selectbox("Who's joining you tonight?", OCCASIONS)
            c1, c2 = st.columns(2)
            with c1:
                spice_tolerance = st.selectbox("Spice tolerance", SPICE, index=1)
            with c2:
                budget_pref = st.selectbox("Budget, per person", BUDGETS, index=1)
            adventurousness = st.slider("Familiar \u2192 Adventurous", 0, 100, 50)
            dietary = st.multiselect("Dietary needs (optional)", DIETARY_OPTIONS)
            favorite_cuisines = st.multiselect("Cuisines you already love (optional)", CUISINES)
            notes = st.text_area("Anything else? (optional)", placeholder="e.g. celebrating an anniversary, allergic to shellfish...")
            submitted = st.form_submit_button("See my picks")

        if submitted:
            if not MENU:
                st.error("The menu data is missing. Please add your `menu.py` file to the repository.")
            else:
                prefs = {
                    "occasion": occasion,
                    "adventurousness": adventurousness,
                    "spice_tolerance": spice_tolerance,
                    "budget_pref": budget_pref,
                    "dietary": dietary,
                    "favorite_cuisines": favorite_cuisines,
                    "notes": notes or None,
                }

                with st.spinner("Plating your curated recommendations..."):
                    result = recommend_fallback(prefs)

                st.markdown(f'<p class="concierge-note">{result["concierge_note"]}</p>', unsafe_allow_html=True)
                
                for r in result["recommendations"]:
                    pairing_html = f'<p class="ticket-pairing">Pairs well: {r["pairing"]}</p>' if r.get("pairing") else ""
                    
                    # Collapse all HTML onto a single line to completely bypass Markdown formatting
                    html_content = f'<div class="ticket"><h3>{r["name"]} &nbsp; <span class="ticket-price">AED {r["price_aed"]}</span></h3><p class="ticket-reason">{r["reason"]}</p>{pairing_html}<p class="ticket-source">Curated pick</p></div>'
                    
                    st.markdown(html_content, unsafe_allow_html=True)
