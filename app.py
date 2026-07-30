import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. SETUP & THEME CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Dubai Fusion Strategy & Launch Model", layout="wide")

GOLD = "#D4AF37"
TEAL = "#006666"
COPPER = "#B87333"
PLATINUM = "#E5E4E2"
DARK_BG = "#111111"

DISTINCT_COLORS = [GOLD, TEAL, COPPER, "#8B0000", PLATINUM, "#5F9EA0", "#D2691E", "#4682B4"]

# ---------------------------------------------------------
# 2. DATA GENERATION (Simulating your 500 Survey Responses)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500  # Survey Respondents
    
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
    
    # Generate multi-select cuisines (2 to 3 per person to find blends)
    df['Cuisines_Enjoyed'] = [", ".join(np.random.choice(cuisines_pool, np.random.randint(2, 4), replace=False)) for _ in range(n)]
    df['Age'] = df['Age'].clip(18, 70)
    
    return df

df_raw = load_data()

# ---------------------------------------------------------
# 3. GLOBAL SIDEBAR
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
# 4. TAB ROUTING
# ---------------------------------------------------------
st.title("🍽️ Dubai Fusion: Pre-Launch Survey Analytics")

tab_bp, tab_demo, tab_menu, tab_ops, tab_fin = st.tabs([
    "📑 Executive Blueprint",
    "📊 Demographics & Location", 
    "🥢 Cuisine & Menu Engineering", 
    "🛋️ Atmosphere & Operations",
    "📈 Financial Projections"
])

# =========================================================
# TAB 1: Executive Blueprint
# =========================================================
with tab_bp:
    st.subheader("Data-Driven Launch Strategy (Based on 500 Responses)")
    
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Survey Validations", f"{len(df)}", "Respondents")
    f2.metric("Market AOV (Spend)", f"AED {df['Spend_Capacity (AED)'].mean():.0f}", "Avg Capacity")
    f3.metric("Peak Demand Window", df['Preferred_Timing'].mode()[0], "Highest Volume")
    f4.metric("Top Location Request", df['Area_Preference'].mode()[0], "Prime Real Estate")
    
    bp_html = f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:20px; margin-top:25px;">
        <div style="background:#111; border-left:4px solid {GOLD}; padding:20px; border-radius:8px;">
            <h4 style="color:{GOLD}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">📍 Location Strategy</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">Survey data heavily indexes toward corporate/premium hubs. Targeting real estate in <b>{df['Area_Preference'].mode()[0]}</b> captures the highest concentration of high-spend individuals. A secondary outpost in DIFC caters to the corporate lunch crowd.</p>
        </div>
        <div style="background:#111; border-left:4px solid {TEAL}; padding:20px; border-radius:8px;">
            <h4 style="color:{TEAL}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🛋️ Interior & Atmosphere</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">The audience demands a dual-experience environment. The layout must prioritize a <b>{df['Seating_Vibe'].mode()[0]}</b> for the evening rush, transitioning smoothly from an intimate dining setting into a high-energy lounge to maximize late-night beverage sales.</p>
        </div>
        <div style="background:#111; border-left:4px solid {COPPER}; padding:20px; border-radius:8px;">
            <h4 style="color:{COPPER}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🥢 Menu & Blending</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">The data confirms a strong appetite for bold combinations. Menu engineering should focus heavily on the top intersecting cuisine pairs discovered in the survey, allowing for a streamlined supply chain by cross-utilizing premium ingredients.</p>
        </div>
        <div style="background:#111; border-left:4px solid {PLATINUM}; padding:20px; border-radius:8px;">
            <h4 style="color:{PLATINUM}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">⏱️ Operational Efficiency</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">Peak staffing and inventory must align with the <b>{df['Preferred_Timing'].mode()[0]}</b> window. Table turnover rates must be tightly managed during this period, whereas lunch operations can operate with a leaner team utilizing automation.</p>
        </div>
    </div>
    """
    st.markdown(bp_html, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🚀 Real-World Execution Timeline")
    
    timeline_html = f"""
    <div style="border-left:3px solid {TEAL}; padding-left:20px; margin-top:10px;">
        <div style="margin-bottom:20px; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Days 1 - 30: Location & Culinary R&D</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Action:</b> Leverage survey data to negotiate lease terms in the top-voted neighborhood. Finalize R&D for the top 3 fusion blends. Secure supply chain contracts for cross-utilized ingredients.</div>
        </div>
        <div style="margin-bottom:20px; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Days 31 - 60: Interior Zoning & Staffing</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Action:</b> Zone the restaurant seating to match the survey's vibe preferences (e.g., 60% High-Energy Lounge, 40% Intimate Dining). Hire front-of-house staff optimized for the identified peak operational hours.</div>
        </div>
        <div style="margin-bottom:0; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Days 61 - 90: Soft Launch Targeting</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Action:</b> Send exclusive soft-launch invitations to the 500 original survey respondents. Use their initial feedback to calibrate menu pricing and service flow before the official grand opening.</div>
        </div>
    </div>
    """
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
            # Calculate top requested pairs for Fusion
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
    
    # Calculate baseline revenue directly from survey spend capacity
    total_expected_diners = len(df) * 4 # Assuming respondents visit 4 times a month on average for model scaling
    base_avg_spend = df['Spend_Capacity (AED)'].mean()
    base_revenue = total_expected_diners * base_avg_spend
    
    st.markdown(f"**Baseline Monthly Revenue Model:** Based on targeting {total_expected_diners:,} monthly covers at the survey's average capacity of AED {base_avg_spend:.0f}.")
    
    price_increase = st.slider("Test Menu Price Increase Premium (%)", min_value=0, max_value=30, value=15, step=1)
    
    # Simple elasticity: higher price drops volume
    scaled_drop_off = (price_increase / 100.0) * -0.65 # Assume 0.65 elasticity factor
    
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
