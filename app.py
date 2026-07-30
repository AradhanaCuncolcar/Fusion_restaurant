import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. SETUP & THEME CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Dubai Fusion Strategy & Crisis Model", layout="wide")

GOLD = "#D4AF37"
TEAL = "#006666"
COPPER = "#B87333"
PLATINUM = "#E5E4E2"
DARK_BG = "#111111"

# ---------------------------------------------------------
# 2. DATA GENERATION
# ---------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1000
    areas = ['Downtown Dubai', 'Dubai Marina', 'Jumeirah', 'DIFC', 'Business Bay']
    cuisines_pool = ['Japanese', 'Peruvian', 'Levantine', 'Italian', 'Indian', 'Mexican']
    
    data = {
        'Respondent_ID': range(1, n + 1),
        'Age': np.random.normal(loc=34, scale=8, size=n).astype(int),
        'Area': np.random.choice(areas, n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
        'Spend (AED)': np.random.normal(loc=250, scale=75, size=n).clip(50, 800),
        'Visit_Freq_Monthly': np.random.poisson(lam=3, size=n).clip(1, 15),
        'Price_Increase15pct_VisitFreq_Change_Pct': np.random.normal(loc=-0.20, scale=0.1, size=n).clip(-1, 0)
    }
    df = pd.DataFrame(data)
    df['Cuisines_Enjoyed'] = [", ".join(np.random.choice(cuisines_pool, np.random.randint(2, 5), replace=False)) for _ in range(n)]
    df['Age'] = df['Age'].clip(18, 75)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '46+'])
    return df

df_raw = load_data()

# ---------------------------------------------------------
# 3. GLOBAL SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Global Filters")
st.sidebar.markdown("Use these controls to slice the strategic data.")
min_age, max_age = int(df_raw['Age'].min()), int(df_raw['Age'].max())
selected_age = st.sidebar.slider("Age Range", min_age, max_age, (20, 50))
all_areas = df_raw['Area'].unique().tolist()
selected_areas = st.sidebar.multiselect("Select Area(s)", all_areas, default=all_areas)

df = df_raw[(df_raw['Age'] >= selected_age[0]) & (df_raw['Age'] <= selected_age[1]) & (df_raw['Area'].isin(selected_areas))]
if df.empty:
    st.error("No data available for the selected filters.")
    st.stop()

# ---------------------------------------------------------
# 4. TAB ROUTING
# ---------------------------------------------------------
st.title("🍽️ Dubai Fusion Restaurant: Strategic Intelligence")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Demographics & Menu", 
    "🤖 ML & Personas", 
    "📈 Price Elasticity", 
    "🚨 Crisis Model",
    "📑 Business Plan"
])

# =========================================================
# TAB 1: Demographics & Menu Strategy
# =========================================================
with tab1:
    st.subheader("Audience Composition & Preferences")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sample Size", f"{len(df):,}")
    col2.metric("Avg Spend (AED)", f"AED {df['Spend (AED)'].mean():.2f}")
    col3.metric("Avg Monthly Visits", f"{df['Visit_Freq_Monthly'].mean():.1f}")
    col4.metric("Avg Age", f"{df['Age'].mean():.1f}")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        # Beautifully formatted Histogram in a bubble box
        with st.container(border=True):
            fig_age = px.histogram(df, x="Age", nbins=15, title="Age Distribution", color_discrete_sequence=[TEAL], text_auto=True, template="plotly_dark")
            fig_age.update_traces(marker_line_width=1.5, marker_line_color=PLATINUM, textposition="outside", cliponaxis=False)
            fig_age.update_layout(bargap=0.1, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_age, use_container_width=True)
            
    with c2:
        # Beautifully formatted Bar Chart in a bubble box
        with st.container(border=True):
            area_counts = df['Area'].value_counts().reset_index()
            area_counts.columns = ['Area', 'Count']
            fig_area = px.bar(area_counts, x="Area", y="Count", title="Geographic Footprint", color_discrete_sequence=[GOLD], text_auto=True, template="plotly_dark")
            fig_area.update_traces(textposition="outside", cliponaxis=False)
            fig_area.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_area, use_container_width=True)

    # Horizontal Bar Chart in a bubble box
    with st.container(border=True):
        cuisines_exploded = df['Cuisines_Enjoyed'].str.split(', ').explode().value_counts().reset_index()
        cuisines_exploded.columns = ['Cuisine', 'Count']
        fig_cuisine = px.bar(cuisines_exploded, x="Count", y="Cuisine", orientation='h', title="Most Requested Culinary Influences", color_discrete_sequence=[COPPER], text_auto=True, template="plotly_dark")
        fig_cuisine.update_traces(textposition="outside", cliponaxis=False)
        fig_cuisine.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_cuisine, use_container_width=True)

# =========================================================
# TAB 2: Machine Learning & Persona Mining
# =========================================================
with tab2:
    st.subheader("Algorithmic Persona Generation")
    features = ['Spend (AED)', 'Visit_Freq_Monthly', 'Age']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster_Name'] = df['Cluster'].map({0: 'Value Regulars', 1: 'High-Roller Foodies', 2: 'Infrequent Explorers'})
    
    # 3D Scatter in a bubble box
    with st.container(border=True):
        fig_cluster = px.scatter_3d(df, x='Age', y='Spend (AED)', z='Visit_Freq_Monthly', color='Cluster_Name', color_discrete_sequence=[GOLD, TEAL, PLATINUM], title="3D Behavioral Topography", template="plotly_dark", opacity=0.7)
        fig_cluster.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.markdown("#### Persona Synthesis")
    cluster_means = df.groupby('Cluster_Name')[features].mean()
    
    p_html = f"""<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">"""
    for c_name, row in cluster_means.iterrows():
        if c_name == 'Value Regulars':
            color, icon, desc = TEAL, "👥", "Consistent baseline revenue. They visit frequently but manage their check size carefully."
        elif c_name == 'High-Roller Foodies':
            color, icon, desc = GOLD, "💎", "The profit engines. Older demographic, highly inelastic to price, order premium fusion items."
        else:
            color, icon, desc = PLATINUM, "🧭", "Curiosity-driven visitors. They come in once a month for the 'experience'."
        
        p_html += f"""<div style="background:#111; border:2px solid {color}; border-radius:10px; padding:20px;">"""
        p_html += f"""<div style="display:flex; align-items:center; gap:10px; margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:10px;"><div style="font-size:30px;">{icon}</div><h4 style="color:{color}; margin:0; font-weight:900; text-transform:uppercase;">{c_name}</h4></div>"""
        p_html += f"""<div style="color:#ccc; font-size:0.95em; line-height:1.5;"><p><b>Avg Age:</b> {int(row['Age'])}</p><p><b>Avg Spend:</b> AED {int(row['Spend (AED)'])}</p><p><b>Visits/Mo:</b> {row['Visit_Freq_Monthly']:.1f}</p><p><strong style="color:{color};">Takeaway:</strong> {desc}</p></div></div>"""
    p_html += "</div>"
    st.markdown(p_html, unsafe_allow_html=True)

# =========================================================
# TAB 3: Standard Price Elasticity
# =========================================================
with tab3:
    st.subheader("Price Elasticity & Revenue Modeling")
    price_increase = st.slider("Proposed Menu Price Increase (%)", min_value=0, max_value=30, value=15, step=1)
    total_monthly_visits = df['Visit_Freq_Monthly'].sum()
    base_avg_spend = df['Spend (AED)'].mean()
    base_revenue = total_monthly_visits * base_avg_spend
    avg_drop_off_15pct = df['Price_Increase15pct_VisitFreq_Change_Pct'].mean() 
    scaled_drop_off = (price_increase / 15.0) * avg_drop_off_15pct 
    new_visits = total_monthly_visits * (1 + scaled_drop_off)
    new_spend = base_avg_spend * (1 + (price_increase/100))
    new_revenue = new_visits * new_spend
    revenue_delta = new_revenue - base_revenue

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Baseline Monthly Revenue", f"AED {base_revenue:,.0f}")
    col_b.metric("Projected Monthly Revenue", f"AED {new_revenue:,.0f}", f"{revenue_delta:,.0f} AED")
    col_c.metric("Projected Visit Drop-off", f"{abs(scaled_drop_off)*100:.1f}%", delta_color="inverse")
    
    # Waterfall Chart in a bubble box
    with st.container(border=True):
        fig_waterfall = go.Figure(go.Waterfall(name="Revenue Impact", orientation="v", measure=["absolute", "relative", "relative", "total"], x=["Base Revenue", "Price Increase Gain", "Volume Loss", "Adjusted Revenue"], y=[base_revenue, total_monthly_visits * base_avg_spend * (price_increase/100), new_revenue - (base_revenue * (1+(price_increase/100))), new_revenue], connector={"line": {"color": "rgb(63, 63, 63)"}}, increasing={"marker": {"color": GOLD}}, decreasing={"marker": {"color": COPPER}}, totals={"marker": {"color": TEAL}}))
        fig_waterfall.update_layout(title="Net Revenue Waterfall Analysis", template="plotly_dark", margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_waterfall, use_container_width=True)

# =========================================================
# TAB 4: Pandemic Stress-Test (6-Month Crisis Model)
# =========================================================
with tab4:
    st.subheader("Financial Contagion & Survival Stress-Test")
    st.markdown("Model a severe 6-month operational disruption.")
    
    CASH_RESERVES = 750000 
    FIXED_COSTS = 120000 
    BASE_DINE_IN_REV = 250000
    BASE_DELIVERY_REV = 80000
    BASE_AOV = 150 
    
    st.sidebar.markdown("---")
    st.sidebar.header("Crisis Variables")
    cap_limit = st.sidebar.slider("Dine-In Capacity Limit (%)", 0, 100, 30, step=10)
    del_surge = st.sidebar.slider("Delivery Volume Surge (%)", 0, 200, 50, step=10)
    supply_mult = st.sidebar.slider("Supply Chain Cost Multiplier", 1.0, 2.0, 1.2, step=0.1)
    del_fee = st.sidebar.slider("3rd-Party Delivery Fee (%)", 10, 40, 30, step=5)
    
    new_dine_in_rev = BASE_DINE_IN_REV * (cap_limit / 100.0)
    gross_delivery_rev = BASE_DELIVERY_REV * (1 + (del_surge / 100.0))
    net_delivery_rev = gross_delivery_rev * (1 - (del_fee / 100.0))
    total_rev = new_dine_in_rev + net_delivery_rev
    
    cogs_pct = 0.30 * supply_mult
    variable_costs = (new_dine_in_rev + gross_delivery_rev) * cogs_pct 
    monthly_burn = (FIXED_COSTS + variable_costs) - total_rev
    
    delivery_margin_per_order = BASE_AOV * (1 - cogs_pct - (del_fee / 100.0))
    bep_delivery_orders = FIXED_COSTS / delivery_margin_per_order if delivery_margin_per_order > 0 else float('inf')
    
    # Crisis Waterfall Chart in a bubble box
    with st.container(border=True):
        fig_cf = go.Figure(go.Waterfall(name="Cash Flow", orientation="v", measure=["relative", "relative", "relative", "relative", "total"], x=["New Dine-In Rev", "Net Delivery Rev", "Fixed Costs", "Variable Costs (COGS)", "Net Monthly Cash Flow"], y=[new_dine_in_rev, net_delivery_rev, -FIXED_COSTS, -variable_costs, -monthly_burn], connector={"line": {"color": PLATINUM}}, increasing={"marker": {"color": TEAL}}, decreasing={"marker": {"color": COPPER}}, totals={"marker": {"color": GOLD if monthly_burn <= 0 else "#8B0000"}}))
        fig_cf.update_layout(title="Monthly Crisis Cash Flow", template="plotly_dark", margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_cf, use_container_width=True)

    st.markdown("### STRATEGIC CONTINGENCY IMPERATIVES 🔗")
    
    crisis_html = f"""
    <div style="display:flex; flex-wrap:wrap; gap:20px; margin-top:15px; font-family:sans-serif;">
        <!-- Card 1 -->
        <div style="flex:1; min-width:300px; background:#111; border:2px solid {GOLD}; border-radius:10px; padding:20px;">
            <div style="display:flex; align-items:center; gap:15px; margin-bottom:10px;">
                <div style="font-size:45px;">🥡</div>
                <h4 style="color:{GOLD}; margin:0; font-weight:900; line-height:1.2; font-size:1.1em;">1. DARK KITCHEN<br>ARCHITECTURE</h4>
            </div>
            <p style="font-style:italic; color:#ccc; font-size:0.9em; margin-top:0;">Condition: Dine-In restricted &lt; 30%</p>
            <p style="color:#eee; font-size:0.95em; line-height:1.5;"><b style="color:white;">Risk:</b> Fixed real-estate costs become a lethal liability.</p>
            <p style="color:#eee; font-size:0.95em; line-height:1.5;"><b style="color:white;">Action:</b> Negotiate a clause to sub-lease kitchen space to a secondary virtual brand to offset the <strong style="color:{GOLD};">{FIXED_COSTS/1000:,.0f}k AED fixed costs</strong>.</p>
        </div>
        <!-- Card 2 -->
        <div style="flex:1; min-width:300px; background:#111; border:2px solid {COPPER}; border-radius:10px; padding:20px;">
            <div style="display:flex; align-items:center; gap:15px; margin-bottom:10px;">
                <div style="font-size:45px;">🛵</div>
                <h4 style="color:{COPPER}; margin:0; font-weight:900; line-height:1.2; font-size:1.1em;">2. MARGIN SHIELDING</h4>
            </div>
            <p style="font-style:italic; color:#ccc; font-size:0.9em; margin-top:0;">Target: {del_fee}% Commission Mitigation</p>
            <p style="color:#eee; font-size:0.95em; line-height:1.5;"><b style="color:white;">Challenge:</b> The current model requires <strong style="color:{COPPER};">{bep_delivery_orders:,.0f} delivery orders</strong> just to break even.</p>
            <p style="color:#eee; font-size:0.95em; line-height:1.5;"><b style="color:white;">Action:</b> Engineer a 'Delivery-Only' menu subset using lower-cost ingredients (buffering the <strong style="color:{COPPER};">{supply_mult}x supply multiplier</strong>) to protect the base <strong style="color:{COPPER};">{BASE_AOV:,.0f} AED average order value</strong>.</p>
        </div>
        <!-- Card 3 -->
        <div style="flex:1; min-width:300px; background:#111; border:2px solid {TEAL}; border-radius:10px; padding:20px;">
            <div style="display:flex; align-items:center; gap:15px; margin-bottom:10px;">
                <div style="font-size:45px;">🏦</div>
                <h4 style="color:{TEAL}; margin:0; font-weight:900; line-height:1.2; font-size:1.1em;">3. LIQUIDITY<br>PRESERVATION</h4>
            </div>
            <p style="font-style:italic; color:#ccc; font-size:0.9em; margin-top:0;">Status: Capital Protection</p>
            <p style="color:#eee; font-size:0.95em; line-height:1.5;"><b style="color:white;">Warning:</b> Rapid cash hemorrhage occurs if COGS spikes simultaneously with platform fees.</p>
            <p style="color:#eee; font-size:0.95em; line-height:1.5;"><b style="color:white;">Action:</b> Secure a rolling credit facility equalling <strong style="color:{TEAL};">3 months of OPEX</strong> prior to launch, rather than scrambling for equity dilution post-crisis.</p>
        </div>
    </div>
    """
    st.markdown(crisis_html, unsafe_allow_html=True)

# =========================================================
# TAB 5: Strategic Business Plan
# =========================================================
with tab5:
    st.subheader("Enterprise Blueprint & Investor Pitch")
    
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Target Year 1 Revenue", "AED 4.2M", "Base + Delivery")
    f2.metric("Target Gross Margin", "70%", "Optimized COGS")
    f3.metric("Required OPEX Runway", "3 Months", "Crisis Buffer")
    f4.metric("Breakeven Timeline", "Month 8", "Post-Launch")
    
    bp_html = f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:20px; margin-top:25px;">
        <div style="background:#111; border-left:4px solid {GOLD}; padding:20px; border-radius:8px;">
            <h4 style="color:{GOLD}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">🏢 1. Executive Summary</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">Operating at the intersection of hospitality and corporate scaling, this venture utilizes advanced AI and predictive analytics to disrupt Dubai's fusion culinary space. Positioned for strategic business leadership and rapid expansion, our model guarantees premium dining experiences backed by optimized unit economics.</p>
        </div>
        <div style="background:#111; border-left:4px solid {TEAL}; padding:20px; border-radius:8px;">
            <h4 style="color:{TEAL}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">📊 2. Market Analysis</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;">Targeting the lucrative intersection of <b>High-Roller Foodies</b> (high LTV, inelastic) and <b>Value Regulars</b>. Located strategically in high-density corporate and residential hubs (Downtown Dubai, Dubai Marina), capitalizing on the demand for hybrid experiential dining.</p>
        </div>
        <div style="background:#111; border-left:4px solid {COPPER}; padding:20px; border-radius:8px;">
            <h4 style="color:{COPPER}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">💰 3. Pricing Strategy</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;"><b>Dynamic & Value-Based:</b> Leveraging price elasticity algorithms to maintain a 15% premium on anchor dishes while offering high-margin entry items. Algorithmic adjustments track supply chain multipliers to fiercely protect the base 150 AED Average Order Value.</p>
        </div>
        <div style="background:#111; border-left:4px solid {PLATINUM}; padding:20px; border-radius:8px;">
            <h4 style="color:{PLATINUM}; margin-top:0; font-weight:900; border-bottom:1px solid #333; padding-bottom:10px;">⚙️ 4. Operational Plan</h4>
            <p style="color:#eee; font-size:0.95em; line-height:1.6;"><b>Tech-First Infrastructure:</b> Operations rely on intelligent process automation bots to handle 3rd-party delivery EDI and streamline client support workflows. Minimizes human error in routing and enables seamless integration with secondary virtual dark-kitchen brands.</p>
        </div>
    </div>
    """
    st.markdown(bp_html, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🚀 90-Day Execution Plan")
    
    timeline_html = f"""
    <div style="border-left:3px solid {TEAL}; padding-left:20px; margin-top:10px;">
        <div style="margin-bottom:20px; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Days 1 - 30: Prototyping & Capital Allocation</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Objectives:</b> Finalize the interactive dashboard prototype and synthesize the investor pitch. Secure the rolling credit facility for 3 months of OPEX. Initiate lease negotiations with dark-kitchen scalability clauses.</div>
        </div>
        <div style="margin-bottom:20px; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Days 31 - 60: Tech Infrastructure & Automation</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Objectives:</b> Deploy process automation bots for inventory management and EDI integration with third-party delivery platforms. Conduct localized marketing tests targeting the core persona clusters.</div>
        </div>
        <div style="margin-bottom:0; position:relative;">
            <div style="position:absolute; left:-27px; top:5px; width:11px; height:11px; background:{TEAL}; border-radius:50%; border:2px solid #111;"></div>
            <h5 style="color:{TEAL}; margin:0 0 5px 0; font-weight:bold; font-size:1.1em;">Days 61 - 90: Soft Launch & Model Calibration</h5>
            <div style="background:#111; padding:15px; border-radius:8px; border:1px solid #333; color:#eee; font-size:0.95em;"><b>Objectives:</b> Execute localized soft launch. Activate Apriori-based upselling logic on digital menus. Calibrate price elasticity algorithms based on real-world customer acquisition costs (CAC) and conversion data prior to hard launch.</div>
        </div>
    </div>
    """
    st.markdown(timeline_html, unsafe_allow_html=True)
