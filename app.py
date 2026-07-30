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

# Brand Palette (Premium Dark Mode Aesthetic)
GOLD = "#D4AF37"
TEAL = "#006666"
COPPER = "#B87333"
PLATINUM = "#E5E4E2"
DARK_BG = "#111111"

# ---------------------------------------------------------
# 2. DATA GENERATION (Synthetic Dubai Fusion Survey Data)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1000
    
    areas = ['Downtown Dubai', 'Dubai Marina', 'Jumeirah', 'DIFC', 'Business Bay']
    cuisines_pool = ['Japanese', 'Peruvian', 'Levantine', 'Italian', 'Indian', 'Mexican']
    
    # Generate Base Demographics
    data = {
        'Respondent_ID': range(1, n + 1),
        'Age': np.random.normal(loc=34, scale=8, size=n).astype(int),
        'Area': np.random.choice(areas, n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
        'Spend (AED)': np.random.normal(loc=250, scale=75, size=n).clip(50, 800),
        'Visit_Freq_Monthly': np.random.poisson(lam=3, size=n).clip(1, 15),
        'Price_Increase15pct_VisitFreq_Change_Pct': np.random.normal(loc=-0.20, scale=0.1, size=n).clip(-1, 0)
    }
    df = pd.DataFrame(data)
    
    # Generate multi-select cuisines
    df['Cuisines_Enjoyed'] = [
        ", ".join(np.random.choice(cuisines_pool, np.random.randint(2, 5), replace=False)) 
        for _ in range(n)
    ]
    
    # Preprocessing & Binning
    df['Age'] = df['Age'].clip(18, 75)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '46+'])
    
    return df

df_raw = load_data()

# ---------------------------------------------------------
# 3. GLOBAL SIDEBAR (Interactivity)
# ---------------------------------------------------------
st.sidebar.header("Global Filters")
st.sidebar.markdown("Use these controls to slice the strategic data.")

min_age, max_age = int(df_raw['Age'].min()), int(df_raw['Age'].max())
selected_age = st.sidebar.slider("Age Range", min_age, max_age, (20, 50))

all_areas = df_raw['Area'].unique().tolist()
selected_areas = st.sidebar.multiselect("Select Area(s)", all_areas, default=all_areas)

# Apply Filters
df = df_raw[
    (df_raw['Age'] >= selected_age[0]) & 
    (df_raw['Age'] <= selected_age[1]) & 
    (df_raw['Area'].isin(selected_areas))
]

if df.empty:
    st.error("No data available for the selected filters.")
    st.stop()

# ---------------------------------------------------------
# 4. TAB ROUTING
# ---------------------------------------------------------
st.title("🍽️ Dubai Fusion Restaurant: Strategic Intelligence")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Demographics & Menu Strategy", 
    "🤖 ML & Persona Mining", 
    "📈 Price Elasticity", 
    "🚨 Pandemic Stress-Test"
])

# =========================================================
# TAB 1: Demographics & Menu Strategy
# =========================================================
with tab1:
    st.subheader("Audience Composition & Preferences")
    
    # Top-Level KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sample Size", f"{len(df):,}")
    col2.metric("Avg Spend (AED)", f"AED {df['Spend (AED)'].mean():.2f}")
    col3.metric("Avg Monthly Visits", f"{df['Visit_Freq_Monthly'].mean():.1f}")
    col4.metric("Avg Age", f"{df['Age'].mean():.1f}")
    
    st.divider()
    
    # Charts Row 1
    c1, c2 = st.columns(2)
    
    with c1:
        fig_age = px.histogram(
            df, x="Age", nbins=15, title="Age Distribution",
            color_discrete_sequence=[TEAL], text_auto=True,
            template="plotly_dark"
        )
        fig_age.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig_age, use_container_width=True)
        
    with c2:
        area_counts = df['Area'].value_counts().reset_index()
        area_counts.columns = ['Area', 'Count']
        fig_area = px.bar(
            area_counts, x="Area", y="Count", title="Geographic Footprint",
            color_discrete_sequence=[GOLD], text_auto=True,
            template="plotly_dark"
        )
        fig_area.update_layout(xaxis_title="Area", yaxis_title="Count")
        st.plotly_chart(fig_area, use_container_width=True)

    # Charts Row 2: Exploded Cuisines
    st.markdown("### Menu Strategy: Top Fusion Pairings")
    cuisines_exploded = df['Cuisines_Enjoyed'].str.split(', ').explode().value_counts().reset_index()
    cuisines_exploded.columns = ['Cuisine', 'Count']
    
    fig_cuisine = px.bar(
        cuisines_exploded, x="Count", y="Cuisine", orientation='h',
        title="Most Requested Culinary Influences",
        color_discrete_sequence=[COPPER], text_auto=True,
        template="plotly_dark"
    )
    fig_cuisine.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Count", yaxis_title="Cuisine")
    st.plotly_chart(fig_cuisine, use_container_width=True)


# =========================================================
# TAB 2: Machine Learning & Persona Mining
# =========================================================
with tab2:
    st.subheader("Algorithmic Persona Generation")
    
    # 1. Association Rules (Apriori)
    st.markdown("### 🛒 Association Rules (Menu & Demographic Synergies)")
    
    ohe_demo = pd.get_dummies(df[['Area', 'Age_Group']])
    ohe_cuisines = df['Cuisines_Enjoyed'].str.get_dummies(sep=', ')
    basket = pd.concat([ohe_demo, ohe_cuisines], axis=1).astype(bool)
    
    frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
        if not rules.empty:
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            rules = rules.sort_values('lift', ascending=False).head(10)
            
            display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            st.dataframe(display_rules.style.format({'support': '{:.2f}', 'confidence': '{:.2f}', 'lift': '{:.2f}'}), use_container_width=True)
        else:
            st.info("No strong association rules found with lift > 1.2.")
    else:
        st.info("Not enough data to calculate support thresholds.")
        
    st.divider()
    
    # 2. K-Means Clustering
    st.markdown("### 🧬 Customer Segmentation (K-Means)")
    
    features = ['Spend (AED)', 'Visit_Freq_Monthly', 'Age']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster_Name'] = df['Cluster'].map({0: 'Value Regulars', 1: 'High-Roller Foodies', 2: 'Infrequent Explorers'})
    
    fig_cluster = px.scatter_3d(
        df, x='Age', y='Spend (AED)', z='Visit_Freq_Monthly',
        color='Cluster_Name', color_discrete_sequence=[GOLD, TEAL, PLATINUM],
        title="3D Behavioral Topography", template="plotly_dark",
        opacity=0.7
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # --- REDESIGNED PERSONA CARDS ---
    st.markdown("#### Persona Synthesis")
    
    # Standard String for CSS 
    persona_css = """
    <style>
        .persona-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 10px; }
        .persona-card { background-color: #111111; padding: 20px; border-radius: 10px; box-shadow: inset 0 0 10px rgba(255,255,255,0.05); }
        .persona-card h4 { margin-top: 0; font-weight: 900; font-size: 1.2em; text-transform: uppercase; }
        .persona-stats { font-size: 0.95em; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #333; color: #E5E4E2; }
        .persona-stats b { color: white; }
        .persona-body { font-size: 0.95em; line-height: 1.5; color: #E5E4E2; }
        .p-teal { border: 2px solid #006666; } .p-teal h4 { color: #006666; } .p-teal strong { color: #006666; }
        .p-gold { border: 2px solid #D4AF37; } .p-gold h4 { color: #D4AF37; } .p-gold strong { color: #D4AF37; }
        .p-plat { border: 2px solid #E5E4E2; } .p-plat h4 { color: #E5E4E2; } .p-plat strong { color: #E5E4E2; }
    </style>
    """
    
    persona_html = '<div class="persona-grid">'
    cluster_means = df.groupby('Cluster_Name')[features].mean()
    
    for c_name, row in cluster_means.iterrows():
        if c_name == 'Value Regulars':
            c_class = "p-teal"
            desc = "Consistent baseline revenue. They visit frequently but manage their check size carefully."
        elif c_name == 'High-Roller Foodies':
            c_class = "p-gold"
            desc = "The profit engines. Older demographic, highly inelastic to price, order premium fusion items."
        else:
            c_class = "p-plat"
            desc = "Curiosity-driven visitors. They come in once a month for the 'experience' but aren't anchored to the brand."
            
        # Formatted string with no blank lines
        persona_html += f"""<div class="persona-card {c_class}"><h4>{c_name}</h4><div class="persona-stats"><div><b>Avg Age:</b> {int(row['Age'])}</div><div><b>Avg Spend:</b> AED {int(row['Spend (AED)'])}</div><div><b>Visits/Mo:</b> {row['Visit_Freq_Monthly']:.1f}</div></div><div class="persona-body"><strong>Strategic Takeaway:</strong><br>{desc}</div></div>"""
        
    persona_html += '</div>'
    
    st.markdown(persona_css + persona_html, unsafe_allow_html=True)

# =========================================================
# TAB 3: Standard Price Elasticity (What-If Analysis)
# =========================================================
with tab3:
    st.subheader("Price Elasticity & Revenue Modeling")
    
    price_increase = st.slider("Proposed Menu Price Increase (%)", min_value=0, max_value=30, value=15, step=1)
    
    # Calculate baseline and new revenue
    total_monthly_visits = df['Visit_Freq_Monthly'].sum()
    base_avg_spend = df['Spend (AED)'].mean()
    base_revenue = total_monthly_visits * base_avg_spend
    
    # Calculate weighted drop-off based on survey data
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
    
    # Waterfall Chart
    fig_waterfall = go.Figure(go.Waterfall(
        name="Revenue Impact", orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Base Revenue", "Price Increase Gain", "Volume Loss (Elasticity)", "Adjusted Revenue"],
        textposition="outside",
        text=[f"AED {base_revenue/1000:.0f}k", f"+AED {(total_monthly_visits * base_avg_spend * (price_increase/100))/1000:.0f}k", 
              f"-AED {abs(new_revenue - (base_revenue * (1+(price_increase/100))))/1000:.0f}k", f"AED {new_revenue/1000:.0f}k"],
        y=[base_revenue, 
           total_monthly_visits * base_avg_spend * (price_increase/100), 
           new_revenue - (base_revenue * (1+(price_increase/100))), 
           new_revenue],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": GOLD}},
        decreasing={"marker": {"color": COPPER}},
        totals={"marker": {"color": TEAL}}
    ))
    
    fig_waterfall.update_layout(title="Net Revenue Waterfall Analysis", template="plotly_dark")
    st.plotly_chart(fig_waterfall, use_container_width=True)


# =========================================================
# TAB 4: Pandemic Stress-Test (6-Month Crisis Model)
# =========================================================
with tab4:
    st.subheader("Financial Contagion & Survival Stress-Test")
    st.markdown("Model a severe 6-month operational disruption. Adjust the parameters below to determine survivability.")
    
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
    cash_runway = CASH_RESERVES / monthly_burn if monthly_burn > 0 else float('inf')
    
    delivery_margin_per_order = BASE_AOV * (1 - cogs_pct - (del_fee / 100.0))
    bep_delivery_orders = FIXED_COSTS / delivery_margin_per_order if delivery_margin_per_order > 0 else float('inf')
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly Burn Rate", f"AED {monthly_burn:,.0f}" if monthly_burn > 0 else "Profitable")
    c2.metric("Cash Runway", f"{cash_runway:.1f} Months" if monthly_burn > 0 else "Infinite")
    c3.metric("Delivery Break-Even", f"{bep_delivery_orders:,.0f} Orders/Mo" if bep_delivery_orders != float('inf') else "Unachievable")
    
    st.markdown("### Monthly Crisis Cash Flow")
    fig_cf = go.Figure(go.Waterfall(
        name="Cash Flow", orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["New Dine-In Rev", "Net Delivery Rev", "Fixed Costs", "Variable Costs (COGS)", "Net Monthly Cash Flow"],
        text=[f"AED {new_dine_in_rev/1000:.0f}k", f"AED {net_delivery_rev/1000:.0f}k", 
              f"-AED {FIXED_COSTS/1000:.0f}k", f"-AED {variable_costs/1000:.0f}k", f"AED {-monthly_burn/1000:.0f}k"],
        y=[new_dine_in_rev, net_delivery_rev, -FIXED_COSTS, -variable_costs, -monthly_burn],
        textposition="outside",
        connector={"line": {"color": PLATINUM}},
        increasing={"marker": {"color": TEAL}},
        decreasing={"marker": {"color": COPPER}},
        totals={"marker": {"color": GOLD if monthly_burn <= 0 else "#8B0000"}}
    ))
    fig_cf.update_layout(template="plotly_dark")
    st.plotly_chart(fig_cf, use_container_width=True)
    
    st.markdown("### 6-Month Liquidity Depletion Curve")
    months = [f"Month {i}" for i in range(7)]
    runway_data = [CASH_RESERVES - (max(monthly_burn, 0) * i) for i in range(7)]
    
    fig_runway = px.line(
        x=months, y=runway_data, markers=True, text=[f"{val/1000:.0f}k" for val in runway_data],
        title="Cash Reserves Trajectory", template="plotly_dark"
    )
    fig_runway.update_traces(line_color=COPPER, textposition="top center")
    fig_runway.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Bankruptcy Line")
    fig_runway.update_layout(xaxis_title="Timeline", yaxis_title="Cash Balance (AED)")
    st.plotly_chart(fig_runway, use_container_width=True)

    # --- REDESIGNED CONTINGENCY CARDS ---
    st.divider()
    st.markdown("### Strategic Contingency Imperatives 🔗")

    crisis_css = """
    <style>
        .crisis-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 10px; }
        .crisis-card { background-color: #111111; padding: 20px; border-radius: 10px; box-shadow: inset 0 0 10px rgba(255,255,255,0.05); }
        .crisis-card h4 { margin-top: 0; font-weight: 900; font-size: 1.1em; display: flex; align-items: center; gap: 10px; }
        .crisis-subtext { font-style: italic; color: #E5E4E2; font-size: 0.9em; margin-top: -10px; margin-bottom: 15px; }
        .crisis-body { color: #E5E4E2; line-height: 1.5; font-size: 0.95em; }
        .crisis-body b { color: white; }
        .card-1 { border: 2px solid #D4AF37; color: #D4AF37; } .card-1 h4 { color: #D4AF37; } .card-1 strong { color: #D4AF37; font-size: 1.05em; }
        .card-2 { border: 2px solid #B87333; color: #B87333; } .card-2 h4 { color: #B87333; } .card-2 strong { color: #B87333; font-size: 1.05em; }
        .card-3 { border: 2px solid #006666; color: #006666; } .card-3 h4 { color: #006666; } .card-3 strong { color: #006666; font-size: 1.05em; }
    </style>
    """
    
    # F-string formatted with zero blank lines between elements
    crisis_html = f"""<div class="crisis-grid"><div class="crisis-card card-1"><h4>1. DARK KITCHEN ARCHITECTURE</h4><p class="crisis-subtext">Condition: Dine-In restricted < 30%</p><div class="crisis-body"><p><b>Risk:</b> Fixed real-estate costs become a lethal liability.</p><p><b>Action:</b> Negotiate a clause to sub-lease kitchen space to a secondary virtual brand to offset the <strong>{FIXED_COSTS/1000:,.0f}k AED fixed costs</strong>.</p></div></div><div class="crisis-card card-2"><h4>2. MARGIN SHIELDING</h4><p class="crisis-subtext">Target: {del_fee}% Commission Mitigation</p><div class="crisis-body"><p><b>Challenge:</b> Current model requires <strong>{bep_delivery_orders:,.0f} delivery orders</strong> just to break even.</p><p><b>Action:</b> Engineer a 'Delivery-Only' menu subset using lower-cost ingredients (buffering the <strong>{supply_mult}x supply multiplier</strong>) to protect the base <strong>{BASE_AOV:,.0f} AED avg order value</strong>.</p></div></div><div class="crisis-card card-3"><h4>3. LIQUIDITY PRESERVATION</h4><p class="crisis-subtext">Status: Capital Protection</p><div class="crisis-body"><p><b>Warning:</b> COGS spikes and platform fees cause rapid cash hemorrhage.</p><p><b>Action:</b> Secure a rolling credit facility equalling <strong>3 months of OPEX</strong> before launch, to avoid equity dilution post-crisis.</p></div></div></div>"""
    
    st.markdown(crisis_css + crisis_html, unsafe_allow_html=True)
