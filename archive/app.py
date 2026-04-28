import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ============================================================
# RETAIL GROWTH ADVISOR
# Clean UI version: simple for users, strong for data mining demo
# No API. Uses only local CSV files.
# ============================================================

st.set_page_config(
    page_title="Retail Growth Advisor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- MODERN UI STYLE ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {
    color: #0f172a !important;
}
.hero {
    background: radial-gradient(circle at top left, #38bdf8 0%, transparent 28%),
                linear-gradient(135deg, #020617 0%, #0f172a 45%, #1e293b 100%);
    color: white;
    padding: 34px 38px;
    border-radius: 28px;
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.28);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.12);
}
.hero h1 {
    font-size: 46px;
    line-height: 1.05;
    margin: 0 0 10px 0;
    letter-spacing: -1px;
}
.hero p {
    font-size: 18px;
    max-width: 900px;
    color: #cbd5e1;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(14,165,233,0.16);
    color: #bae6fd;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 13px;
    margin-bottom: 14px;
    border: 1px solid rgba(186,230,253,0.25);
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 19px 20px;
    box-shadow: 0 9px 24px rgba(15, 23, 42, 0.08);
    height: 118px;
}
.kpi-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.kpi-number {
    color: #020617;
    font-size: 31px;
    font-weight: 900;
    margin-top: 6px;
}
.kpi-note {
    color: #64748b;
    font-size: 13px;
    margin-top: 2px;
}
.panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
    margin-bottom: 16px;
}
.card-title {
    color: #020617;
    font-size: 20px;
    font-weight: 850;
    margin-bottom: 7px;
}
.card-body {
    color: #475569;
    font-size: 15px;
    line-height: 1.45;
}
.green {border-left: 8px solid #16a34a;}
.orange {border-left: 8px solid #f59e0b;}
.blue {border-left: 8px solid #2563eb;}
.red {border-left: 8px solid #dc2626;}
.purple {border-left: 8px solid #7c3aed;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 9px 16px;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    background: #dbeafe !important;
    color: #0f172a !important;
    border-color: #93c5fd !important;
}
.small-muted {
    color: #64748b;
    font-size: 13px;
}
hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FILE CHECK ----------------
required_files = ["order_products__train.csv", "products.csv"]
missing_files = [f for f in required_files if not Path(f).exists()]

if missing_files:
    st.error("Missing required file(s): " + ", ".join(missing_files))
    st.info("Put app.py in the same folder as order_products__train.csv and products.csv.")
    st.stop()

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.title("🛠 Control Panel")
st.sidebar.caption("Use Fast Demo during presentation. Use Balanced for stronger screenshots.")

speed_mode = st.sidebar.selectbox(
    "Run mode",
    ["Fast demo", "Balanced", "Deeper analysis"],
    index=0
)

mode_defaults = {
    "Fast demo": {"rows": 60000, "top": 90, "support": 0.005, "confidence": 0.05},
    "Balanced": {"rows": 120000, "top": 130, "support": 0.003, "confidence": 0.05},
    "Deeper analysis": {"rows": 220000, "top": 180, "support": 0.002, "confidence": 0.05},
}

st.sidebar.subheader("Data")
row_limit = st.sidebar.slider(
    "Rows to read",
    min_value=20000,
    max_value=300000,
    value=mode_defaults[speed_mode]["rows"],
    step=20000
)
use_prior = st.sidebar.checkbox("Use prior-order file if available", value=False)
use_order_time = st.sidebar.checkbox("Use order time features", value=True)

st.sidebar.subheader("Basket Mining")
top_n_products = st.sidebar.slider(
    "Products used in basket mining",
    min_value=40,
    max_value=250,
    value=mode_defaults[speed_mode]["top"],
    step=10
)
algorithm = st.sidebar.selectbox("Mining algorithm", ["FP-Growth", "Apriori"], index=0)
min_support = st.sidebar.select_slider(
    "Support",
    options=[0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02],
    value=mode_defaults[speed_mode]["support"]
)
min_confidence = st.sidebar.select_slider(
    "Confidence",
    options=[0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    value=mode_defaults[speed_mode]["confidence"]
)
min_lift = st.sidebar.select_slider("Lift", options=[1.0, 1.1, 1.2, 1.5, 2.0], value=1.1)
max_len = st.sidebar.selectbox("Max items in a pattern", [2, 3], index=1)

st.sidebar.subheader("Profit Simulator")
avg_price = st.sidebar.slider("Average item price ($)", 1.0, 25.0, 5.0, 0.5)
profit_margin = st.sidebar.slider("Profit margin", 0.10, 0.70, 0.30, 0.05)

# ---------------- HELPER FUNCTIONS ----------------
def money(x):
    return f"${x:,.0f}"


def pct(x):
    return f"{x * 100:.1f}%"


def card(title, body, border="blue"):
    st.markdown(
        f"""
        <div class="panel {border}">
            <div class="card-title">{title}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-number">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data(show_spinner=False)
def prepare_base_data(row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products):
    if use_prior and Path("order_products__prior.csv").exists():
        train_rows = int(row_limit * 0.45)
        prior_rows = row_limit - train_rows
        train = pd.read_csv("order_products__train.csv", nrows=train_rows)
        prior = pd.read_csv("order_products__prior.csv", nrows=prior_rows)
        order_products = pd.concat([train, prior], ignore_index=True)
    else:
        order_products = pd.read_csv("order_products__train.csv", nrows=row_limit)

    raw_rows = len(order_products)
    products = pd.read_csv("products.csv")
    df = order_products.merge(products, on="product_id", how="left")

    if Path("aisles.csv").exists():
        aisles = pd.read_csv("aisles.csv")
        df = df.merge(aisles, on="aisle_id", how="left")

    if Path("departments.csv").exists():
        departments = pd.read_csv("departments.csv")
        df = df.merge(departments, on="department_id", how="left")

    if use_order_time and Path("orders.csv").exists():
        order_cols = pd.read_csv("orders.csv", nrows=1).columns.tolist()
        needed = [c for c in ["order_id", "order_dow", "order_hour_of_day", "days_since_prior_order"] if c in order_cols]
        if len(needed) > 1:
            orders = pd.read_csv("orders.csv", usecols=needed)
            orders = orders[orders["order_id"].isin(df["order_id"].unique())]
            df = df.merge(orders, on="order_id", how="left")

    before_rows = len(df)
    duplicate_rows = int(df.duplicated(subset=["order_id", "product_id"]).sum())

    df = df.dropna(subset=["order_id", "product_id", "product_name"])
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df = df[df["product_name"] != ""]
    df = df.drop_duplicates(subset=["order_id", "product_id"])
    after_rows = len(df)

    cleaning_report = pd.DataFrame({
        "Step": ["Rows loaded", "Rows before cleaning", "Duplicate order/product rows removed", "Rows after cleaning"],
        "Value": [raw_rows, before_rows, duplicate_rows, after_rows]
    })

    total_orders = df["order_id"].nunique()

    sales = (
        df.groupby("product_name")
        .agg(Orders=("order_id", "nunique"), Units=("product_id", "count"))
        .reset_index()
        .sort_values("Units", ascending=False)
    )

    if "reordered" in df.columns:
        reorder_rates = (
            df.groupby("product_name")["reordered"]
            .mean()
            .reset_index()
            .rename(columns={"reordered": "Reorder Rate"})
        )
        sales = sales.merge(reorder_rates, on="product_name", how="left")
    else:
        sales["Reorder Rate"] = 0.0

    sales["Order Share"] = sales["Orders"] / total_orders
    sales["Estimated Revenue"] = sales["Units"] * avg_price
    sales["Estimated Profit"] = sales["Estimated Revenue"] * profit_margin
    sales["Cumulative Sales Share"] = sales["Units"].cumsum() / sales["Units"].sum()

    def abc_class(x):
        if x <= 0.80:
            return "A - Core seller"
        if x <= 0.95:
            return "B - Normal seller"
        return "C - Slow seller"

    sales["Product Class"] = sales["Cumulative Sales Share"].apply(abc_class)
    sales["Retail Action"] = sales["Product Class"].map({
        "A - Core seller": "Stock up and keep visible",
        "B - Normal seller": "Keep normal stock and test bundles",
        "C - Slow seller": "Put on sale or reduce reorder"
    })

    basket_size = df.groupby("order_id")["product_id"].nunique().reset_index(name="Basket Size")
    basket_dist = basket_size["Basket Size"].value_counts().reset_index()
    basket_dist.columns = ["Basket Size", "Number of Orders"]
    basket_dist = basket_dist.sort_values("Basket Size")

    top_products = sales.head(top_n_products)["product_name"].tolist()
    basket_source = df[df["product_name"].isin(top_products)].drop_duplicates(subset=["order_id", "product_name"]).copy()
    basket_source["in_cart"] = True
    basket = basket_source.pivot_table(
        index="order_id",
        columns="product_name",
        values="in_cart",
        aggfunc="max",
        fill_value=False
    ).astype(bool)

    if "department" in df.columns:
        departments = (
            df.groupby("department")
            .agg(Orders=("order_id", "nunique"), Units=("product_id", "count"), Products=("product_name", "nunique"))
            .reset_index()
            .sort_values("Units", ascending=False)
        )
        departments["Estimated Revenue"] = departments["Units"] * avg_price
        departments["Estimated Profit"] = departments["Estimated Revenue"] * profit_margin
    else:
        departments = pd.DataFrame()

    if "aisle" in df.columns:
        aisles = (
            df.groupby("aisle")
            .agg(Orders=("order_id", "nunique"), Units=("product_id", "count"), Products=("product_name", "nunique"))
            .reset_index()
            .sort_values("Units", ascending=False)
        )
        aisles["Estimated Revenue"] = aisles["Units"] * avg_price
        aisles["Estimated Profit"] = aisles["Units"] * avg_price * profit_margin
    else:
        aisles = pd.DataFrame()

    return df, cleaning_report, sales, basket_size, basket_dist, basket, departments, aisles


def itemset_to_text(itemset):
    return ", ".join(sorted(list(itemset)))


@st.cache_data(show_spinner=False)
def run_market_basket(row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, algorithm, min_support, min_confidence, min_lift, max_len):
    _, _, _, _, _, basket, _, _ = prepare_base_data(
        row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products
    )

    if basket.empty:
        return pd.DataFrame(), pd.DataFrame()

    if algorithm == "Apriori":
        frequent = apriori(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        frequent = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=max_len)

    if frequent.empty:
        return frequent, pd.DataFrame()

    frequent["Item Count"] = frequent["itemsets"].apply(len)
    frequent["Itemset"] = frequent["itemsets"].apply(itemset_to_text)
    frequent = frequent.sort_values(["support", "Item Count"], ascending=False)

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        return frequent, pd.DataFrame()

    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return frequent, pd.DataFrame()

    rules["Customer Buys"] = rules["antecedents"].apply(itemset_to_text)
    rules["Recommend"] = rules["consequents"].apply(itemset_to_text)
    rules["Rule"] = rules["Customer Buys"] + " → " + rules["Recommend"]
    rules["Rule Score"] = rules["confidence"] * rules["lift"]

    def retail_use(row):
        if row["lift"] >= 2 and row["confidence"] >= 0.15:
            return "Best for bundle or checkout recommendation"
        if row["lift"] >= 1.5:
            return "Good for shelf placement or coupon test"
        return "Use carefully as a test rule"

    rules["Retail Use"] = rules.apply(retail_use, axis=1)
    rules = rules.sort_values(["Rule Score", "lift", "confidence"], ascending=False)
    return frequent, rules


@st.cache_data(show_spinner=False)
def run_reorder_classification(row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products):
    df, _, _, _, _, _, _, _ = prepare_base_data(
        row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products
    )

    if "reordered" not in df.columns:
        return pd.DataFrame(), [], pd.DataFrame(), {}, {}, 0, 0

    work = df.copy()
    product_stats = (
        work.groupby("product_id")
        .agg(product_order_count=("order_id", "count"), product_reorder_rate=("reordered", "mean"))
        .reset_index()
    )
    work = work.merge(product_stats, on="product_id", how="left")

    basket_feature = work.groupby("order_id")["product_id"].nunique().reset_index(name="basket_size")
    work = work.merge(basket_feature, on="order_id", how="left")

    possible_features = [
        "add_to_cart_order", "order_dow", "order_hour_of_day", "days_since_prior_order",
        "basket_size", "product_order_count", "product_reorder_rate", "aisle_id", "department_id"
    ]
    features = [c for c in possible_features if c in work.columns]

    model_df = work[features + ["reordered"]].replace([np.inf, -np.inf], np.nan).copy()
    model_df = model_df.dropna(subset=["reordered"])

    for col in features:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    model_df["reordered"] = model_df["reordered"].astype(int)

    if len(model_df) > 60000:
        model_df = model_df.sample(60000, random_state=42)

    X = model_df[features]
    y = model_df["reordered"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    rows = []
    reports = {}
    matrices = {}

    for name, model in models.items():
        if name in ["Naive Bayes", "KNN"]:
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1 Score": f1_score(y_test, prediction, zero_division=0)
        })

        reports[name] = classification_report(y_test, prediction, output_dict=True, zero_division=0)
        matrices[name] = confusion_matrix(y_test, prediction)

    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    return model_df, features, results, reports, matrices, len(X_train), len(X_test)


def recommend_from_cart(rules, cart_items):
    if rules.empty or len(cart_items) == 0:
        return pd.DataFrame()

    cart_set = set(cart_items)
    matched = rules[rules["antecedents"].apply(lambda x: set(x).issubset(cart_set))].copy()

    rows = []
    for _, row in matched.iterrows():
        for item in row["consequents"]:
            if item not in cart_set:
                rows.append({
                    "Recommended Product": item,
                    "Because Cart Has": itemset_to_text(row["antecedents"]),
                    "Confidence": row["confidence"],
                    "Lift": row["lift"],
                    "Rule Score": row["Rule Score"],
                    "Retail Use": row["Retail Use"]
                })

    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs

    recs = recs.sort_values(["Rule Score", "Lift"], ascending=False)
    recs = recs.drop_duplicates("Recommended Product")
    return recs


def download_button(df, label, filename):
    if not df.empty:
        st.download_button(
            label=label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )

# ---------------- LOAD BASE DATA ----------------
with st.spinner("Loading local retail data..."):
    df, cleaning_report, sales, basket_size, basket_dist, basket, departments, aisles = prepare_base_data(
        row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products
    )

high_sellers = sales.head(20)
low_sellers = sales.tail(20).sort_values("Units", ascending=True)
stock_up = sales[sales["Product Class"] == "A - Core seller"].head(30)
sale_candidates = sales[sales["Product Class"] == "C - Slow seller"].sort_values("Units", ascending=True).head(30)

if "frequent" not in st.session_state:
    st.session_state.frequent = pd.DataFrame()
if "rules" not in st.session_state:
    st.session_state.rules = pd.DataFrame()
if "mining_done" not in st.session_state:
    st.session_state.mining_done = False

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="badge">Local CSV Data Mining System • No API • Retail Decision Support</div>
    <h1>Retail Growth Advisor</h1>
    <p>Turns grocery transactions into clear business actions: stock up, discount, bundle, recommend, and predict reorder behavior.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI ROW ----------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    kpi("Orders Analyzed", f"{df['order_id'].nunique():,}", "cleaned transactions")
with m2:
    kpi("Products Found", f"{df['product_name'].nunique():,}", "unique products")
with m3:
    kpi("Average Basket", f"{basket_size['Basket Size'].mean():.2f}", "items per order")
with m4:
    kpi("Rules Ready", f"{len(st.session_state.rules):,}", "after basket mining")

# ---------------- MAIN TABS ----------------
tab_overview, tab_actions, tab_mining, tab_prediction, tab_evidence = st.tabs([
    "Executive View",
    "Retail Actions",
    "Basket Mining",
    "Reorder Prediction",
    "Evidence & Downloads"
])

# ============================================================
# TAB 1: EXECUTIVE VIEW
# ============================================================
with tab_overview:
    st.subheader("What should the retailer do first?")

    c1, c2, c3 = st.columns(3)

    with c1:
        if not stock_up.empty:
            item = stock_up.iloc[0]
            card(
                "✅ Stock Up",
                f"Keep **{item['product_name']}** available. It appears as a core seller with **{int(item['Units']):,} units** in the selected data. Estimated profit impact: **{money(item['Estimated Profit'])}**.",
                "green"
            )

    with c2:
        if not sale_candidates.empty:
            item = sale_candidates.iloc[0]
            card(
                "🏷️ Put on Sale",
                f"Test a discount or bundle for **{item['product_name']}**. It is a slow seller with **{int(item['Units']):,} units** in this sample.",
                "orange"
            )

    with c3:
        if st.session_state.rules.empty:
            card(
                "🧺 Find Bundles",
                "Go to **Basket Mining** and click **Run Basket Mining**. That will reveal product combinations for checkout recommendations and shelf placement.",
                "blue"
            )
        else:
            rule = st.session_state.rules.iloc[0]
            card(
                "🧺 Best Bundle",
                f"When customers buy **{rule['Customer Buys']}**, recommend **{rule['Recommend']}**. Lift: **{rule['lift']:.2f}**. Confidence: **{rule['confidence']:.2f}**.",
                "blue"
            )

    st.markdown("---")
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Top Product Demand")
        st.bar_chart(high_sellers.set_index("product_name")["Units"])

    with right:
        st.subheader("Basket Size Pattern")
        st.bar_chart(basket_dist.set_index("Basket Size")["Number of Orders"])

    st.info("Presentation tip: start here. Explain that the app first gives a simple business decision, then users can open the mining and evidence tabs for the technical proof.")

# ============================================================
# TAB 2: RETAIL ACTIONS
# ============================================================
with tab_actions:
    st.subheader("Retail Action Center")
    st.caption("This tab hides large tables behind buttons/toggles so the app feels simple for a real user.")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        show_stock = st.toggle("Show stock-up list", value=False)
    with a2:
        show_sale = st.toggle("Show sale list", value=False)
    with a3:
        show_sections = st.toggle("Show departments/aisles", value=False)
    with a4:
        show_all_products = st.toggle("Show full product strategy", value=False)

    st.markdown("---")

    if not any([show_stock, show_sale, show_sections, show_all_products]):
        c1, c2, c3 = st.columns(3)
        with c1:
            card("1. Protect Core Sellers", "High-demand A-class products should stay in stock. Running out of these products can lose easy sales.", "green")
        with c2:
            card("2. Move Slow Sellers", "C-class products are better candidates for discounts, small bundles, or lower reorder quantity.", "orange")
        with c3:
            card("3. Increase Basket Size", "Use basket rules to recommend related products during checkout or place them near each other in the store.", "purple")

    if show_stock:
        st.subheader("Stock-Up Products")
        st.caption("These products drive demand. A retailer should keep them available and visible.")
        st.dataframe(
            stock_up[["product_name", "Units", "Orders", "Reorder Rate", "Estimated Revenue", "Estimated Profit", "Retail Action"]],
            width="stretch"
        )

    if show_sale:
        st.subheader("Sale / Reduce-Reorder Products")
        st.caption("These products are slow sellers. A retailer can test discounts or bundle them with popular items.")
        st.dataframe(
            sale_candidates[["product_name", "Units", "Orders", "Reorder Rate", "Estimated Revenue", "Estimated Profit", "Retail Action"]],
            width="stretch"
        )

    if show_sections:
        section_left, section_right = st.columns(2)
        with section_left:
            st.subheader("Department Performance")
            if departments.empty:
                st.warning("departments.csv was not found.")
            else:
                st.bar_chart(departments.head(12).set_index("department")["Units"])
                with st.expander("Open department table"):
                    st.dataframe(departments.head(30), width="stretch")
        with section_right:
            st.subheader("Aisle Performance")
            if aisles.empty:
                st.warning("aisles.csv was not found.")
            else:
                st.bar_chart(aisles.head(12).set_index("aisle")["Units"])
                with st.expander("Open aisle table"):
                    st.dataframe(aisles.head(30), width="stretch")

    if show_all_products:
        st.subheader("Full Product Strategy Table")
        st.caption("A = core seller, B = normal seller, C = slow seller.")
        st.dataframe(sales, width="stretch")

# ============================================================
# TAB 3: BASKET MINING
# ============================================================
with tab_mining:
    st.subheader("Basket Mining Lab")
    st.caption("This step runs only when clicked because association rule mining is the heaviest part.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        run_button = st.button("Run Basket Mining", type="primary")
    with c2:
        clear_button = st.button("Clear Rules")
    with c3:
        st.write(f"Current settings: **{algorithm}**, support **{min_support}**, confidence **{min_confidence}**, lift **{min_lift}**")

    if clear_button:
        st.session_state.frequent = pd.DataFrame()
        st.session_state.rules = pd.DataFrame()
        st.session_state.mining_done = False
        st.rerun()

    if run_button:
        with st.spinner("Mining product combinations..."):
            frequent, rules = run_market_basket(
                row_limit, use_prior, use_order_time, avg_price, profit_margin,
                top_n_products, algorithm, min_support, min_confidence, min_lift, max_len
            )
            st.session_state.frequent = frequent
            st.session_state.rules = rules
            st.session_state.mining_done = True

    frequent = st.session_state.frequent
    rules = st.session_state.rules

    if not st.session_state.mining_done:
        card("Ready to Mine", "Click **Run Basket Mining** to find frequent itemsets and association rules. Use Fast Demo for quick presentation performance.", "blue")
    else:
        r1, r2, r3 = st.columns(3)
        with r1:
            kpi("Frequent Itemsets", f"{len(frequent):,}", "product patterns")
        with r2:
            kpi("Association Rules", f"{len(rules):,}", "recommendation rules")
        with r3:
            avg_lift = rules["lift"].mean() if not rules.empty else 0
            kpi("Average Lift", f"{avg_lift:.2f}", "rule strength")

        if rules.empty:
            st.warning("No rules found. Try lower support, lower lift, or more rows.")
        else:
            best = rules.iloc[0]
            card(
                "Best Recommendation Rule",
                f"If a customer buys **{best['Customer Buys']}**, recommend **{best['Recommend']}**. This rule has lift **{best['lift']:.2f}** and confidence **{best['confidence']:.2f}**.",
                "green"
            )

            show_rule_table = st.toggle("Open association rule table", value=False)
            if show_rule_table:
                show_rules = rules[["Rule", "Customer Buys", "Recommend", "support", "confidence", "lift", "Rule Score", "Retail Use"]].head(60)
                st.dataframe(show_rules, width="stretch")

            show_itemsets = st.toggle("Open frequent itemset table", value=False)
            if show_itemsets and not frequent.empty:
                st.dataframe(frequent[["Itemset", "Item Count", "support"]].head(60), width="stretch")

            st.subheader("Cart Advisor")
            st.caption("Pick products in a customer cart. The app suggests what to recommend next.")
            cart_items = st.multiselect("Customer cart", sorted(list(basket.columns)))
            recs = recommend_from_cart(rules, cart_items)

            if len(cart_items) == 0:
                st.info("Choose one or more cart products to see recommendations.")
            elif recs.empty:
                st.warning("No recommendation found for this cart. Try another product.")
            else:
                top_rec = recs.iloc[0]
                card(
                    "Checkout Recommendation",
                    f"Recommend **{top_rec['Recommended Product']}** because the cart already has **{top_rec['Because Cart Has']}**. Lift: **{top_rec['Lift']:.2f}**. Confidence: **{top_rec['Confidence']:.2f}**.",
                    "purple"
                )
                with st.expander("Open all cart recommendations"):
                    st.dataframe(recs.head(20), width="stretch")

# ============================================================
# TAB 4: REORDER PREDICTION
# ============================================================
with tab_prediction:
    st.subheader("Reorder Prediction")
    st.caption("Classification model: predicts reordered = 0 or 1. It only trains after clicking the button.")

    if "reordered" not in df.columns:
        st.warning("The reordered column was not found. Use order_products__train.csv or prior data with the reordered column.")
    else:
        run_predict = st.button("Run Classification Models", type="primary")

        if run_predict:
            with st.spinner("Training Decision Tree, Random Forest, Naive Bayes, and KNN..."):
                model_df, features, results, reports, matrices, train_size, test_size = run_reorder_classification(
                    row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products
                )
                st.session_state.class_model_df = model_df
                st.session_state.class_features = features
                st.session_state.class_results = results
                st.session_state.class_reports = reports
                st.session_state.class_matrices = matrices
                st.session_state.class_train_size = train_size
                st.session_state.class_test_size = test_size

        if "class_results" not in st.session_state:
            card(
                "Classification Not Run Yet",
                "Click **Run Classification Models** to build a training/test split and compare Decision Tree, Random Forest, Naive Bayes, and KNN.",
                "blue"
            )
        else:
            results = st.session_state.class_results
            best = results.iloc[0]
            p1, p2, p3 = st.columns(3)
            with p1:
                kpi("Best Model", best["Model"], "highest F1 score")
            with p2:
                kpi("Accuracy", f"{best['Accuracy']:.3f}", "test set")
            with p3:
                kpi("F1 Score", f"{best['F1 Score']:.3f}", "balanced score")

            st.write("Selected features:", st.session_state.class_features)
            st.write("Training set size:", st.session_state.class_train_size, " | Test set size:", st.session_state.class_test_size)

            st.subheader("Model Comparison")
            st.bar_chart(results.set_index("Model")[["Accuracy", "F1 Score"]])

            if st.toggle("Open model results table", value=False):
                st.dataframe(results, width="stretch")

            class_dist = st.session_state.class_model_df["reordered"].value_counts().reset_index()
            class_dist.columns = ["Class", "Count"]
            class_dist["Meaning"] = class_dist["Class"].map({0: "Not Reordered", 1: "Reordered"})

            if st.toggle("Open class distribution and confusion matrix", value=False):
                st.subheader("Class Distribution")
                st.dataframe(class_dist, width="stretch")

                chosen_model = st.selectbox("Choose model", results["Model"].tolist())
                matrix = pd.DataFrame(
                    st.session_state.class_matrices[chosen_model],
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"]
                )
                st.subheader("Confusion Matrix")
                st.dataframe(matrix, width="stretch")

                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(st.session_state.class_reports[chosen_model]).transpose(), width="stretch")

# ============================================================
# TAB 5: EVIDENCE AND DOWNLOADS
# ============================================================
with tab_evidence:
    st.subheader("Evidence, Testing, and Downloads")

    e1, e2 = st.columns(2)
    with e1:
        with st.expander("Data cleaning proof"):
            st.dataframe(cleaning_report, width="stretch")
            st.dataframe(df.head(30), width="stretch")

        with st.expander("Basket format proof"):
            st.caption("Rows are orders. Columns are products. True means the product appeared in that order.")
            st.write("Basket shape:", basket.shape)
            st.dataframe(basket.head(8), width="stretch")

    with e2:
        st.subheader("Run Support Test")
        st.caption("This compares support values to show evaluation, not just final output.")
        if st.button("Run Support Test", type="primary"):
            rows = []
            for support_value in [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]:
                temp_frequent, temp_rules = run_market_basket(
                    row_limit, use_prior, use_order_time, avg_price, profit_margin,
                    top_n_products, algorithm, support_value, min_confidence, min_lift, max_len
                )
                rows.append({
                    "Support": support_value,
                    "Frequent Itemsets": len(temp_frequent),
                    "Association Rules": len(temp_rules),
                    "Average Lift": temp_rules["lift"].mean() if not temp_rules.empty else 0,
                    "Average Confidence": temp_rules["confidence"].mean() if not temp_rules.empty else 0
                })
            st.session_state.support_test = pd.DataFrame(rows)

        if "support_test" in st.session_state:
            st.dataframe(st.session_state.support_test, width="stretch")
            st.line_chart(st.session_state.support_test.set_index("Support")[["Frequent Itemsets", "Association Rules"]])

    st.markdown("---")
    st.subheader("Download Results")
    d1, d2, d3 = st.columns(3)
    with d1:
        download_button(sales, "Download Product Strategy", "product_strategy.csv")
        download_button(stock_up, "Download Stock-Up List", "stock_up_products.csv")
    with d2:
        download_button(sale_candidates, "Download Sale Candidate List", "sale_candidates.csv")
        download_button(cleaning_report, "Download Cleaning Report", "cleaning_report.csv")
    with d3:
        if not st.session_state.rules.empty:
            export_rules = st.session_state.rules[["Rule", "Customer Buys", "Recommend", "support", "confidence", "lift", "Rule Score", "Retail Use"]]
            download_button(export_rules, "Download Basket Rules", "basket_rules.csv")
        if "class_results" in st.session_state:
            download_button(st.session_state.class_results, "Download Classification Results", "classification_results.csv")

st.caption("Profit is estimated because this dataset does not include real item prices or product costs. Heavy mining and model training are click-based to keep the app fast.")
