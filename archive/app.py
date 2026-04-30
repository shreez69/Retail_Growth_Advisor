import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ============================================================
# RETAIL GROWTH ADVISOR PRO
# No API. Local CSV only.
# Data mining methods: Association Rules, FP-Growth/Apriori,
# ABC/Pareto Analysis, Product Clustering, Reorder Classification,
# Timing Analysis, Campaign Simulation, Cart Recommendation Engine.
# ============================================================

st.set_page_config(
    page_title="Retail Growth Advisor Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
[data-testid="stSidebar"] {
    display: none;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
.hero-pro {
    background: radial-gradient(circle at top left, rgba(56,189,248,0.55) 0%, transparent 24%),
                radial-gradient(circle at bottom right, rgba(124,58,237,0.32) 0%, transparent 26%),
                linear-gradient(135deg, #020617 0%, #0f172a 44%, #111827 100%);
    color: white;
    padding: 34px 40px;
    border-radius: 30px;
    box-shadow: 0 22px 52px rgba(15, 23, 42, 0.30);
    margin-bottom: 17px;
    border: 1px solid rgba(255,255,255,0.14);
}
.hero-pro h1 {
    font-size: 48px;
    line-height: 1.03;
    margin: 0 0 10px 0;
    letter-spacing: -1.4px;
}
.hero-pro p {
    font-size: 18px;
    max-width: 1050px;
    color: #cbd5e1;
    margin: 0;
}
.badge-row {margin-bottom: 14px;}
.badge {
    display: inline-block;
    background: rgba(14,165,233,0.16);
    color: #bae6fd;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 13px;
    margin: 2px 4px 2px 0;
    border: 1px solid rgba(186,230,253,0.25);
    font-weight: 800;
}
.control-shell {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 15px 18px;
    box-shadow: 0 10px 28px rgba(15,23,42,0.08);
    margin-bottom: 15px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 18px 20px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    min-height: 116px;
}
.kpi-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .07em;
}
.kpi-number {
    color: #020617;
    font-size: 30px;
    font-weight: 950;
    margin-top: 6px;
}
.kpi-note {
    color: #64748b;
    font-size: 13px;
    margin-top: 3px;
}
.panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 21px 22px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
    margin-bottom: 16px;
}
.card-title {
    color: #020617;
    font-size: 20px;
    font-weight: 900;
    margin-bottom: 7px;
}
.card-body {
    color: #475569;
    font-size: 15px;
    line-height: 1.48;
}
.green {border-left: 8px solid #16a34a;}
.orange {border-left: 8px solid #f59e0b;}
.blue {border-left: 8px solid #2563eb;}
.red {border-left: 8px solid #dc2626;}
.purple {border-left: 8px solid #7c3aed;}
.black {border-left: 8px solid #020617;}
.alert-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 15px 17px;
    margin-bottom: 10px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}
.alert-title {
    font-weight: 900;
    color: #020617;
    margin-bottom: 4px;
}
.alert-text {
    color: #475569;
    font-size: 14px;
    line-height: 1.42;
}
.chip {
    display: inline-block;
    background: #e0f2fe;
    color: #075985;
    border: 1px solid #bae6fd;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
    margin: 3px 4px 3px 0;
}
.big-score {
    font-size: 58px;
    font-weight: 950;
    color: #020617;
    line-height: 1;
}
.score-caption {
    color: #64748b;
    font-size: 14px;
}
.cart-pill {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    background: #fee2e2;
    color: #991b1b;
    font-weight: 850;
    font-size: 13px;
    margin: 3px 4px;
    border: 1px solid #fecaca;
}
.hl {
    background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%);
    color: #78350f;
    padding: 2px 7px;
    border-radius: 9px;
    font-weight: 900;
    white-space: normal;
}
.nav-note {
    color: #64748b;
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 10px;
}
.stRadio > div {
    gap: 8px;
}
.stButton > button {
    border-radius: 14px;
    font-weight: 850;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 15px rgba(15,23,42,0.07);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
    border: none;
}
[data-testid="stExpander"] {
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 7px 18px rgba(15,23,42,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- REQUIRED FILE CHECK ----------------
required_files = ["order_products__train.csv", "products.csv"]
missing_files = [f for f in required_files if not Path(f).exists()]

if missing_files:
    st.error("Missing required file(s): " + ", ".join(missing_files))
    st.info("Put app.py in the same folder as order_products__train.csv and products.csv.")
    st.stop()

# ---------------- HELPERS ----------------
def money(x):
    return f"${x:,.0f}"


def pct(x):
    return f"{x * 100:.1f}%"


def safe_div(a, b):
    return a / b if b else 0


def itemset_to_text(itemset):
    return ", ".join(sorted(list(itemset)))


def highlight_text(text):
    text = str(text)
    return re.sub("[*][*](.*?)[*][*]", lambda m: f"<span class='hl'>{m.group(1)}</span>", text)


def card(title, body, border="blue"):
    title = highlight_text(title)
    body = highlight_text(body)
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


def alert_box(title, text, tag="Action"):
    title = highlight_text(title)
    text = highlight_text(text)
    st.markdown(
        f"""
        <div class="alert-box">
            <div class="alert-title">{title}</div>
            <div class="alert-text">{text}</div>
            <span class="chip">{tag}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def download_button(df, label, filename):
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.download_button(
            label=label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )


def no_banana_filter(df, product_col="Recommended Product"):
    if df.empty or product_col not in df.columns:
        return df
    bad_words = ["banana", "bag of organic bananas"]
    mask = ~df[product_col].str.lower().apply(lambda x: any(word in x for word in bad_words))
    filtered = df[mask]
    return filtered if not filtered.empty else df

# ---------------- TOP HERO ----------------
st.markdown("""
<div class="hero-pro">
    <h1>Retail Growth Advisor Pro</h1>
    <p>A clean business dashboard that turns grocery transactions into smarter stocking, promotions, product scanning, basket recommendations, and reorder predictions.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- TOP MENU BAR / CONTROLS ----------------
st.markdown('<div class="control-shell">', unsafe_allow_html=True)

quick1, quick2, quick3 = st.columns([1, 1, 2])
with quick1:
    app_theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
with quick2:
    view_mode = st.selectbox("View style", ["Simple", "Advanced"], index=0)
with quick3:
    st.caption("Simple mode hides extra pages and makes the app easier to present. Advanced mode shows all data mining features.")

if view_mode == "Simple":
    menu_options = ["Dashboard", "Smart Cart", "Product Scanner", "Mining Lab", "Model Lab", "Evidence"]
else:
    menu_options = ["Dashboard", "Smart Cart", "Product Scanner", "Mining Lab", "Advanced Analytics", "Model Lab", "Evidence"]

page = st.radio(
    "Menu",
    menu_options,
    horizontal=True,
    label_visibility="collapsed"
)

if app_theme == "Dark":
    st.markdown("""
    <style>
    .stApp {background: #020617; color: #e5e7eb;}
    .control-shell, .panel, .kpi-card, [data-testid="stExpander"] {
        background: #0f172a !important;
        border-color: #334155 !important;
        color: #e5e7eb !important;
    }
    .card-title, .kpi-number, .alert-title, h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
    }
    .card-body, .kpi-note, .score-caption, .nav-note, p, label, .stCaptionContainer {
        color: #cbd5e1 !important;
    }
    .alert-box {
        background: #111827 !important;
        border-color: #334155 !important;
    }
    .alert-text {color: #cbd5e1 !important;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background: #ffffff; color: #0f172a;}
    </style>
    """, unsafe_allow_html=True)

with st.expander("⚙️ Open Dashboard Controls", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        speed_mode = st.selectbox("Run mode", ["Fast demo", "Balanced", "Deeper analysis"], index=0)
    mode_defaults = {
        "Fast demo": {"rows": 60000, "top": 100, "support": 0.005, "confidence": 0.05},
        "Balanced": {"rows": 120000, "top": 150, "support": 0.003, "confidence": 0.05},
        "Deeper analysis": {"rows": 220000, "top": 220, "support": 0.002, "confidence": 0.05},
    }
    with c2:
        row_limit = st.slider("Rows to read", 20000, 300000, mode_defaults[speed_mode]["rows"], 20000)
    with c3:
        use_prior = st.checkbox("Use prior-order file if available", value=False)
        use_order_time = st.checkbox("Use order time features", value=True)
    with c4:
        top_n_products = st.slider("Products used in basket mining", 40, 300, mode_defaults[speed_mode]["top"], 10)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        algorithm = st.selectbox("Mining algorithm", ["FP-Growth", "Apriori"], index=0)
    with c6:
        min_support = st.select_slider("Support", options=[0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02], value=mode_defaults[speed_mode]["support"])
    with c7:
        min_confidence = st.select_slider("Confidence", options=[0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], value=mode_defaults[speed_mode]["confidence"])
    with c8:
        min_lift = st.select_slider("Lift", options=[1.0, 1.1, 1.2, 1.5, 2.0], value=1.1)
        max_len = st.selectbox("Max items in pattern", [2, 3], index=1)

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        avg_price = st.slider("Average item price ($)", 1.0, 25.0, 5.0, 0.5)
    with c10:
        profit_margin = st.slider("Profit margin", 0.10, 0.70, 0.30, 0.05)
    with c11:
        promo_discount = st.slider("Promo discount", 0.00, 0.50, 0.10, 0.05)
    with c12:
        expected_uplift = st.slider("Expected sales lift", 0.00, 0.80, 0.15, 0.05)
        campaign_budget = st.slider("Campaign budget ($)", 100, 10000, 1500, 100)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "cart" not in st.session_state:
    st.session_state.cart = []
if "frequent" not in st.session_state:
    st.session_state.frequent = pd.DataFrame()
if "rules" not in st.session_state:
    st.session_state.rules = pd.DataFrame()
if "mining_done" not in st.session_state:
    st.session_state.mining_done = False
if "last_scanned_product" not in st.session_state:
    st.session_state.last_scanned_product = None

# ---------------- DATA PREP ----------------
@st.cache_data(show_spinner=False)
def prepare_base_data(row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift):
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
        df = df.merge(pd.read_csv("aisles.csv"), on="aisle_id", how="left")

    if Path("departments.csv").exists():
        df = df.merge(pd.read_csv("departments.csv"), on="department_id", how="left")

    if use_order_time and Path("orders.csv").exists():
        order_cols = pd.read_csv("orders.csv", nrows=1).columns.tolist()
        needed = [c for c in ["order_id", "user_id", "order_dow", "order_hour_of_day", "days_since_prior_order"] if c in order_cols]
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

    group_cols = {
        "Orders": ("order_id", "nunique"),
        "Units": ("product_id", "count")
    }
    if "add_to_cart_order" in df.columns:
        group_cols["Avg Cart Position"] = ("add_to_cart_order", "mean")

    sales = df.groupby("product_name").agg(**group_cols).reset_index().sort_values("Units", ascending=False)

    if "reordered" in df.columns:
        reorder_rates = df.groupby("product_name")["reordered"].mean().reset_index().rename(columns={"reordered": "Reorder Rate"})
        sales = sales.merge(reorder_rates, on="product_name", how="left")
    else:
        sales["Reorder Rate"] = 0.0

    sales["Order Share"] = sales["Orders"] / max(total_orders, 1)
    sales["Estimated Revenue"] = sales["Units"] * avg_price
    sales["Estimated Profit"] = sales["Estimated Revenue"] * profit_margin
    sales["Cumulative Sales Share"] = sales["Units"].cumsum() / max(sales["Units"].sum(), 1)

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

    max_units = max(sales["Units"].max(), 1)
    max_orders = max(sales["Orders"].max(), 1)
    sales["Restock Priority Score"] = (
        (sales["Units"] / max_units) * 55
        + sales["Reorder Rate"].fillna(0) * 25
        + (sales["Orders"] / max_orders) * 20
    ).round(1)
    sales["Markdown Risk Score"] = (
        (1 - (sales["Units"] / max_units)) * 55
        + (1 - sales["Reorder Rate"].fillna(0)) * 25
        + (sales["Product Class"].eq("C - Slow seller").astype(int)) * 20
    ).round(1)

    sales["Promo Revenue Gain"] = sales["Estimated Revenue"] * expected_uplift
    sales["Promo Discount Cost"] = sales["Promo Revenue Gain"] * promo_discount
    sales["Promo Profit Gain"] = sales["Promo Revenue Gain"] * profit_margin - sales["Promo Discount Cost"]

    # Extra backend scores for a bigger project feel.
    if "Avg Cart Position" in sales.columns:
        max_position = max(sales["Avg Cart Position"].max(), 1)
        cart_position_score = 1 - (sales["Avg Cart Position"] / max_position)
    else:
        cart_position_score = 0.5

    sales["Bundle Potential Score"] = (
        sales["Order Share"].fillna(0) * 35
        + sales["Reorder Rate"].fillna(0) * 30
        + cart_position_score * 20
        + (sales["Product Class"].eq("B - Normal seller").astype(int)) * 15
    ).round(1)

    sales["Retention Value Score"] = (
        sales["Reorder Rate"].fillna(0) * 60
        + (sales["Orders"] / max_orders) * 25
        + (sales["Estimated Profit"] / max(sales["Estimated Profit"].max(), 1)) * 15
    ).round(1)

    sales["Promotion Priority Score"] = (
        sales["Markdown Risk Score"].fillna(0) * 0.55
        + sales["Bundle Potential Score"].fillna(0) * 0.25
        + (1 - sales["Reorder Rate"].fillna(0)) * 20
    ).round(1)

    sales["Risk Level"] = np.select(
        [sales["Markdown Risk Score"] >= 80, sales["Markdown Risk Score"] >= 60],
        ["High", "Medium"],
        default="Low"
    )

    # Attach most common product metadata
    meta_cols = [c for c in ["product_id", "aisle_id", "department_id", "aisle", "department"] if c in df.columns]
    if meta_cols:
        meta = df.groupby("product_name")[meta_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).reset_index()
        sales = sales.merge(meta, on="product_name", how="left")

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

    departments = pd.DataFrame()
    if "department" in df.columns:
        departments = (
            df.groupby("department")
            .agg(Orders=("order_id", "nunique"), Units=("product_id", "count"), Products=("product_name", "nunique"))
            .reset_index()
            .sort_values("Units", ascending=False)
        )
        departments["Estimated Revenue"] = departments["Units"] * avg_price
        departments["Estimated Profit"] = departments["Estimated Revenue"] * profit_margin

    aisles = pd.DataFrame()
    if "aisle" in df.columns:
        aisles = (
            df.groupby("aisle")
            .agg(Orders=("order_id", "nunique"), Units=("product_id", "count"), Products=("product_name", "nunique"))
            .reset_index()
            .sort_values("Units", ascending=False)
        )
        aisles["Estimated Revenue"] = aisles["Units"] * avg_price
        aisles["Estimated Profit"] = aisles["Units"] * avg_price * profit_margin

    timing = pd.DataFrame()
    if {"order_dow", "order_hour_of_day"}.issubset(df.columns):
        timing = (
            df.groupby(["order_dow", "order_hour_of_day"])
            .agg(Units=("product_id", "count"), Orders=("order_id", "nunique"))
            .reset_index()
            .sort_values("Units", ascending=False)
        )

    return df, cleaning_report, sales, basket_size, basket_dist, basket, departments, aisles, timing


@st.cache_data(show_spinner=False)
def run_market_basket(row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift, algorithm, min_support, min_confidence, min_lift, max_len):
    _, _, _, _, _, basket, _, _, _ = prepare_base_data(
        row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift
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
def build_cooccurrence_recs(row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift, cart_items):
    _, _, sales, _, _, basket, _, _, _ = prepare_base_data(
        row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift
    )
    if basket.empty or len(cart_items) == 0:
        return pd.DataFrame()

    available_items = [item for item in cart_items if item in basket.columns]
    if not available_items:
        return pd.DataFrame()

    cart_orders = basket[available_items].all(axis=1)
    cart_count = int(cart_orders.sum())
    if cart_count == 0:
        cart_orders = basket[available_items].any(axis=1)
        cart_count = int(cart_orders.sum())

    if cart_count == 0:
        return pd.DataFrame()

    candidate_counts = basket.loc[cart_orders].sum().sort_values(ascending=False)
    candidate_counts = candidate_counts.drop(labels=[x for x in cart_items if x in candidate_counts.index], errors="ignore")

    recs = candidate_counts.reset_index()
    recs.columns = ["Recommended Product", "Co-occurrence Count"]
    recs["Confidence"] = recs["Co-occurrence Count"] / cart_count
    base_rates = basket.mean().rename("Base Rate").reset_index().rename(columns={"product_name": "Recommended Product", "index": "Recommended Product"})
    recs = recs.merge(base_rates, on="Recommended Product", how="left")
    recs["Lift"] = recs["Confidence"] / recs["Base Rate"].replace(0, np.nan)
    recs["Rule Score"] = recs["Confidence"] * recs["Lift"].fillna(0)
    recs["Because Cart Has"] = ", ".join(cart_items)
    recs["Retail Use"] = "Co-occurrence fallback recommendation"

    profit_map = sales.set_index("product_name")["Estimated Profit"].to_dict()
    reorder_map = sales.set_index("product_name")["Reorder Rate"].to_dict()
    recs["Estimated Profit"] = recs["Recommended Product"].map(profit_map).fillna(0)
    recs["Reorder Rate"] = recs["Recommended Product"].map(reorder_map).fillna(0)
    recs = recs.sort_values(["Rule Score", "Lift", "Co-occurrence Count"], ascending=False)
    return recs


def recommend_from_cart(rules, fallback_recs, cart_items, sales, mode="Diversified", avoid_banana=True, top_k=8):
    if len(cart_items) == 0:
        return pd.DataFrame()

    rows = []
    cart_set = set(cart_items)

    if isinstance(rules, pd.DataFrame) and not rules.empty:
        matched = rules[rules["antecedents"].apply(lambda x: set(x).issubset(cart_set))].copy()
        for _, row in matched.iterrows():
            for item in row["consequents"]:
                if item not in cart_set:
                    rows.append({
                        "Recommended Product": item,
                        "Because Cart Has": itemset_to_text(row["antecedents"]),
                        "Confidence": row["confidence"],
                        "Lift": row["lift"],
                        "Rule Score": row["Rule Score"],
                        "Retail Use": row["Retail Use"],
                        "Source": "Association Rule"
                    })

    recs = pd.DataFrame(rows)
    if not fallback_recs.empty:
        fallback = fallback_recs.copy()
        fallback["Source"] = "Co-occurrence Mining"
        common_cols = ["Recommended Product", "Because Cart Has", "Confidence", "Lift", "Rule Score", "Retail Use", "Source"]
        recs = pd.concat([recs, fallback[common_cols]], ignore_index=True)

    if recs.empty:
        return recs

    recs = recs.drop_duplicates("Recommended Product", keep="first")
    recs = recs[~recs["Recommended Product"].isin(cart_items)].copy()

    sales_lookup = sales.set_index("product_name")
    recs["Estimated Profit"] = recs["Recommended Product"].map(sales_lookup["Estimated Profit"]).fillna(0)
    recs["Reorder Rate"] = recs["Recommended Product"].map(sales_lookup["Reorder Rate"]).fillna(0)
    recs["Product Class"] = recs["Recommended Product"].map(sales_lookup["Product Class"]).fillna("Unknown")
    recs["Department"] = recs["Recommended Product"].map(sales_lookup["department"] if "department" in sales_lookup.columns else pd.Series(dtype=str)).fillna("Unknown")

    # Recommendation modes stop every answer from being only Banana.
    if mode == "Profit Focused":
        max_profit = max(recs["Estimated Profit"].max(), 1)
        recs["Final Score"] = recs["Rule Score"].fillna(0) * 0.55 + (recs["Estimated Profit"] / max_profit) * 0.45
    elif mode == "Reorder Focused":
        recs["Final Score"] = recs["Rule Score"].fillna(0) * 0.60 + recs["Reorder Rate"].fillna(0) * 0.40
    elif mode == "Discovery Mode":
        # Lower weight for ultra common staples so different products show up.
        popularity = sales_lookup["Order Share"] if "Order Share" in sales_lookup.columns else pd.Series(dtype=float)
        recs["Popularity"] = recs["Recommended Product"].map(popularity).fillna(0)
        recs["Final Score"] = recs["Rule Score"].fillna(0) * 0.80 - recs["Popularity"] * 0.35
    else:
        recs["Final Score"] = recs["Rule Score"].fillna(0)

    if avoid_banana:
        recs = no_banana_filter(recs, "Recommended Product")

    # Diversify departments so the list looks like a real retail recommendation shelf.
    if mode == "Diversified" and "Department" in recs.columns:
        chosen = []
        used_departments = set()
        for _, row in recs.sort_values("Final Score", ascending=False).iterrows():
            dept = row.get("Department", "Unknown")
            if dept not in used_departments or len(chosen) < 3:
                chosen.append(row)
                used_departments.add(dept)
            if len(chosen) >= top_k:
                break
        recs = pd.DataFrame(chosen) if chosen else recs

    return recs.sort_values("Final Score", ascending=False).head(top_k)


@st.cache_data(show_spinner=False)
def run_product_clustering(sales):
    if sales.empty or len(sales) < 4:
        return pd.DataFrame()
    features = ["Units", "Orders", "Reorder Rate", "Order Share", "Estimated Profit", "Restock Priority Score", "Markdown Risk Score"]
    available = [c for c in features if c in sales.columns]
    work = sales[["product_name"] + available].copy()
    work[available] = work[available].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(work[available])
    k = min(4, len(work))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    work["Cluster"] = km.fit_predict(X)

    # Label clusters by behavior
    summary = work.groupby("Cluster")[available].mean().reset_index()
    labels = {}
    for _, row in summary.iterrows():
        c = int(row["Cluster"])
        if row.get("Units", 0) == summary["Units"].max():
            labels[c] = "High Demand Staples"
        elif row.get("Markdown Risk Score", 0) == summary["Markdown Risk Score"].max():
            labels[c] = "Markdown Candidates"
        elif row.get("Reorder Rate", 0) == summary["Reorder Rate"].max():
            labels[c] = "Loyalty / Reorder Items"
        else:
            labels[c] = "Normal Assortment"
    work["Cluster Label"] = work["Cluster"].map(labels)
    return work.merge(sales, on="product_name", how="left", suffixes=("", "_original"))


@st.cache_data(show_spinner=False)
def run_reorder_classification(row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift):
    df, _, _, _, _, _, _, _, _ = prepare_base_data(
        row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift
    )

    if "reordered" not in df.columns:
        return pd.DataFrame(), [], pd.DataFrame(), {}, {}, pd.DataFrame(), 0, 0

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
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=90, max_depth=12, random_state=42, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    rows = []
    reports = {}
    matrices = {}
    importances = pd.DataFrame()

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

        if name == "Random Forest" and hasattr(model, "feature_importances_"):
            importances = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)

    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    return model_df, features, results, reports, matrices, importances, len(X_train), len(X_test)

# ---------------- LOAD DATA ----------------
with st.spinner("Loading and preparing local retail data..."):
    df, cleaning_report, sales, basket_size, basket_dist, basket, departments, aisles, timing = prepare_base_data(
        row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift
    )

stock_up = sales[sales["Product Class"] == "A - Core seller"].sort_values("Restock Priority Score", ascending=False).head(35)
sale_candidates = sales[sales["Product Class"] == "C - Slow seller"].sort_values("Markdown Risk Score", ascending=False).head(35)
high_sellers = sales.head(20)
clustered_products = run_product_clustering(sales)

# ---------------- KPI BAR ----------------
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    kpi("Orders", f"{df['order_id'].nunique():,}", "cleaned baskets")
with m2:
    kpi("Products", f"{df['product_name'].nunique():,}", "unique products")
with m3:
    kpi("Avg Basket", f"{basket_size['Basket Size'].mean():.2f}", "items per order")
with m4:
    reorder_rate = df["reordered"].mean() if "reordered" in df.columns else 0
    kpi("Reorder Rate", pct(reorder_rate), "repeat demand")
with m5:
    kpi("Rules Ready", f"{len(st.session_state.rules):,}", "after mining")

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    st.subheader("Executive Dashboard")

    core_count = int((sales["Product Class"] == "A - Core seller").sum())
    slow_count = int((sales["Product Class"] == "C - Slow seller").sum())
    avg_basket = basket_size["Basket Size"].mean()
    health_score = 0
    health_score += min(avg_basket / 5, 1) * 30
    health_score += min(reorder_rate / 0.70, 1) * 30 if reorder_rate > 0 else 10
    health_score += (1 - min(slow_count / max(len(sales), 1), 0.50)) * 20
    health_score += min(len(st.session_state.rules) / 20, 1) * 20
    health_score = int(round(health_score))

    left, right = st.columns([0.85, 2.15])
    with left:
        st.markdown(
            f"""
            <div class="panel blue">
                <div class="card-title">Business Health Score</div>
                <div class="big-score">{health_score}</div>
                <div class="score-caption">Based on basket size, reorder behavior, slow sellers, and rules discovered.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        a, b, c = st.columns(3)
        with a:
            if not stock_up.empty:
                item = stock_up.iloc[0]
                card("✅ Stock Up", f"Keep **{item['product_name']}** available. It has a restock priority score of **{item['Restock Priority Score']}**.", "green")
        with b:
            if not sale_candidates.empty:
                item = sale_candidates.iloc[0]
                card("🏷️ Sale Candidate", f"Test a promotion for **{item['product_name']}**. Markdown risk score: **{item['Markdown Risk Score']}**.", "orange")
        with c:
            if st.session_state.rules.empty:
                card("🧺 Build Bundles", "Open **Mining Lab** and run basket mining to find product-pair recommendations.", "purple")
            else:
                rule = st.session_state.rules.iloc[0]
                card("🧺 Best Bundle", f"If cart has **{rule['Customer Buys']}**, recommend **{rule['Recommend']}**. Lift: **{rule['lift']:.2f}**.", "purple")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        alert_box("Data mining method 1", "Association rule mining finds products customers buy together for checkout recommendations.", "Apriori / FP-Growth")
    with c2:
        alert_box("Data mining method 2", "Product clustering groups items into demand patterns like high-demand staples, reorder items, and markdown candidates.", "K-Means")
    with c3:
        alert_box("Data mining method 3", "Classification predicts whether a product is likely to be reordered using Decision Tree, Random Forest, Naive Bayes, and KNN.", "Classification")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Top Product Demand")
        st.bar_chart(high_sellers.set_index("product_name")["Units"])
    with right:
        st.subheader("Basket Size Pattern")
        st.bar_chart(basket_dist.set_index("Basket Size")["Number of Orders"])

# ============================================================
# SMART CART
# ============================================================
elif page == "Smart Cart":
    st.subheader("Smart Cart Recommendation Engine")
    st.caption("This fixes the issue where the app keeps showing only Banana. You can choose recommendation style, remove common staples, start a new cart, and add/remove products like a real shopping cart.")

    top_cart_items = sales.head(120)["product_name"].tolist()
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        add_item = st.selectbox("Scan / add product to cart", [""] + top_cart_items)
    with c2:
        if st.button("➕ Add Item", type="primary") and add_item:
            if add_item not in st.session_state.cart:
                st.session_state.cart.append(add_item)
            st.rerun()
    with c3:
        if st.button("🧹 Start New Cart"):
            st.session_state.cart = []
            st.rerun()
    with c4:
        if st.button("⚡ Add Demo Cart"):
            st.session_state.cart = [x for x in ["Organic Strawberries", "Organic Hass Avocado", "Organic Baby Spinach"] if x in top_cart_items]
            if len(st.session_state.cart) == 0:
                st.session_state.cart = top_cart_items[:3]
            st.rerun()

    if st.session_state.cart:
        st.markdown("**Current Cart**")
        st.markdown(" ".join([f'<span class="cart-pill">{item}</span>' for item in st.session_state.cart]), unsafe_allow_html=True)
        remove_item = st.selectbox("Remove product", [""] + st.session_state.cart)
        if st.button("Remove Selected") and remove_item:
            st.session_state.cart = [x for x in st.session_state.cart if x != remove_item]
            st.rerun()
    else:
        card("Empty Cart", "Add or scan products above. Then the app will recommend what to add next based on basket mining and co-occurrence patterns.", "blue")

    st.markdown("---")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        rec_mode = st.selectbox("Recommendation style", ["Diversified", "Profit Focused", "Reorder Focused", "Discovery Mode"], index=0)
    with rc2:
        avoid_banana = st.checkbox("Avoid common Banana-heavy suggestions", value=True)
    with rc3:
        top_k = st.slider("Number of recommendations", 3, 12, 6)

    if st.button("Run Smart Recommendation", type="primary"):
        fallback = build_cooccurrence_recs(
            row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift, st.session_state.cart
        )
        recs = recommend_from_cart(st.session_state.rules, fallback, st.session_state.cart, sales, rec_mode, avoid_banana, top_k)
        st.session_state.latest_cart_recs = recs

    if "latest_cart_recs" in st.session_state and not st.session_state.latest_cart_recs.empty:
        recs = st.session_state.latest_cart_recs
        best = recs.iloc[0]
        card(
            "Best Next Product",
            f"Recommend **{best['Recommended Product']}** because the cart has **{best['Because Cart Has']}**. Source: **{best['Source']}**. Lift: **{best['Lift']:.2f}**. Confidence: **{best['Confidence']:.2f}**.",
            "purple"
        )
        st.dataframe(recs[["Recommended Product", "Because Cart Has", "Source", "Confidence", "Lift", "Final Score", "Estimated Profit", "Reorder Rate", "Product Class", "Retail Use"]], width="stretch")
    elif st.session_state.cart:
        st.info("Click Run Smart Recommendation. If rules are not ready, the app still uses local co-occurrence mining as a fallback.")

# ============================================================
# PRODUCT SCANNER
# ============================================================
elif page == "Product Scanner":
    st.subheader("Product Scanner / Item Lookup")
    st.caption("No real scanner API is used. This simulates scan-to-order using local product_id or product name from products.csv.")

    products_file = pd.read_csv("products.csv")
    s1, s2, s3 = st.columns([1, 1, 1])
    with s1:
        scan_code = st.text_input("Scan product_id / type ID", placeholder="Example: 24852")
    with s2:
        scan_name = st.selectbox("Or search product name", [""] + sales["product_name"].tolist())
    with s3:
        scan_button = st.button("🔎 Scan / Lookup", type="primary")

    selected_product = None
    if scan_button:
        if scan_code.strip().isdigit():
            pid = int(scan_code.strip())
            found = products_file[products_file["product_id"] == pid]
            if not found.empty:
                selected_product = found.iloc[0]["product_name"]
        elif scan_name:
            selected_product = scan_name
        st.session_state.last_scanned_product = selected_product

    if scan_name and not scan_button:
        st.session_state.last_scanned_product = scan_name

    selected_product = st.session_state.last_scanned_product

    if not selected_product:
        card("Ready to Scan", "Enter a product_id or pick a product name. The app will show item details, suggested action, and similar products.", "blue")
    elif selected_product not in sales["product_name"].values:
        st.warning("Product exists in products.csv, but it is not in the selected transaction sample. Increase rows or use prior-order file.")
    else:
        selected = sales[sales["product_name"] == selected_product].iloc[0]
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            kpi("Units", f"{int(selected['Units']):,}", "in selected data")
        with p2:
            kpi("Orders", f"{int(selected['Orders']):,}", "cart appearances")
        with p3:
            kpi("Reorder Rate", pct(selected["Reorder Rate"]), "repeat buying")
        with p4:
            kpi("Class", selected["Product Class"].split(" - ")[0], selected["Retail Action"])

        border = "green" if selected["Product Class"].startswith("A") else "orange" if selected["Product Class"].startswith("C") else "blue"
        card(
            "Scan Result",
            f"**{selected_product}** should be handled as **{selected['Product Class']}**. Suggested action: **{selected['Retail Action']}**. Restock score: **{selected['Restock Priority Score']}**. Markdown risk: **{selected['Markdown Risk Score']}**.",
            border
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Item Business Scores")
            score_df = pd.DataFrame({"Score": ["Restock Priority", "Markdown Risk", "Reorder Rate"], "Value": [selected["Restock Priority Score"], selected["Markdown Risk Score"], selected["Reorder Rate"] * 100]})
            st.bar_chart(score_df.set_index("Score")["Value"])
        with c2:
            st.subheader("Similar Items")
            similar = sales.copy()
            if "department" in sales.columns and pd.notna(selected.get("department", np.nan)):
                similar = similar[similar["department"] == selected["department"]]
            elif "aisle" in sales.columns and pd.notna(selected.get("aisle", np.nan)):
                similar = similar[similar["aisle"] == selected["aisle"]]
            similar = similar[similar["product_name"] != selected_product].head(15)
            st.dataframe(similar[["product_name", "Units", "Reorder Rate", "Product Class", "Retail Action"]], width="stretch")

        if st.button("Add Scanned Item to Cart"):
            if selected_product not in st.session_state.cart:
                st.session_state.cart.append(selected_product)
            st.success("Added to Smart Cart.")

# ============================================================
# MINING LAB
# ============================================================
elif page == "Mining Lab":
    st.subheader("Basket Mining Lab")
    st.caption("Run FP-Growth or Apriori to find frequent itemsets and association rules. This is the main data mining proof.")

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        run_button = st.button("Run Basket Mining", type="primary")
    with b2:
        clear_button = st.button("Clear Rules")
    with b3:
        st.write(f"Current settings: **{algorithm}**, support **{min_support}**, confidence **{min_confidence}**, lift **{min_lift}**, top products **{top_n_products}**")

    if clear_button:
        st.session_state.frequent = pd.DataFrame()
        st.session_state.rules = pd.DataFrame()
        st.session_state.mining_done = False
        st.rerun()

    if run_button:
        with st.spinner("Mining product combinations from local CSV data..."):
            frequent, rules = run_market_basket(
                row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift,
                algorithm, min_support, min_confidence, min_lift, max_len
            )
            st.session_state.frequent = frequent
            st.session_state.rules = rules
            st.session_state.mining_done = True

    frequent = st.session_state.frequent
    rules = st.session_state.rules

    if not st.session_state.mining_done:
        card("Ready to Mine", "Click **Run Basket Mining**. After that, Smart Cart will use association rules instead of only co-occurrence fallback.", "blue")
    else:
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            kpi("Frequent Itemsets", f"{len(frequent):,}", "patterns")
        with r2:
            kpi("Association Rules", f"{len(rules):,}", "recommendations")
        with r3:
            avg_lift = rules["lift"].mean() if not rules.empty else 0
            kpi("Avg Lift", f"{avg_lift:.2f}", "rule strength")
        with r4:
            kpi("Basket Shape", f"{basket.shape[0]:,} × {basket.shape[1]:,}", "orders × products")

        if rules.empty:
            st.warning("No rules found. Lower support/lift, lower confidence, or increase rows/top products.")
        else:
            best = rules.iloc[0]
            card("Best Association Rule", f"If a customer buys **{best['Customer Buys']}**, recommend **{best['Recommend']}**. Lift **{best['lift']:.2f}**, confidence **{best['confidence']:.2f}**.", "green")
            st.subheader("Rule Strength Map")
            st.scatter_chart(rules.head(120), x="confidence", y="lift", size="support")

            c1, c2 = st.columns(2)
            with c1:
                with st.expander("Open Association Rule Table"):
                    st.dataframe(rules[["Rule", "Customer Buys", "Recommend", "support", "confidence", "lift", "Rule Score", "Retail Use"]].head(150), width="stretch")
            with c2:
                with st.expander("Open Frequent Itemsets"):
                    st.dataframe(frequent[["Itemset", "Item Count", "support"]].head(150), width="stretch")

# ============================================================
# ADVANCED ANALYTICS
# ============================================================
elif page == "Advanced Analytics":
    st.subheader("Advanced Analytics + More Data Mining")
    st.caption("This page makes the project look bigger: ABC analysis, product clustering, campaign simulation, timing analysis, and big-data style processing.")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        kpi("A Products", f"{int((sales['Product Class'] == 'A - Core seller').sum()):,}", "core sellers")
    with a2:
        kpi("B Products", f"{int((sales['Product Class'] == 'B - Normal seller').sum()):,}", "normal sellers")
    with a3:
        kpi("C Products", f"{int((sales['Product Class'] == 'C - Slow seller').sum()):,}", "slow sellers")
    with a4:
        top20_share = safe_div(sales.head(20)["Units"].sum(), sales["Units"].sum())
        kpi("Top 20 Share", pct(top20_share), "demand concentration")

    tab1, tab2, tab3, tab4 = st.tabs(["ABC / Pareto", "Product Clusters", "Campaign Simulator", "Timing + Big Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ABC Product Strategy")
            st.dataframe(sales[["product_name", "Units", "Orders", "Reorder Rate", "Product Class", "Retail Action", "Restock Priority Score", "Markdown Risk Score"]].head(100), width="stretch")
        with c2:
            st.subheader("Pareto Curve")
            pareto = sales[["product_name", "Cumulative Sales Share"]].head(100).copy()
            pareto["Product Rank"] = np.arange(1, len(pareto) + 1)
            st.line_chart(pareto.set_index("Product Rank")["Cumulative Sales Share"])

    with tab2:
        if clustered_products.empty:
            st.warning("Not enough product data for clustering.")
        else:
            st.subheader("K-Means Product Clustering")
            cluster_summary = clustered_products.groupby("Cluster Label").agg(Products=("product_name", "count"), Avg_Units=("Units", "mean"), Avg_Reorder=("Reorder Rate", "mean"), Avg_Markdown_Risk=("Markdown Risk Score", "mean")).reset_index()
            st.dataframe(cluster_summary, width="stretch")
            st.bar_chart(cluster_summary.set_index("Cluster Label")["Products"])
            with st.expander("Open clustered product table"):
                st.dataframe(clustered_products[["product_name", "Cluster Label", "Units", "Reorder Rate", "Estimated Profit", "Restock Priority Score", "Markdown Risk Score", "Retail Action"]].head(150), width="stretch")

    with tab3:
        campaign_type = st.radio("Campaign type", ["Move slow sellers", "Protect best sellers", "Increase repeat purchases"], horizontal=True)
        if campaign_type == "Move slow sellers":
            campaign_base = sale_candidates.copy()
            goal = "discount or bundle slow movers"
        elif campaign_type == "Protect best sellers":
            campaign_base = stock_up.copy()
            goal = "protect core products"
        else:
            campaign_base = sales.sort_values("Reorder Rate", ascending=False).head(35).copy()
            goal = "target repeat buyers"

        campaign_base["Expected Extra Revenue"] = campaign_base["Estimated Revenue"] * expected_uplift
        campaign_base["Discount Cost"] = campaign_base["Expected Extra Revenue"] * promo_discount
        campaign_base["Expected Extra Profit"] = campaign_base["Expected Extra Revenue"] * profit_margin - campaign_base["Discount Cost"]
        campaign_base["Budget Share"] = campaign_base["Expected Extra Revenue"] / max(campaign_base["Expected Extra Revenue"].sum(), 1)
        campaign_base["Suggested Spend"] = campaign_base["Budget Share"] * campaign_budget
        campaign_base["Net Campaign Profit"] = campaign_base["Expected Extra Profit"] - campaign_base["Suggested Spend"]
        campaign_base = campaign_base.sort_values("Net Campaign Profit", ascending=False)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Campaign", campaign_type, goal)
        with c2:
            kpi("Revenue Lift", money(campaign_base["Expected Extra Revenue"].sum()), f"with {pct(expected_uplift)} lift")
        with c3:
            kpi("Net Profit", money(campaign_base["Net Campaign Profit"].sum()), "after spend")

        if not campaign_base.empty:
            best = campaign_base.iloc[0]
            card("Best Campaign Target", f"Start with **{best['product_name']}**. Suggested spend: **{money(best['Suggested Spend'])}**. Estimated net profit: **{money(best['Net Campaign Profit'])}**.", "purple")
            st.bar_chart(campaign_base.head(15).set_index("product_name")["Net Campaign Profit"])
            st.dataframe(campaign_base[["product_name", "Units", "Reorder Rate", "Expected Extra Revenue", "Discount Cost", "Suggested Spend", "Net Campaign Profit", "Retail Action"]].head(35), width="stretch")
            st.session_state.latest_campaign_plan = campaign_base

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Timing Analysis")
            if timing.empty:
                st.info("Add orders.csv to unlock day/hour timing analysis.")
            else:
                best_time = timing.iloc[0]
                card("Busiest Shopping Window", f"Day code **{int(best_time['order_dow'])}**, hour **{int(best_time['order_hour_of_day'])}:00** has the highest activity in this sample.", "blue")
                heat = timing.pivot_table(index="order_hour_of_day", columns="order_dow", values="Orders", fill_value=0)
                st.dataframe(heat, width="stretch")
        with c2:
            st.subheader("Big Data Style Pipeline")
            pipeline = pd.DataFrame({
                "Step": ["1. Load", "2. Clean", "3. Transform", "4. Mine", "5. Model", "6. Recommend"],
                "What happens": [
                    "Read only selected rows for fast demo or more rows for deeper analysis",
                    "Remove missing products and duplicate order/product rows",
                    "Convert transactions into basket matrix and product strategy features",
                    "Run FP-Growth or Apriori to discover frequent product patterns",
                    "Train classifiers to predict reorder behavior",
                    "Turn results into stock, sale, bundle, and cart actions"
                ]
            })
            st.dataframe(pipeline, width="stretch")
            card("Big Data Explanation", f"The app reads up to **{row_limit:,} rows**, caches processing, builds an order-product basket with shape **{basket.shape[0]:,} × {basket.shape[1]:,}**, and only runs heavy mining/modeling when clicked.", "black")

# ============================================================
# MODEL LAB
# ============================================================
elif page == "Model Lab":
    st.subheader("Reorder Prediction Model Lab")
    st.caption("Classification model predicts reordered = 0 or 1. It runs Decision Tree, Random Forest, Naive Bayes, and KNN.")

    if "reordered" not in df.columns:
        st.warning("The reordered column was not found. Use order_products__train.csv or order_products__prior.csv with reordered column.")
    else:
        if st.button("Run Classification Models", type="primary"):
            with st.spinner("Training models locally..."):
                model_df, features, results, reports, matrices, importances, train_size, test_size = run_reorder_classification(
                    row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift
                )
                st.session_state.class_model_df = model_df
                st.session_state.class_features = features
                st.session_state.class_results = results
                st.session_state.class_reports = reports
                st.session_state.class_matrices = matrices
                st.session_state.class_importances = importances
                st.session_state.class_train_size = train_size
                st.session_state.class_test_size = test_size

        if "class_results" not in st.session_state:
            card("Model Not Run Yet", "Click the button to train four local machine learning models and compare their accuracy, precision, recall, and F1 score.", "blue")
        else:
            results = st.session_state.class_results
            best = results.iloc[0]
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                kpi("Best Model", best["Model"], "highest F1")
            with p2:
                kpi("Accuracy", f"{best['Accuracy']:.3f}", "test data")
            with p3:
                kpi("F1 Score", f"{best['F1 Score']:.3f}", "balanced metric")
            with p4:
                kpi(
                    "80/20 Split",
                    f"{st.session_state.class_train_size:,}/{st.session_state.class_test_size:,}",
                    "train/test"
                )

            st.write("Selected features:", st.session_state.class_features)
            st.bar_chart(results.set_index("Model")[["Accuracy", "F1 Score"]])

            if not st.session_state.class_importances.empty:
                st.subheader("Random Forest Feature Importance")
                st.bar_chart(st.session_state.class_importances.set_index("Feature")["Importance"])

            with st.expander("Open detailed model results"):
                st.dataframe(results, width="stretch")
                chosen_model = st.selectbox("Choose model", results["Model"].tolist())
                matrix = pd.DataFrame(st.session_state.class_matrices[chosen_model], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
                st.subheader("Confusion Matrix")
                st.dataframe(matrix, width="stretch")
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(st.session_state.class_reports[chosen_model]).transpose(), width="stretch")

# ============================================================
# EVIDENCE
# ============================================================
elif page == "Evidence":
    st.subheader("Evidence, Testing, and Downloads")

    e1, e2 = st.columns(2)
    with e1:
        with st.expander("Data cleaning proof", expanded=True):
            st.dataframe(cleaning_report, width="stretch")
            st.dataframe(df.head(30), width="stretch")

        with st.expander("Basket format proof"):
            st.caption("Rows are orders. Columns are products. True means product appeared in that order.")
            st.write("Basket shape:", basket.shape)
            st.dataframe(basket.head(8), width="stretch")

    with e2:
        st.subheader("Support Test")
        st.caption("This compares support values to show experiment/evaluation work.")
        if st.button("Run Support Test", type="primary"):
            rows = []
            for support_value in [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]:
                temp_frequent, temp_rules = run_market_basket(
                    row_limit, use_prior, use_order_time, avg_price, profit_margin, top_n_products, promo_discount, expected_uplift,
                    algorithm, support_value, min_confidence, min_lift, max_len
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
        if "latest_campaign_plan" in st.session_state:
            download_button(st.session_state.latest_campaign_plan, "Download Campaign Plan", "campaign_plan.csv")

st.caption("All results are generated locally from CSV files. No API is used. Model Lab uses an 80/20 training/testing split. Estimated profit uses the price and margin assumptions from Dashboard Controls because the dataset does not include real prices or product costs.")
