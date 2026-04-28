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
# Fast local CSV data mining app
# No API. No live internet. Uses only local Instacart CSV files.
# ============================================================

st.set_page_config(
    page_title="Retail Growth Advisor",
    page_icon="🛒",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.main-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
    padding: 30px 34px;
    border-radius: 24px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.22);
}
.main-hero h1 {
    font-size: 42px;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.main-hero p {
    font-size: 17px;
    margin: 0;
    color: #cbd5e1;
}
.action-box {
    background: white;
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
    min-height: 130px;
}
.action-box h3 {
    margin-top: 0;
    color: #0f172a;
    font-size: 20px;
}
.action-box p {
    color: #475569;
    font-size: 15px;
}
.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
}
.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .04em;
}
.metric-number {
    color: #0f172a;
    font-size: 30px;
    font-weight: 800;
    margin-top: 4px;
}
.small-note {
    color: #64748b;
    font-size: 14px;
}
.good-border {border-left: 7px solid #16a34a;}
.warn-border {border-left: 7px solid #f59e0b;}
.bad-border {border-left: 7px solid #dc2626;}
.info-border {border-left: 7px solid #2563eb;}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #f8fafc;
    border-radius: 12px;
    padding: 8px 14px;
    border: 1px solid #e2e8f0;
}
.stTabs [aria-selected="true"] {
    background-color: #e0f2fe !important;
    color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FILE CHECK ----------------
required_files = ["order_products__train.csv", "products.csv"]
missing_files = [f for f in required_files if not Path(f).exists()]

if missing_files:
    st.error("Missing required file(s): " + ", ".join(missing_files))
    st.info("Put this app.py file in the same folder as order_products__train.csv and products.csv.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Control Panel")

speed_mode = st.sidebar.selectbox(
    "Speed mode",
    ["Fast demo", "Balanced", "Deeper analysis"],
    index=0,
    help="Fast demo loads quickly. Deeper analysis uses more rows and may take longer."
)

mode_defaults = {
    "Fast demo": {"rows": 60000, "top": 90, "support": 0.005, "confidence": 0.05},
    "Balanced": {"rows": 120000, "top": 130, "support": 0.003, "confidence": 0.05},
    "Deeper analysis": {"rows": 220000, "top": 180, "support": 0.002, "confidence": 0.05},
}

st.sidebar.subheader("Data Size")
row_limit = st.sidebar.slider(
    "Transaction rows to read",
    min_value=20000,
    max_value=300000,
    value=mode_defaults[speed_mode]["rows"],
    step=20000
)

use_prior = st.sidebar.checkbox(
    "Add prior-order file if available",
    value=False,
    help="This can improve data size but makes loading slower."
)

use_order_time = st.sidebar.checkbox(
    "Use order time features if orders.csv exists",
    value=True,
    help="Adds order day, hour, and days since prior order for classification."
)

st.sidebar.subheader("Basket Mining")
top_n_products = st.sidebar.slider(
    "Popular products used for basket rules",
    min_value=40,
    max_value=250,
    value=mode_defaults[speed_mode]["top"],
    step=10
)
algorithm = st.sidebar.selectbox("Mining method", ["FP-Growth", "Apriori"], index=0)
min_support = st.sidebar.select_slider(
    "Minimum support",
    options=[0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02],
    value=mode_defaults[speed_mode]["support"]
)
min_confidence = st.sidebar.select_slider(
    "Minimum confidence",
    options=[0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    value=mode_defaults[speed_mode]["confidence"]
)
min_lift = st.sidebar.select_slider(
    "Minimum lift",
    options=[1.0, 1.1, 1.2, 1.5, 2.0],
    value=1.1
)
max_len = st.sidebar.selectbox("Maximum products in one pattern", [2, 3], index=1)

st.sidebar.subheader("Profit Simulator")
avg_price = st.sidebar.slider("Estimated average item price", 1.0, 25.0, 5.0, 0.5)
profit_margin = st.sidebar.slider("Estimated profit margin", 0.10, 0.70, 0.30, 0.05)

# ---------------- FUNCTIONS ----------------
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
            order_ids = df["order_id"].unique()
            orders = orders[orders["order_id"].isin(order_ids)]
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

    def business_move(row):
        if row["lift"] >= 2 and row["confidence"] >= 0.15:
            return "Best for bundle or checkout recommendation"
        if row["lift"] >= 1.5:
            return "Good for shelf placement or coupon test"
        return "Use carefully as a test rule"

    rules["Retail Use"] = rules.apply(business_move, axis=1)
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
        "add_to_cart_order",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
        "basket_size",
        "product_order_count",
        "product_reorder_rate",
        "aisle_id",
        "department_id"
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

# ---------------- LOAD BASE DATA ONLY ----------------
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

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-hero">
    <h1>Retail Growth Advisor</h1>
    <p>Fast data mining dashboard that tells a retailer what to stock, what to discount, what to bundle, and what customers may reorder.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metric_values = [
    (m1, "Orders Analyzed", f"{df['order_id'].nunique():,}", "good-border"),
    (m2, "Products Found", f"{df['product_name'].nunique():,}", "info-border"),
    (m3, "Avg Basket Size", f"{basket_size['Basket Size'].mean():.2f}", "warn-border"),
    (m4, "Rules Ready", f"{len(st.session_state.rules):,}", "good-border" if st.session_state.mining_done else "bad-border"),
]

for col, label, value, border in metric_values:
    with col:
        st.markdown(f"""
        <div class="metric-card {border}">
            <div class="metric-label">{label}</div>
            <div class="metric-number">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- TABS ----------------
tab_home, tab_clean, tab_products, tab_sections, tab_basket, tab_cart, tab_predict, tab_test, tab_export = st.tabs([
    "Home",
    "Clean Data",
    "Product Demand",
    "Store Sections",
    "Basket Rules",
    "Cart Advisor",
    "Reorder Prediction",
    "Test Results",
    "Downloads"
])

with tab_home:
    st.subheader("Store Action Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        best_stock = stock_up.iloc[0] if not stock_up.empty else None
        st.markdown("<div class='action-box good-border'>", unsafe_allow_html=True)
        st.markdown("### Stock Up")
        if best_stock is not None:
            st.write(f"**{best_stock['product_name']}**")
            st.write(f"Demand: {int(best_stock['Units']):,} units")
        else:
            st.write("No stock-up item found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        best_sale = sale_candidates.iloc[0] if not sale_candidates.empty else None
        st.markdown("<div class='action-box warn-border'>", unsafe_allow_html=True)
        st.markdown("### Put on Sale")
        if best_sale is not None:
            st.write(f"**{best_sale['product_name']}**")
            st.write(f"Demand: {int(best_sale['Units']):,} units")
        else:
            st.write("No sale item found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='action-box info-border'>", unsafe_allow_html=True)
        st.markdown("### Next Step")
        st.write("Open **Basket Rules** and press **Run Basket Mining** to find product bundles and checkout recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Top Products by Demand")
    st.bar_chart(high_sellers.set_index("product_name")["Units"])

    st.subheader("Basket Size Pattern")
    st.bar_chart(basket_dist.set_index("Basket Size")["Number of Orders"])

with tab_clean:
    st.subheader("Data Cleaning Results")
    st.dataframe(cleaning_report, use_container_width=True)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(35), use_container_width=True)

    st.subheader("Basket Format Preview")
    st.caption("Rows are orders. Columns are products. True means the customer bought that product.")
    st.dataframe(basket.head(8), use_container_width=True)
    st.write("Basket shape:", basket.shape)

with tab_products:
    st.subheader("High-Demand Products")
    st.caption("These are the best products to keep in stock and display clearly.")
    st.dataframe(high_sellers, use_container_width=True)

    st.subheader("Low-Demand Products")
    st.caption("These are candidates for discounts, bundles, or lower reorder quantity.")
    st.dataframe(low_sellers, use_container_width=True)

    st.subheader("Product Class Summary")
    abc_summary = sales["Product Class"].value_counts().reset_index()
    abc_summary.columns = ["Product Class", "Number of Products"]
    st.dataframe(abc_summary, use_container_width=True)

with tab_sections:
    st.subheader("Department Performance")
    if departments.empty:
        st.warning("departments.csv was not found, so this section is skipped.")
    else:
        st.dataframe(departments.head(25), use_container_width=True)
        st.bar_chart(departments.head(15).set_index("department")["Units"])

    st.subheader("Aisle Performance")
    if aisles.empty:
        st.warning("aisles.csv was not found, so this section is skipped.")
    else:
        st.dataframe(aisles.head(25), use_container_width=True)
        st.bar_chart(aisles.head(15).set_index("aisle")["Units"])

with tab_basket:
    st.subheader("Basket Rules")
    st.caption("This is the heaviest step, so it only runs when you press the button.")

    run_button = st.button("Run Basket Mining", type="primary")

    if run_button:
        with st.spinner("Finding frequent itemsets and association rules..."):
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
        st.info("Press **Run Basket Mining** to generate itemsets and rules.")
    else:
        st.subheader("Frequent Itemsets")
        if frequent.empty:
            st.warning("No itemsets found. Lower support or use more rows.")
        else:
            st.dataframe(frequent[["Itemset", "Item Count", "support"]].head(50), use_container_width=True)

        st.subheader("Association Rules")
        if rules.empty:
            st.warning("No rules found. Try support 0.003, confidence 0.05, and lift 1.0.")
        else:
            show_rules = rules[["Rule", "Customer Buys", "Recommend", "support", "confidence", "lift", "Rule Score", "Retail Use"]].head(50)
            st.dataframe(show_rules, use_container_width=True)
            st.bar_chart(show_rules.head(20).set_index("Rule")["lift"])

with tab_cart:
    st.subheader("Cart Advisor")
    st.caption("Pick what a customer has in their cart. The app recommends what to show next.")

    if st.session_state.rules.empty:
        st.warning("Run Basket Mining first in the Basket Rules tab.")
    else:
        cart_items = st.multiselect("Customer cart", sorted(list(basket.columns)))
        recs = recommend_from_cart(st.session_state.rules, cart_items)

        if len(cart_items) == 0:
            st.info("Choose one or more cart products to see recommendations.")
        elif recs.empty:
            st.warning("No recommendation found for this cart. Try another product.")
        else:
            st.dataframe(recs.head(20), use_container_width=True)
            top_rec = recs.iloc[0]
            st.success(
                f"Best recommendation: {top_rec['Recommended Product']} "
                f"with lift {top_rec['Lift']:.2f} and confidence {top_rec['Confidence']:.2f}."
            )

with tab_predict:
    st.subheader("Reorder Prediction")
    st.caption("This runs classification models only when you press the button to avoid slow page loading.")

    if "reordered" not in df.columns:
        st.warning("The reordered column was not found. Use order_products__train.csv or prior data with the reordered column.")
    else:
        if st.button("Run Reorder Prediction Models", type="primary"):
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
            st.info("Press **Run Reorder Prediction Models** to create the training/test set and model comparison.")
        else:
            st.write("Selected features:", st.session_state.class_features)
            st.write("Training set size:", st.session_state.class_train_size)
            st.write("Test set size:", st.session_state.class_test_size)

            class_dist = st.session_state.class_model_df["reordered"].value_counts().reset_index()
            class_dist.columns = ["Class", "Count"]
            class_dist["Meaning"] = class_dist["Class"].map({0: "Not Reordered", 1: "Reordered"})
            st.subheader("Class Distribution")
            st.dataframe(class_dist, use_container_width=True)

            st.subheader("Model Comparison")
            st.dataframe(st.session_state.class_results, use_container_width=True)
            st.bar_chart(st.session_state.class_results.set_index("Model")["Accuracy"])

            best = st.session_state.class_results.iloc[0]
            st.success(f"Best model: {best['Model']} | Accuracy: {best['Accuracy']:.3f} | F1 Score: {best['F1 Score']:.3f}")

            chosen_model = st.selectbox("Show confusion matrix for", st.session_state.class_results["Model"].tolist())
            matrix = pd.DataFrame(
                st.session_state.class_matrices[chosen_model],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            )
            st.dataframe(matrix, use_container_width=True)
            st.dataframe(pd.DataFrame(st.session_state.class_reports[chosen_model]).transpose(), use_container_width=True)

with tab_test:
    st.subheader("Test Results")
    st.caption("Compare settings to show evaluation, not just final output.")

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
        test_df = pd.DataFrame(rows)
        st.session_state.support_test = test_df

    if "support_test" in st.session_state:
        st.dataframe(st.session_state.support_test, use_container_width=True)
        st.line_chart(st.session_state.support_test.set_index("Support")[["Frequent Itemsets", "Association Rules"]])
    else:
        st.info("Press **Run Support Test** to compare mining settings.")

with tab_export:
    st.subheader("Download Result Files")
    download_button(sales, "Download Product Strategy", "product_strategy.csv")
    download_button(stock_up, "Download Stock-Up List", "stock_up_products.csv")
    download_button(sale_candidates, "Download Sale Candidate List", "sale_candidates.csv")
    download_button(cleaning_report, "Download Cleaning Report", "cleaning_report.csv")

    if not st.session_state.rules.empty:
        export_rules = st.session_state.rules[["Rule", "Customer Buys", "Recommend", "support", "confidence", "lift", "Rule Score", "Retail Use"]]
        download_button(export_rules, "Download Basket Rules", "basket_rules.csv")

    if "class_results" in st.session_state:
        download_button(st.session_state.class_results, "Download Classification Results", "classification_results.csv")

st.caption("Profit is estimated because this dataset does not include real item prices or product costs.")
