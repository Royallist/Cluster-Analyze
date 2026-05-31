import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette: #F5F5F5 · #76ABAE · #303841 · #FF5722 ───────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background — paper texture via SVG data-URI ── */
.stApp {
    background-color: #F5F5F5;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3CfeBlend in='SourceGraphic' mode='multiply'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23noise)' opacity='0.045'/%3E%3C/svg%3E"),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 28px,
            rgba(118,171,174,0.045) 28px,
            rgba(118,171,174,0.045) 29px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 28px,
            rgba(118,171,174,0.025) 28px,
            rgba(118,171,174,0.025) 29px
        );
    color: #303841;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(245,245,245,0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(118,171,174,0.35) !important;
}
section[data-testid="stSidebar"] * { color: #303841 !important; }

/* ── Nav drawer overlay (muncul saat sidebar ditutup) ── */
.nav-drawer {
    background: rgba(245,245,245,0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(118,171,174,0.35);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(48,56,65,0.12);
}
.nav-drawer-title {
    font-size: 0.65rem;
    font-weight: 700;
    color: #76ABAE;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 0.8rem;
}

/* ── Toggle button — styled via Streamlit button override ── */
div[data-testid="stButton"].nav-toggle > button {
    background: #303841 !important;
    color: #F5F5F5 !important;
    border: none !important;
    border-radius: 10px !important;
    width: 42px !important;
    height: 42px !important;
    min-height: unset !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 4px 16px rgba(48,56,65,0.25) !important;
    transition: background 0.2s !important;
}
div[data-testid="stButton"].nav-toggle > button:hover {
    background: #76ABAE !important;
}

/* ── Hero card — glassmorphism ── */
.hero-card {
    background: rgba(48, 56, 65, 0.72);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(118,171,174,0.30);
    border-radius: 20px;
    padding: 2.6rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 8px 32px rgba(48,56,65,0.18),
        inset 0 1px 0 rgba(255,255,255,0.12);
}
/* Paper grain overlay on hero */
.hero-card::before {
    content: '';
    position: absolute; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E");
    pointer-events: none;
    border-radius: 20px;
}
/* Teal glow orb top-left */
.hero-card::after {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(118,171,174,0.28) 0%, transparent 70%);
    pointer-events: none;
}
/* Orange accent orb bottom-right */
.hero-accent {
    position: absolute;
    bottom: -50px; right: -30px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,87,34,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #F5F5F5;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem;
    position: relative; z-index: 1;
}
.hero-sub {
    color: rgba(245,245,245,0.60);
    font-size: 0.92rem;
    margin: 0;
    position: relative; z-index: 1;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(118,171,174,0.30);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(48,56,65,0.07);
}
.metric-card .num {
    font-size: 2rem; font-weight: 700; color: #76ABAE;
}
.metric-card .lbl {
    font-size: 0.68rem; color: #76869a;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 3px;
}

/* ── Cluster cards ── */
.cluster-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border-left: 4px solid;
    background: rgba(255,255,255,0.62);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-top: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 2px 10px rgba(48,56,65,0.08);
}
.c0 { border-left-color: #76ABAE; }
.c1 { border-left-color: #9b6fd4; }
.c2 { border-left-color: #3daa6a; }
.c3 { border-left-color: #FF5722; }
.cluster-name {
    font-size: 0.85rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 0.4rem;
}
.cluster-desc { font-size: 0.87rem; color: #4a5568; line-height: 1.65; }

/* ── Section title ── */
.section-title {
    font-size: 0.68rem; font-weight: 700;
    color: #76ABAE;
    text-transform: uppercase; letter-spacing: 2.5px;
    margin: 1.2rem 0 1rem;
}

/* ── Inputs ── */
.stNumberInput > label, .stSlider > label {
    color: #303841 !important; font-size: 0.82rem !important; font-weight: 500 !important;
}
.stNumberInput input {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(118,171,174,0.4) !important;
    color: #303841 !important; border-radius: 8px !important;
}

/* ── Predict button — accent orange ── */
.stButton > button {
    background: #FF5722 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.6px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(255,87,34,0.30) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #e64a19 !important;
    box-shadow: 0 6px 20px rgba(255,87,34,0.40) !important;
    transform: translateY(-1px);
}

/* ── Divider ── */
hr { border-color: rgba(118,171,174,0.25) !important; margin: 1.5rem 0 !important; }

/* ── Expander ── */
details {
    background: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(118,171,174,0.30) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px) !important;
}

/* ── DataFrame ── */
.stDataFrame {
    border: 1px solid rgba(118,171,174,0.30) !important;
    border-radius: 10px !important;
}

/* ── Hide Streamlit chrome ── */
footer, #MainMenu, header { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────────────────
if "show_nav" not in st.session_state:
    st.session_state.show_nav = False
if "page" not in st.session_state:
    st.session_state.page = "🏠 Dashboard"

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_FEATURES = [
    'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
    'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
]

CLUSTER_INFO = {
    0: {
        "name": "Pengguna Konservatif",
        "emoji": "🛡️",
        "color": "#76ABAE",
        "css": "c0",
        "desc": "Pembelian rendah, hati-hati dalam penggunaan kredit. Cenderung membayar lebih dari tagihan tapi tidak sering berbelanja.",
        "strategy": "Tawarkan produk entry-level, cashback untuk kebutuhan sehari-hari, dan edukasi manfaat penggunaan kredit.",
        "badge_bg": "rgba(118,171,174,0.12)",
    },
    1: {
        "name": "Pengguna Premium",
        "emoji": "💎",
        "color": "#9b6fd4",
        "css": "c1",
        "desc": "Semua nilai tinggi — saldo besar, pembelian besar, pembayaran besar. Pelanggan bernilai tinggi yang aktif.",
        "strategy": "Tawarkan layanan premium, travel rewards, lounge access, dan produk eksklusif. Prioritaskan retention.",
        "badge_bg": "rgba(155,111,212,0.10)",
    },
    2: {
        "name": "Pengguna Aktif Harian",
        "emoji": "🛒",
        "color": "#3daa6a",
        "css": "c2",
        "desc": "Saldo dan nominal transaksi rendah, namun frekuensi transaksi sangat tinggi. Berbelanja sering dalam jumlah kecil.",
        "strategy": "Program loyalty points, cashback untuk minimarket & F&B, serta notifikasi promo harian.",
        "badge_bg": "rgba(61,170,106,0.10)",
    },
    3: {
        "name": "Pengguna Pasif",
        "emoji": "🌱",
        "color": "#FF5722",
        "css": "c3",
        "desc": "Semua nilai rendah — saldo, pembelian, maupun pembayaran. Kemungkinan pengguna baru atau tidak aktif.",
        "strategy": "Program onboarding, edukasi finansial, insentif first-purchase, dan reminder aktivasi kartu.",
        "badge_bg": "rgba(255,87,34,0.10)",
    },
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        km         = joblib.load(os.path.join(model_dir, "km.pkl"))
        pca        = joblib.load(os.path.join(model_dir, "pca_final.pkl"))
        rob_scaler = joblib.load(os.path.join(model_dir, "robust_scaler.pkl"))
        std_scaler = joblib.load(os.path.join(model_dir, "standard_scaler.pkl"))
        return km, pca, rob_scaler, std_scaler, True
    except Exception:
        return None, None, None, None, False

km, pca_model, robust_scaler, standard_scaler, models_loaded = load_models()

def get_scaler_cols():
    if not models_loaded:
        return [], []
    try:
        return list(standard_scaler.feature_names_in_), list(robust_scaler.feature_names_in_)
    except AttributeError:
        return [], ALL_FEATURES

def preprocess(df: pd.DataFrame):
    std_cols, rob_cols = get_scaler_cols()
    data = df[ALL_FEATURES].copy().astype(float)
    if std_cols:
        data[std_cols] = standard_scaler.transform(data[std_cols])
    if rob_cols:
        data[rob_cols] = robust_scaler.transform(data[rob_cols])
    return pca_model.transform(data)

def predict_cluster(df: pd.DataFrame):
    return km.predict(preprocess(df))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:0.68rem;font-weight:700;color:#76ABAE;'
        'text-transform:uppercase;letter-spacing:2.5px;margin-bottom:1.2rem">Navigasi</p>',
        unsafe_allow_html=True
    )
    sidebar_page = st.radio("", ["🏠 Dashboard", "🔍 Prediksi Manual", "📂 Upload CSV", "📖 Panduan Cluster"],
                    index=["🏠 Dashboard","🔍 Prediksi Manual","📂 Upload CSV","📖 Panduan Cluster"].index(st.session_state.page),
                    label_visibility="collapsed",
                    key="sidebar_radio")
    if sidebar_page != st.session_state.page:
        st.session_state.page = sidebar_page
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.68rem;font-weight:700;color:#76ABAE;'
        'text-transform:uppercase;letter-spacing:2.5px;margin-bottom:0.8rem">Status Model</p>',
        unsafe_allow_html=True
    )
    if models_loaded:
        st.success("✅ Model berhasil dimuat")
        std_cols, rob_cols = get_scaler_cols()
        with st.expander("🔎 Detail kolom scaler"):
            st.caption(f"**StandardScaler** ({len(std_cols)} kolom): {', '.join(std_cols) if std_cols else '—'}")
            st.caption(f"**RobustScaler** ({len(rob_cols)} kolom): {', '.join(rob_cols) if rob_cols else '—'}")
    else:
        st.error("❌ File model tidak ditemukan")
        st.code("km.pkl\npca_final.pkl\nrobust_scaler.pkl\nstandard_scaler.pkl")

# ── Top bar: toggle button + breadcrumb ──────────────────────────────────────
PAGE_ICONS = {
    "🏠 Dashboard": "🏠",
    "🔍 Prediksi Manual": "🔍",
    "📂 Upload CSV": "📂",
    "📖 Panduan Cluster": "📖",
}
NAV_PAGES = ["🏠 Dashboard", "🔍 Prediksi Manual", "📂 Upload CSV", "📖 Panduan Cluster"]

top_col1, top_col2 = st.columns([0.04, 0.96])
with top_col1:
    st.markdown('<div class="nav-toggle">', unsafe_allow_html=True)
    if st.button("☰", key="nav_toggle_btn", help="Buka/tutup navigasi"):
        st.session_state.show_nav = not st.session_state.show_nav
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
with top_col2:
    st.markdown(
        f'<p style="margin:0;padding:0.55rem 0 0;font-size:0.78rem;color:#76ABAE;font-weight:500">'
        f'{st.session_state.page}</p>',
        unsafe_allow_html=True
    )

# ── Inline nav drawer (muncul saat tombol ☰ diklik) ──────────────────────────
if st.session_state.show_nav:
    labels = ["🏠 Dashboard", "🔍 Prediksi Manual", "📂 Upload CSV", "📖 Panduan Cluster"]
    st.markdown(
        '<div class="nav-drawer"><div class="nav-drawer-title">☰ &nbsp;Pilih Halaman</div></div>',
        unsafe_allow_html=True
    )
    nav_cols = st.columns(4)
    for col, lbl in zip(nav_cols, labels):
        with col:
            is_active = (st.session_state.page == lbl)
            # Override warna tombol aktif via inline CSS trick
            if is_active:
                st.markdown(
                    f'<style>div[data-testid="stButton"] button[title="{lbl}"] '
                    f'{{ background:#76ABAE !important; color:#fff !important; }}</style>',
                    unsafe_allow_html=True
                )
            if st.button(lbl, key=f"nav_btn_{lbl}", use_container_width=True):
                st.session_state.page = lbl
                st.session_state.show_nav = False
                st.rerun()
    st.markdown("---")

page = st.session_state.page

# ── Helper: hero banner ───────────────────────────────────────────────────────
def hero(icon, title, sub):
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-accent"></div>
        <p class="hero-title">{icon} {title}</p>
        <p class="hero-sub">{sub}</p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    hero("💳", "Credit Card Segmentation",
         "Customer segmentation berbasis K-Means Clustering · Hacktiv8 MSIB-06 · Desvin Sitohang")

    st.markdown('<p class="section-title">Ringkasan Dataset Training</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (num, lbl) in zip([c1,c2,c3,c4], [
        ("4.475","Total Nasabah"),("4","Jumlah Cluster"),
        ("16","Fitur Digunakan"),("95%","PCA Variance")
    ]):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="num">{num}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<p class="section-title">4 Segmen Nasabah</p>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    for i, (cid, info) in enumerate(CLUSTER_INFO.items()):
        with (ca if i % 2 == 0 else cb):
            st.markdown(f"""
            <div class="cluster-card {info['css']}">
                <div class="cluster-name" style="color:{info['color']}">{info['emoji']} Cluster {cid} — {info['name']}</div>
                <div class="cluster-desc">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Alur Pipeline</p>', unsafe_allow_html=True)
    steps = ["BigQuery","CSV","EDA","Outlier Handling","Missing Value","Feature Selection","Scaling","PCA","K-Means","Segmen"]
    cols  = st.columns(len(steps))
    for col, step in zip(cols, steps):
        with col:
            st.markdown(
                f"<div style='text-align:center;font-size:0.62rem;color:#76ABAE;"
                f"background:rgba(255,255,255,0.65);backdrop-filter:blur(6px);"
                f"border:1px solid rgba(118,171,174,0.3);border-radius:8px;"
                f"padding:0.45rem 0.1rem;font-weight:600'>{step}</div>",
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI MANUAL
# ══════════════════════════════════════════════════════════════════════════════
elif "Manual" in page:
    hero("🔍", "Prediksi Manual", "Masukkan data nasabah secara manual untuk mengetahui segmennya")

    if not models_loaded:
        st.error("Model belum dimuat. Pastikan file .pkl tersedia.")
        st.stop()

    st.markdown('<p class="section-title">💰 Informasi Keuangan</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        balance      = st.number_input("Saldo (BALANCE)", min_value=0.0, value=1500.0, step=100.0)
        credit_limit = st.number_input("Batas Kredit (CREDIT_LIMIT)", min_value=0.0, value=5000.0, step=500.0)
        payments     = st.number_input("Total Pembayaran (PAYMENTS)", min_value=0.0, value=1200.0, step=100.0)
    with col2:
        purchases    = st.number_input("Total Pembelian (PURCHASES)", min_value=0.0, value=800.0, step=100.0)
        oneoff       = st.number_input("Pembelian Sekaligus (ONEOFF_PURCHASES)", min_value=0.0, value=200.0, step=50.0)
        installments = st.number_input("Pembelian Cicilan (INSTALLMENTS_PURCHASES)", min_value=0.0, value=600.0, step=50.0)
    with col3:
        cash_advance = st.number_input("Tarik Tunai (CASH_ADVANCE)", min_value=0.0, value=0.0, step=50.0)
        min_payments = st.number_input("Pembayaran Minimum (MINIMUM_PAYMENTS)", min_value=0.0, value=200.0, step=50.0)
        cash_adv_trx = st.number_input("Jumlah Transaksi Tarik Tunai (CASH_ADVANCE_TRX)", min_value=0, value=0, step=1)

    st.markdown("---")
    st.markdown('<p class="section-title">📈 Frekuensi & Transaksi</p>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        bal_freq     = st.slider("Frekuensi Saldo (BALANCE_FREQUENCY)", 0.0, 1.0, 0.9, 0.01)
        purch_freq   = st.slider("Frekuensi Pembelian (PURCHASES_FREQUENCY)", 0.0, 1.0, 0.5, 0.01)
    with col5:
        oneoff_freq  = st.slider("Frekuensi Pembelian Sekaligus (ONEOFF_PURCHASES_FREQUENCY)", 0.0, 1.0, 0.2, 0.01)
        install_freq = st.slider("Frekuensi Pembelian Cicilan (PURCHASES_INSTALLMENTS_FREQUENCY)", 0.0, 1.0, 0.4, 0.01)
    with col6:
        cash_adv_freq = st.slider("Frekuensi Tarik Tunai (CASH_ADVANCE_FREQUENCY)", 0.0, 1.0, 0.0, 0.01)
        prc_full      = st.slider("Persentase Bayar Penuh (PRC_FULL_PAYMENT)", 0.0, 1.0, 0.3, 0.01)

    purch_trx = st.number_input("Jumlah Transaksi Pembelian (PURCHASES_TRX)", min_value=0, value=10, step=1)

    st.markdown("---")
    if st.button("🚀  PREDIKSI SEGMEN"):
        input_data = pd.DataFrame([{
            'BALANCE': balance, 'BALANCE_FREQUENCY': bal_freq,
            'PURCHASES': purchases, 'ONEOFF_PURCHASES': oneoff,
            'INSTALLMENTS_PURCHASES': installments, 'CASH_ADVANCE': cash_advance,
            'PURCHASES_FREQUENCY': purch_freq, 'ONEOFF_PURCHASES_FREQUENCY': oneoff_freq,
            'PURCHASES_INSTALLMENTS_FREQUENCY': install_freq,
            'CASH_ADVANCE_FREQUENCY': cash_adv_freq,
            'CASH_ADVANCE_TRX': float(cash_adv_trx),
            'PURCHASES_TRX': float(purch_trx),
            'CREDIT_LIMIT': credit_limit, 'PAYMENTS': payments,
            'MINIMUM_PAYMENTS': min_payments, 'PRC_FULL_PAYMENT': prc_full,
        }])
        try:
            with st.spinner("Memproses..."):
                label = predict_cluster(input_data)[0]
                info  = CLUSTER_INFO[label]

            st.markdown("---")
            st.markdown(f"""
            <div style="background:{info['badge_bg']};
                        border:1.5px solid {info['color']};
                        backdrop-filter:blur(12px);
                        -webkit-backdrop-filter:blur(12px);
                        border-radius:18px;padding:2.2rem;
                        text-align:center;margin-bottom:1.5rem;
                        box-shadow:0 8px 32px rgba(48,56,65,0.10);">
                <div style="font-size:3rem;margin-bottom:0.5rem">{info['emoji']}</div>
                <div style="font-size:1.7rem;font-weight:700;color:{info['color']}">&nbsp;Cluster {label}</div>
                <div style="font-size:1rem;color:#303841;margin:0.3rem 0 1rem;font-weight:600">{info['name']}</div>
                <div style="font-size:0.9rem;color:#4a5568;max-width:480px;margin:0 auto;line-height:1.75">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.65);backdrop-filter:blur(8px);
                        border:1px solid rgba(118,171,174,0.30);border-radius:12px;
                        padding:1.4rem 1.6rem;box-shadow:0 2px 8px rgba(48,56,65,0.07);">
                <div style="font-size:0.68rem;color:#76ABAE;text-transform:uppercase;
                            letter-spacing:2.5px;font-weight:700;margin-bottom:0.8rem">
                    💡 Rekomendasi Strategi
                </div>
                <div style="font-size:0.92rem;color:#303841;line-height:1.75">{info['strategy']}</div>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Cek sidebar → Detail kolom scaler untuk debug.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD CSV
# ══════════════════════════════════════════════════════════════════════════════
elif "CSV" in page:
    hero("📂", "Prediksi Batch via CSV", "Upload file CSV berisi data nasabah untuk segmentasi massal")

    if not models_loaded:
        st.error("Model belum dimuat.")
        st.stop()

    with st.expander("📋 Format CSV yang diperlukan"):
        st.markdown("16 kolom berikut wajib ada (CUST_ID & TENURE boleh ada, akan diabaikan):")
        st.dataframe(pd.DataFrame([{f: 0.0 for f in ALL_FEATURES}]), use_container_width=True)

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded:
        try:
            df_up   = pd.read_csv(uploaded)
            missing = [c for c in ALL_FEATURES if c not in df_up.columns]
            if missing:
                st.error(f"Kolom tidak ditemukan: {missing}")
                st.stop()

            with st.spinner("Memproses segmentasi..."):
                labels = predict_cluster(df_up)
                df_result = df_up.copy()
                df_result['CLUSTER'] = labels
                df_result['SEGMEN']  = df_result['CLUSTER'].map(
                    lambda x: f"{CLUSTER_INFO[x]['emoji']} {CLUSTER_INFO[x]['name']}")

            st.success(f"✅ Segmentasi selesai! {len(df_result):,} nasabah diproses.")
            st.markdown("---")

            st.markdown('<p class="section-title">📊 Distribusi Segmen</p>', unsafe_allow_html=True)
            counts = df_result['CLUSTER'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('#F5F5F5')
            bar_colors = [CLUSTER_INFO[i]['color'] for i in counts.index]
            bars = ax.bar([f"C{i} {CLUSTER_INFO[i]['emoji']}" for i in counts.index],
                          counts.values, color=bar_colors, width=0.5, edgecolor='none')
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        f'{val:,}', ha='center', va='bottom', color='#303841', fontsize=10)
            ax.set_xlabel('Cluster', color='#76ABAE', fontsize=10)
            ax.set_ylabel('Jumlah Nasabah', color='#76ABAE', fontsize=10)
            ax.tick_params(colors='#303841')
            for spine in ax.spines.values():
                spine.set_edgecolor('rgba(118,171,174,0.3)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown('<p class="section-title">📋 Hasil (50 baris pertama)</p>', unsafe_allow_html=True)
            st.dataframe(df_result[['CLUSTER','SEGMEN']+ALL_FEATURES[:6]].head(50), use_container_width=True)

            st.download_button("⬇ Download Hasil Lengkap (CSV)",
                               data=df_result.to_csv(index=False).encode('utf-8'),
                               file_name="hasil_segmentasi.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PANDUAN CLUSTER
# ══════════════════════════════════════════════════════════════════════════════
elif "Panduan" in page:
    hero("📖", "Panduan Segmen", "Penjelasan karakteristik dan rekomendasi strategi tiap cluster")

    for cid, info in CLUSTER_INFO.items():
        st.markdown(f"""
        <div class="cluster-card {info['css']}" style="padding:1.8rem 2rem;margin-bottom:1.5rem">
            <div class="cluster-name" style="color:{info['color']};font-size:1rem">
                {info['emoji']} Cluster {cid} — {info['name']}
            </div>
            <div class="cluster-desc" style="margin-top:0.5rem">{info['desc']}</div>
            <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(118,171,174,0.2)">
                <span style="font-size:0.68rem;color:{info['color']};text-transform:uppercase;
                             letter-spacing:2px;font-weight:700">💡 Strategi</span><br/>
                <span style="font-size:0.9rem;color:#303841">{info['strategy']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🔬 Tentang Model</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,255,255,0.65);backdrop-filter:blur(8px);
                border:1px solid rgba(118,171,174,0.30);border-radius:14px;
                padding:1.6rem 2rem;line-height:2;font-size:0.9rem;color:#4a5568">
        <b style="color:#303841">Algoritma:</b> K-Means Clustering (k=4, init=k-means++)<br>
        <b style="color:#303841">Preprocessing:</b> Winsorizer IQR Capping → Drop missing value →
            Feature selection (drop TENURE) → StandardScaler + RobustScaler<br>
        <b style="color:#303841">Dimensionality Reduction:</b> PCA (95% explained variance)<br>
        <b style="color:#303841">Evaluasi:</b> Elbow Method + Silhouette Score<br>
        <b style="color:#303841">Dataset:</b> 4.475 nasabah kartu kredit, 16 fitur, 6 bulan terakhir<br>
        <b style="color:#303841">Sumber Data:</b> Google BigQuery — ftds-hacktiv8-project
    </div>
    """, unsafe_allow_html=True)
