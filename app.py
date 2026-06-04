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

# ── CSS — Palet: #F5F5F5 · #76ABAE · #303841 · #FF5722 ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background — paper texture */
.stApp {
    background-color: #F5F5F5;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3CfeBlend in='SourceGraphic' mode='multiply'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23noise)' opacity='0.045'/%3E%3C/svg%3E"),
        repeating-linear-gradient(0deg, transparent, transparent 28px, rgba(118,171,174,0.045) 28px, rgba(118,171,174,0.045) 29px),
        repeating-linear-gradient(90deg, transparent, transparent 28px, rgba(118,171,174,0.025) 28px, rgba(118,171,174,0.025) 29px);
    color: #303841;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(245,245,245,0.92) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(118,171,174,0.35) !important;
}
section[data-testid="stSidebar"] * { color: #303841 !important; }

/* Hero card — glassmorphism */
.hero-card {
    background: rgba(48,56,65,0.72);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(118,171,174,0.30);
    border-radius: 20px;
    padding: 2.6rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(48,56,65,0.18), inset 0 1px 0 rgba(255,255,255,0.12);
}
.hero-card::before {
    content: '';
    position: absolute; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E");
    pointer-events: none; border-radius: 20px;
}
.hero-card::after {
    content: '';
    position: absolute; top: -60px; left: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(118,171,174,0.28) 0%, transparent 70%);
    pointer-events: none;
}
.hero-accent {
    position: absolute; bottom: -50px; right: -30px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,87,34,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem; font-weight: 700; color: #F5F5F5;
    letter-spacing: -0.5px; margin: 0 0 0.4rem;
    position: relative; z-index: 1;
}
.hero-sub {
    color: rgba(245,245,245,0.60); font-size: 0.92rem;
    margin: 0; position: relative; z-index: 1;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(118,171,174,0.30);
    border-radius: 14px; padding: 1.2rem 1rem; text-align: center;
    box-shadow: 0 2px 8px rgba(48,56,65,0.07);
}
.metric-card .num { font-size: 2rem; font-weight: 700; color: #76ABAE; }
.metric-card .lbl { font-size: 0.68rem; color: #76869a; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 3px; }

/* Cluster cards */
.cluster-card {
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    border-left: 4px solid;
    background: rgba(255,255,255,0.62);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    border-top: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 2px 10px rgba(48,56,65,0.08);
}
.c0 { border-left-color: #76ABAE; }
.c1 { border-left-color: #9b6fd4; }
.c2 { border-left-color: #3daa6a; }
.c3 { border-left-color: #FF5722; }
.cluster-name { font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem; }
.cluster-desc { font-size: 0.87rem; color: #4a5568; line-height: 1.65; }

/* Section title */
.section-title {
    font-size: 0.68rem; font-weight: 700; color: #76ABAE;
    text-transform: uppercase; letter-spacing: 2.5px; margin: 1.2rem 0 1rem;
}

/* Input labels */
.stNumberInput > label, .stSlider > label {
    color: #303841 !important; font-size: 0.82rem !important; font-weight: 500 !important;
}
.stNumberInput input {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(118,171,174,0.4) !important;
    color: #303841 !important; border-radius: 8px !important;
}

/* Button — orange accent */
.stButton > button {
    background: #FF5722 !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    letter-spacing: 0.6px !important; padding: 0.75rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(255,87,34,0.30) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { background: #e64a19 !important; }

/* Divider */
hr { border-color: rgba(118,171,174,0.25) !important; margin: 1.5rem 0 !important; }

/* Expander */
details {
    background: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(118,171,174,0.30) !important;
    border-radius: 10px !important;
}

/* DataFrame */
.stDataFrame { border: 1px solid rgba(118,171,174,0.30) !important; border-radius: 10px !important; }

/* Hide Streamlit chrome */
footer, #MainMenu, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state — hanya untuk prediksi ─────────────────────────────────────
if "pred_result" not in st.session_state: st.session_state.pred_result = None
if "pred_inputs" not in st.session_state: st.session_state.pred_inputs = None

# ── Constants ──────────────────────────────────────────────────────────────────
ALL_FEATURES = [
    'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
    'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
]

CLUSTER_INFO = {
    0: {
        "name": "Pengguna Konservatif", "emoji": "🛡️", "color": "#76ABAE", "css": "c0",
        "desc": "Pembelian rendah, hati-hati dalam penggunaan kredit. Cenderung membayar lebih dari tagihan tapi tidak sering berbelanja.",
        "strategy": "Tawarkan produk entry-level, cashback untuk kebutuhan sehari-hari, dan edukasi manfaat penggunaan kredit.",
        "badge_bg": "rgba(118,171,174,0.12)",
    },
    1: {
        "name": "Pengguna Premium", "emoji": "💎", "color": "#9b6fd4", "css": "c1",
        "desc": "Semua nilai tinggi — saldo besar, pembelian besar, pembayaran besar. Pelanggan bernilai tinggi yang aktif.",
        "strategy": "Tawarkan layanan premium, travel rewards, lounge access, dan produk eksklusif. Prioritaskan retention.",
        "badge_bg": "rgba(155,111,212,0.10)",
    },
    2: {
        "name": "Pengguna Aktif Harian", "emoji": "🛒", "color": "#3daa6a", "css": "c2",
        "desc": "Saldo dan nominal transaksi rendah, namun frekuensi transaksi sangat tinggi. Berbelanja sering dalam jumlah kecil.",
        "strategy": "Program loyalty points, cashback untuk minimarket & F&B, serta notifikasi promo harian.",
        "badge_bg": "rgba(61,170,106,0.10)",
    },
    3: {
        "name": "Pengguna Pasif", "emoji": "🌱", "color": "#FF5722", "css": "c3",
        "desc": "Semua nilai rendah — saldo, pembelian, maupun pembayaran. Kemungkinan pengguna baru atau tidak aktif.",
        "strategy": "Program onboarding, edukasi finansial, insentif first-purchase, dan reminder aktivasi kartu.",
        "badge_bg": "rgba(255,87,34,0.10)",
    },
}

# ── Load model ─────────────────────────────────────────────────────────────────
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

def preprocess(df):
    std_cols, rob_cols = get_scaler_cols()
    data = df[ALL_FEATURES].copy().astype(float)
    if std_cols:
        data[std_cols] = standard_scaler.transform(data[std_cols])
    if rob_cols:
        data[rob_cols] = robust_scaler.transform(data[rob_cols])
    return pca_model.transform(data)

def predict_cluster(df):
    return km.predict(preprocess(df))

# ── Sidebar — native radio, selalu bisa dibuka/tutup ─────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:0.68rem;font-weight:700;color:#76ABAE;'
        'text-transform:uppercase;letter-spacing:2.5px;margin-bottom:1.2rem">Navigasi</p>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        label="nav",
        options=["🏠 Dashboard", "🔍 Prediksi Manual", "📂 Upload CSV", "📖 Panduan Cluster"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.68rem;font-weight:700;color:#76ABAE;'
        'text-transform:uppercase;letter-spacing:2.5px;margin-bottom:0.8rem">Status Model</p>',
        unsafe_allow_html=True,
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

# ── Helper ─────────────────────────────────────────────────────────────────────
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
    for col, (num, lbl) in zip([c1, c2, c3, c4], [
        ("4.475", "Total Nasabah"), ("4", "Jumlah Cluster"),
        ("16", "Fitur Digunakan"), ("95%", "PCA Variance"),
    ]):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="num">{num}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
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
    steps = ["BigQuery","CSV","EDA","Outlier","Missing Value","Feature Select","Scaling","PCA","K-Means","Segmen"]
    for col, step in zip(st.columns(len(steps)), steps):
        with col:
            st.markdown(
                f"<div style='text-align:center;font-size:0.62rem;color:#76ABAE;"
                f"background:rgba(255,255,255,0.65);backdrop-filter:blur(6px);"
                f"border:1px solid rgba(118,171,174,0.3);border-radius:8px;"
                f"padding:0.45rem 0.1rem;font-weight:600'>{step}</div>",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI MANUAL
# ══════════════════════════════════════════════════════════════════════════════
elif "Manual" in page:
    hero("🔍", "Prediksi Manual", "Masukkan data nasabah secara manual untuk mengetahui segmennya")

    if not models_loaded:
        st.error("Model belum dimuat. Pastikan file .pkl tersedia.")
        st.stop()

    # Ambil nilai tersimpan (agar form tidak reset setelah prediksi)
    def sv(key, default):
        return st.session_state.pred_inputs.get(key, default) if st.session_state.pred_inputs else default

    st.markdown('<p class="section-title">💰 Informasi Keuangan</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        balance      = st.number_input("Saldo (BALANCE)",                               min_value=0.0, value=sv("balance",       1500.0), step=100.0)
        credit_limit = st.number_input("Batas Kredit (CREDIT_LIMIT)",                   min_value=0.0, value=sv("credit_limit",  5000.0), step=500.0)
        payments     = st.number_input("Total Pembayaran (PAYMENTS)",                   min_value=0.0, value=sv("payments",      1200.0), step=100.0)
    with col2:
        purchases    = st.number_input("Total Pembelian (PURCHASES)",                   min_value=0.0, value=sv("purchases",      800.0), step=100.0)
        oneoff       = st.number_input("Pembelian Sekaligus (ONEOFF_PURCHASES)",        min_value=0.0, value=sv("oneoff",          200.0), step=50.0)
        installments = st.number_input("Pembelian Cicilan (INSTALLMENTS_PURCHASES)",    min_value=0.0, value=sv("installments",   600.0), step=50.0)
    with col3:
        cash_advance = st.number_input("Tarik Tunai (CASH_ADVANCE)",                    min_value=0.0, value=sv("cash_advance",     0.0), step=50.0)
        min_payments = st.number_input("Pembayaran Minimum (MINIMUM_PAYMENTS)",         min_value=0.0, value=sv("min_payments",   200.0), step=50.0)
        cash_adv_trx = st.number_input("Jml Transaksi Tarik Tunai (CASH_ADVANCE_TRX)", min_value=0,   value=sv("cash_adv_trx",      0), step=1)

    st.markdown("---")
    st.markdown('<p class="section-title">📈 Frekuensi & Transaksi</p>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        bal_freq      = st.slider("Frekuensi Saldo (BALANCE_FREQUENCY)",                           0.0, 1.0, sv("bal_freq",      0.9), 0.01)
        purch_freq    = st.slider("Frekuensi Pembelian (PURCHASES_FREQUENCY)",                     0.0, 1.0, sv("purch_freq",    0.5), 0.01)
    with col5:
        oneoff_freq   = st.slider("Frekuensi Pembelian Sekaligus (ONEOFF_PURCHASES_FREQUENCY)",    0.0, 1.0, sv("oneoff_freq",   0.2), 0.01)
        install_freq  = st.slider("Frekuensi Cicilan (PURCHASES_INSTALLMENTS_FREQUENCY)",          0.0, 1.0, sv("install_freq",  0.4), 0.01)
    with col6:
        cash_adv_freq = st.slider("Frekuensi Tarik Tunai (CASH_ADVANCE_FREQUENCY)",               0.0, 1.0, sv("cash_adv_freq", 0.0), 0.01)
        prc_full      = st.slider("Persentase Bayar Penuh (PRC_FULL_PAYMENT)",                     0.0, 1.0, sv("prc_full",      0.3), 0.01)

    purch_trx = st.number_input("Jumlah Transaksi Pembelian (PURCHASES_TRX)", min_value=0, value=sv("purch_trx", 10), step=1)

    st.markdown("---")

    # Tombol prediksi — simpan ke session_state, TANPA st.rerun()
    if st.button("🚀  PREDIKSI SEGMEN"):
        st.session_state.pred_inputs = {
            "balance": balance, "credit_limit": credit_limit, "payments": payments,
            "purchases": purchases, "oneoff": oneoff, "installments": installments,
            "cash_advance": cash_advance, "min_payments": min_payments, "cash_adv_trx": cash_adv_trx,
            "bal_freq": bal_freq, "purch_freq": purch_freq, "oneoff_freq": oneoff_freq,
            "install_freq": install_freq, "cash_adv_freq": cash_adv_freq, "prc_full": prc_full,
            "purch_trx": purch_trx,
        }
        try:
            input_data = pd.DataFrame([{
                'BALANCE': balance, 'BALANCE_FREQUENCY': bal_freq,
                'PURCHASES': purchases, 'ONEOFF_PURCHASES': oneoff,
                'INSTALLMENTS_PURCHASES': installments, 'CASH_ADVANCE': cash_advance,
                'PURCHASES_FREQUENCY': purch_freq, 'ONEOFF_PURCHASES_FREQUENCY': oneoff_freq,
                'PURCHASES_INSTALLMENTS_FREQUENCY': install_freq,
                'CASH_ADVANCE_FREQUENCY': cash_adv_freq,
                'CASH_ADVANCE_TRX': float(cash_adv_trx), 'PURCHASES_TRX': float(purch_trx),
                'CREDIT_LIMIT': credit_limit, 'PAYMENTS': payments,
                'MINIMUM_PAYMENTS': min_payments, 'PRC_FULL_PAYMENT': prc_full,
            }])
            st.session_state.pred_result = int(predict_cluster(input_data)[0])
        except Exception as e:
            st.session_state.pred_result = None
            st.error(f"❌ Error: {e}")
            st.info("Cek sidebar → Detail kolom scaler untuk debug.")

    # Tampilkan hasil — tetap muncul karena dibaca dari session_state
    if st.session_state.pred_result is not None:
        label = st.session_state.pred_result
        info  = CLUSTER_INFO[label]
        st.markdown("---")
        st.markdown(f"""
        <div style="background:{info['badge_bg']};border:1.5px solid {info['color']};
                    backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
                    border-radius:18px;padding:2.2rem;text-align:center;
                    margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(48,56,65,0.10);">
            <div style="font-size:3rem;margin-bottom:0.5rem">{info['emoji']}</div>
            <div style="font-size:1.7rem;font-weight:700;color:{info['color']}">Cluster {label}</div>
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
        if st.button("🔄 Reset / Prediksi Ulang"):
            st.session_state.pred_result = None
            st.session_state.pred_inputs = None
            st.rerun()

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
                labels    = predict_cluster(df_up)
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
            bars = ax.bar(
                [f"C{i} {CLUSTER_INFO[i]['emoji']}" for i in counts.index],
                counts.values,
                color=[CLUSTER_INFO[i]['color'] for i in counts.index],
                width=0.5, edgecolor='none',
            )
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        f'{val:,}', ha='center', va='bottom', color='#303841', fontsize=10)
            ax.set_xlabel('Cluster', color='#76ABAE', fontsize=10)
            ax.set_ylabel('Jumlah Nasabah', color='#76ABAE', fontsize=10)
            ax.tick_params(colors='#303841')
            for spine in ax.spines.values():
                spine.set_edgecolor('#ddd')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown('<p class="section-title">📋 Hasil (50 baris pertama)</p>', unsafe_allow_html=True)
            st.dataframe(df_result[['CLUSTER', 'SEGMEN'] + ALL_FEATURES[:6]].head(50), use_container_width=True)

            st.download_button(
                "⬇ Download Hasil Lengkap (CSV)",
                data=df_result.to_csv(index=False).encode('utf-8'),
                file_name="hasil_segmentasi.csv", mime="text/csv",
            )
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
