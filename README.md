# 💳 Credit Card Customer Segmentation — Graded Challenge 6

> **Hacktiv8 Data Science Fulltime Program | MSIB-06 (HCK-014)**  
> **Desvin Sitohang**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cluster-analyze-npwxjyapszmoq9elelrkr6.streamlit.app/)

🚀 **Live Demo:** [https://cluster-analyze-npwxjyapszmoq9elelrkr6.streamlit.app/](https://cluster-analyze-npwxjyapszmoq9elelrkr6.streamlit.app/)

---

## 📌 Deskripsi Proyek

Proyek ini merupakan implementasi **Customer Segmentation** menggunakan algoritma **K-Means Clustering** pada data informasi penggunaan kartu kredit selama 6 bulan terakhir. Tujuannya adalah membantu pihak bank memahami perilaku dan kebutuhan nasabah kartu kredit mereka agar dapat menyusun strategi pemasaran yang lebih tepat sasaran.

---

## 🎯 Tujuan

1. **Akurasi Segmentasi** — Mengelompokkan pelanggan ke dalam segmen yang mencerminkan kemiripan perilaku penggunaan kartu kredit, diukur menggunakan Silhouette Score.
2. **Peningkatan Penjualan** — Menghasilkan insight yang dapat meningkatkan volume transaksi dan pendapatan bank.
3. **Efisiensi Pemasaran** — Membantu bank menyasar segmen yang tepat dengan kampanye yang lebih relevan.

---

## 🗂️ Dataset

- **Sumber**: Google BigQuery — `ftds-hacktiv8-project.phase1_ftds_msib_hck.credit-card-information`
- **Filter**: Data dengan `CUST_ID` bernilai genap (Batch genap)
- **Ukuran**: 4.475 baris × 18 kolom
- **File CSV**: `P1G6_Set_1_Desvin_Sitohang.csv`

### Query SQL

```sql
SELECT *
FROM `ftds-hacktiv8-project.phase1_ftds_msib_hck.credit-card-information`
WHERE MOD(CUST_ID, 2) = 0;
```

### Deskripsi Kolom

| Kolom | Deskripsi |
|---|---|
| `CUST_ID` | ID unik setiap nasabah |
| `BALANCE` | Saldo yang tersisa di akun |
| `BALANCE_FREQUENCY` | Frekuensi pembaruan saldo (0–1) |
| `PURCHASES` | Total nilai pembelian |
| `ONEOFF_PURCHASES` | Pembelian sekaligus (bukan cicilan) |
| `INSTALLMENTS_PURCHASES` | Pembelian secara cicilan |
| `CASH_ADVANCE` | Penarikan tunai menggunakan kartu kredit |
| `PURCHASES_FREQUENCY` | Frekuensi pembelian (0–1) |
| `ONEOFF_PURCHASES_FREQUENCY` | Frekuensi pembelian sekaligus (0–1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frekuensi pembelian cicilan (0–1) |
| `CASH_ADVANCE_FREQUENCY` | Frekuensi penarikan tunai (0–1) |
| `CASH_ADVANCE_TRX` | Jumlah transaksi penarikan tunai |
| `PURCHASES_TRX` | Jumlah transaksi pembelian |
| `CREDIT_LIMIT` | Batas kredit nasabah |
| `PAYMENTS` | Total pembayaran yang dilakukan |
| `MINIMUM_PAYMENTS` | Total pembayaran minimum |
| `PRC_FULL_PAYMENT` | Persentase pembayaran penuh (0–1) |
| `TENURE` | Lama menjadi nasabah (bulan) |

---

## 🧰 Library yang Digunakan

```python
pandas, numpy                          # Analisis data
matplotlib, seaborn, squarify          # Visualisasi
scipy.stats                            # Statistik
sklearn (preprocessing, decomposition, cluster, metrics)  # ML
feature_engine                         # Penanganan outlier
pickle, joblib                         # Menyimpan model
```

---

## 🔄 Alur Pengerjaan

```
BigQuery → CSV → EDA → Feature Engineering → PCA → K-Means → Evaluasi → Inference
```

### 1. Data Loading
Memuat data dari CSV hasil ekspor BigQuery, lalu memeriksa struktur, tipe data, dan duplikasi.

### 2. Exploratory Data Analysis (EDA)
- Scatter plot perbandingan pembelian sekaligus vs. cicilan
- Distribusi `BALANCE_FREQUENCY`
- KDE plot perbandingan `PURCHASES` vs. `PAYMENTS`
- Analisis segmen pembayaran (full, in-range, zero)

**Insight Utama:**
- Mayoritas pengguna lebih memilih pembelian secara angsuran
- Tingkat pembayaran 0% (tidak bayar sama sekali) sangat tinggi
- Semakin besar pembelian, semakin besar pula pembayaran yang dilakukan

### 3. Feature Engineering

| Langkah | Metode |
|---|---|
| Hapus kolom ID | Drop `CUST_ID` |
| Handling Missing Values | Imputasi median (untuk `CREDIT_LIMIT`, `MINIMUM_PAYMENTS`) |
| Handling Outlier | Winsorizer IQR Capping (skew) + Trimming (`TENURE`) |
| Feature Selection | Pearson Correlation |
| Scaling | StandardScaler (distribusi normal) + RobustScaler (distribusi skew) |

### 4. Dimensionality Reduction (PCA)
- PCA diterapkan untuk mengurangi dimensi sekaligus mempertahankan **95% explained variance**
- Hasil: dimensi berkurang dari 16 → lebih ringkas dan efisien untuk clustering

### 5. Model K-Means

**Penentuan jumlah cluster optimal:**
- **Elbow Method** → siku terbentuk di `k=4`
- **Silhouette Score** → nilai terbaik di `k=2`, namun persebaran di `k=4` lebih representatif secara bisnis

**Model final:** `KMeans(n_clusters=4, init='k-means++', random_state=10)`

---

## 📊 Hasil Clustering

| Cluster | Karakteristik | Label Bisnis |
|---|---|---|
| **0** | Pembelian rendah, pembayaran bervariasi | 🛡️ Pengguna Konservatif |
| **1** | Semua nilai tinggi dan bervariasi | 💎 Pengguna Premium / High Spender |
| **2** | Saldo & pembelian rendah, frekuensi transaksi tinggi | 🛒 Pengguna Aktif Harian |
| **3** | Saldo, pembelian, dan pembayaran semua rendah | 🌱 Pengguna Pasif / Pemula |

---

## 📁 Struktur File

```
📂 Project
├── app.py                              # Streamlit deployment app
├── requirements.txt                    # Dependencies
├── P1G6_Set_1_Desvin_Sitohang.ipynb   # Notebook utama
├── P1G6_Set_1_Desvin_Sitohang.csv     # Dataset hasil BigQuery
├── km.pkl                              # Model K-Means tersimpan
├── pca_final.pkl                       # Model PCA tersimpan
├── robust_scaler.pkl                   # RobustScaler tersimpan
└── standard_scaler.pkl                 # StandardScaler tersimpan
```

---

## ▶️ Cara Menjalankan Lokal

1. Clone repository ini
2. Install semua dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Atau jalankan notebook secara langsung — pastikan file CSV ada di direktori yang sama.

---

## 💡 Conceptual Problems

**1. Apa itu Inertia pada K-Means?**

Inertia adalah jumlah kuadrat jarak setiap titik data ke centroid cluster-nya masing-masing, dijumlahkan untuk semua cluster. Semakin kecil nilai inertia, semakin padat dan kohesif cluster yang terbentuk. K-Means bekerja dengan meminimalkan nilai inertia ini melalui iterasi berulang.

**2. Elbow Method**

Elbow Method digunakan untuk menentukan jumlah cluster optimal pada K-Means dengan cara memplot nilai inertia terhadap jumlah cluster (k). Titik "siku" — di mana penurunan inertia mulai melambat secara signifikan — menjadi kandidat nilai k terbaik. Kelebihannya adalah mudah divisualisasikan, namun kelemahannya adalah titik siku tidak selalu jelas dan bisa bersifat subjektif, sehingga perlu dikombinasikan dengan metode lain seperti Silhouette Score.

---

## 👤 Author

**Desvin Sitohang**  
Batch: MSIB-06 (HCK-014) — Hacktiv8 Data Science Fulltime Program
