import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import pymannkendall as mk

# ============================================================
# PAGE CONFIG & GLOBAL STYLE
# ============================================================

st.set_page_config(
    page_title="Dashboard Curah Hujan Jawa Barat",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS -----------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Root palette */
:root {
    --bg-deep:    #0a0e1a;
    --bg-card:    #111827;
    --bg-card2:   #1a2236;
    --accent-blue:#3b82f6;
    --accent-cyan:#06b6d4;
    --accent-teal:#14b8a6;
    --accent-amber:#f59e0b;
    --text-primary:#f1f5f9;
    --text-muted: #94a3b8;
    --border:     #1e2d45;
    --success:    #10b981;
    --danger:     #ef4444;
}

/* App background */
.stApp {
    background: var(--bg-deep);
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] { display:none; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
    padding-top: 0;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    transition: border-color .25s;
}
div[data-testid="metric-container"]:hover { border-color: var(--accent-blue); }
div[data-testid="metric-container"] label { color: var(--text-muted) !important; font-size:.8rem; letter-spacing:.05em; text-transform:uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size:1.9rem; font-weight:700; }

/* Dataframe */
.stDataFrame { border-radius:10px; overflow:hidden; }

/* Section heading helper class */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 28px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 10px;
}

/* Interpretation box */
.interpretation-box {
    background: linear-gradient(135deg, #0f2040 0%, #0d1f3a 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid var(--accent-blue);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 14px;
    font-size: .88rem;
    line-height: 1.7;
    color: #cbd5e1;
}
.interpretation-box .icon { font-size:1rem; margin-right:6px; }
.interpretation-box strong { color: var(--accent-cyan); }

/* Alert/badge */
.badge {
    display:inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size:.75rem;
    font-weight:600;
    letter-spacing:.04em;
}
.badge-up   { background:#14532d; color:#4ade80; }
.badge-down { background:#450a0a; color:#f87171; }
.badge-neutral { background:#1e3a5f; color:#60a5fa; }

/* Plot area */
.plot-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 8px;
}

/* Sidebar nav items */
.nav-label {
    font-size:.7rem;
    font-weight:700;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:var(--text-muted);
    padding: 20px 16px 6px;
}

/* Buttons */
.stButton > button {
    background: var(--accent-blue) !important;
    color:#fff !important;
    border:none !important;
    border-radius:8px !important;
    font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB DARK THEME
# ============================================================

DARK_BG   = "#0a0e1a"
CARD_BG   = "#111827"
CARD_BG2  = "#1a2236"
BLUE      = "#3b82f6"
CYAN      = "#06b6d4"
TEAL      = "#14b8a6"
AMBER     = "#f59e0b"
ROSE      = "#f43f5e"
VIOLET    = "#a78bfa"
TEXT      = "#f1f5f9"
MUTED     = "#64748b"
GRID      = "#1e2d45"

PALETTE = [BLUE, CYAN, TEAL, AMBER, ROSE, VIOLET,
           "#34d399", "#fb923c", "#818cf8", "#e879f9"]

def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG2)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

def apply_dark_style_multi(fig, axes):
    fig.patch.set_facecolor(CARD_BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        apply_dark_style(fig, ax)


# ============================================================
# BIGQUERY CONNECTION
# ============================================================

PROJECT_ID  = "st-project-488506"
DATASET_NAME = "uas_adbc"
TABLE_NAME   = "dataset_uas"
FULL_TABLE   = f"`{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}`"

@st.cache_resource
def get_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials, project=PROJECT_ID)

client = get_client()

@st.cache_data(ttl=600)
def run_query(query):
    return client.query(query).to_dataframe()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1d4ed8,#0891b2);
                padding:24px 20px; margin:-16px -16px 0; border-radius:0 0 14px 14px;">
        <div style="font-size:1.6rem; margin-bottom:4px;">🌧️</div>
        <div style="font-size:1rem; font-weight:800; color:#fff; letter-spacing:.02em;">
            Dashboard Curah Hujan
        </div>
        <div style="font-size:.75rem; color:#bfdbfe; margin-top:2px;">Jawa Barat — Analisis Hidrologis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigasi</div>', unsafe_allow_html=True)

    menu = st.radio(
        label="Menu",
        options=["📊 Data", "📈 Analisis Tren", "🗺️ Visualisasi"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:.72rem;color:#475569;padding:0 4px;">'
        'Data: BigQuery · Model: Mann-Kendall · Sumber: BMKG Jawa Barat</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SHARED DATA QUERIES
# ============================================================

@st.cache_data(ttl=600)
def load_clean_data():
    sql = f"""
    WITH deduplicated AS (
        SELECT *,
               ROW_NUMBER() OVER(PARTITION BY event_id ORDER BY event_date) AS rn
        FROM {FULL_TABLE}
        WHERE event_id IS NOT NULL
          AND event_date IS NOT NULL
          AND district_city IS NOT NULL
          AND rainfall_mm >= 0
    )
    SELECT event_date,
           INITCAP(TRIM(district_city)) AS district_city,
           rainfall_mm
    FROM deduplicated WHERE rn = 1
    """
    df = run_query(sql)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df

@st.cache_data(ttl=600)
def load_struktur():
    return run_query(f"""
        SELECT COUNT(*) AS jumlah_observasi,
               COUNT(DISTINCT event_id) AS unique_event_id,
               COUNT(DISTINCT district_city) AS jumlah_wilayah,
               MIN(event_date) AS tanggal_awal,
               MAX(event_date) AS tanggal_akhir
        FROM {FULL_TABLE}
    """)

@st.cache_data(ttl=600)
def load_tren():
    df = run_query(f"""
        SELECT DATE_TRUNC(event_date, MONTH) AS bulan_dt,
               ROUND(AVG(rainfall_mm), 2)    AS rata_curah_hujan
        FROM {FULL_TABLE}
        GROUP BY bulan_dt ORDER BY bulan_dt
    """)
    df["bulan_dt"] = pd.to_datetime(df["bulan_dt"])
    return df

@st.cache_data(ttl=600)
def load_wilayah_avg():
    return run_query(f"""
        SELECT INITCAP(TRIM(district_city)) AS district_city,
               ROUND(AVG(rainfall_mm), 2)   AS rata_rata
        FROM {FULL_TABLE}
        GROUP BY district_city ORDER BY rata_rata DESC LIMIT 10
    """)

@st.cache_data(ttl=600)
def load_wilayah_count():
    return run_query(f"""
        SELECT INITCAP(TRIM(district_city)) AS district_city,
               COUNT(*) AS total_kejadian
        FROM {FULL_TABLE}
        GROUP BY district_city ORDER BY total_kejadian DESC LIMIT 10
    """)

@st.cache_data(ttl=600)
def load_musiman():
    return run_query(f"""
        SELECT EXTRACT(MONTH FROM event_date) AS bulan,
               FORMAT_DATE('%b', DATE(2024, CAST(EXTRACT(MONTH FROM event_date) AS INT64), 1)) AS nama_bulan,
               ROUND(AVG(rainfall_mm), 2) AS rata_rata
        FROM {FULL_TABLE} WHERE rainfall_mm >= 0
        GROUP BY bulan, nama_bulan ORDER BY bulan
    """)

@st.cache_data(ttl=600)
def load_heatmap():
    return run_query(f"""
        SELECT EXTRACT(MONTH FROM event_date) AS bulan,
               EXTRACT(YEAR FROM event_date)  AS tahun,
               ROUND(AVG(rainfall_mm), 2)     AS rata_rata
        FROM {FULL_TABLE} WHERE rainfall_mm >= 0
        GROUP BY bulan, tahun ORDER BY bulan, tahun
    """)

@st.cache_data(ttl=600)
def load_tren_wilayah():
    df = run_query(f"""
        WITH top_wil AS (
            SELECT INITCAP(TRIM(district_city)) AS wilayah
            FROM {FULL_TABLE}
            GROUP BY wilayah ORDER BY COUNT(*) DESC LIMIT 10
        )
        SELECT DATE_TRUNC(event_date, MONTH) AS bulan_dt,
               INITCAP(TRIM(t.district_city)) AS wilayah,
               ROUND(AVG(t.rainfall_mm), 2)   AS rata_curah_hujan
        FROM {FULL_TABLE} t
        JOIN top_wil tw ON INITCAP(TRIM(t.district_city)) = tw.wilayah
        GROUP BY bulan_dt, wilayah ORDER BY bulan_dt, wilayah
    """)
    df["bulan_dt"] = pd.to_datetime(df["bulan_dt"])
    return df


# ============================================================
# HELPER: Interpretation box
# ============================================================

def interpret(text: str):
    st.markdown(
        f'<div class="interpretation-box">💡 <strong>Interpretasi:</strong> {text}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE: DATA
# ============================================================

if menu == "📊 Data":

    st.markdown("""
    <h1 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:2rem;
               font-weight:800;color:#f1f5f9;margin-bottom:4px;">
        📊 Ringkasan Dataset
    </h1>
    <p style="color:#64748b;margin-bottom:24px;">
        Tinjauan menyeluruh terhadap dataset curah hujan Jawa Barat yang tersimpan di BigQuery.
    </p>
    """, unsafe_allow_html=True)

    # --- Metrics ---
    df_str = load_struktur()
    tgl_awal = pd.to_datetime(df_str["tanggal_awal"][0]).strftime("%d %b %Y")
    tgl_akhir = pd.to_datetime(df_str["tanggal_akhir"][0]).strftime("%d %b %Y")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🗂️ Total Observasi",    f'{int(df_str["jumlah_observasi"][0]):,}')
    c2.metric("📍 Jumlah Wilayah",     int(df_str["jumlah_wilayah"][0]))
    c3.metric("🔑 Unique Event ID",    int(df_str["unique_event_id"][0]))
    c4.metric("📅 Rentang Data",       f"{tgl_awal} – {tgl_akhir}")

    # --- Preview ---
    st.markdown('<div class="section-title">🔍 Preview Data (10 baris pertama)</div>', unsafe_allow_html=True)
    df_preview = run_query(f"SELECT * FROM {FULL_TABLE} LIMIT 10")
    st.dataframe(df_preview, use_container_width=True)

    # --- Stats deskriptif ---
    st.markdown('<div class="section-title">📐 Statistik Deskriptif</div>', unsafe_allow_html=True)
    df = load_clean_data()

    desc = df["rainfall_mm"].describe().rename({
        "count":"Jumlah Data", "mean":"Rata-rata (mm)",
        "std":"Std. Deviasi", "min":"Minimum",
        "25%":"Kuartil 1 (25%)", "50%":"Median (50%)",
        "75%":"Kuartil 3 (75%)", "max":"Maksimum"
    })
    st.dataframe(
        desc.to_frame("Nilai").style.format("{:.2f}"),
        use_container_width=True
    )

    interpret(
        f"Dataset mencakup <strong>{int(df_str['jumlah_observasi'][0]):,}</strong> observasi dari "
        f"<strong>{int(df_str['jumlah_wilayah'][0])}</strong> wilayah di Jawa Barat. "
        f"Rentang waktu data dari <strong>{tgl_awal}</strong> hingga <strong>{tgl_akhir}</strong>. "
        f"Rata-rata curah hujan adalah <strong>{df['rainfall_mm'].mean():.1f} mm</strong> "
        f"dengan nilai maksimum <strong>{df['rainfall_mm'].max():.1f} mm</strong>, "
        f"menunjukkan variasi yang cukup tinggi antar-kejadian."
    )

    # --- Distribusi ---
    st.markdown('<div class="section-title">📊 Distribusi Curah Hujan</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    apply_dark_style(fig, ax)

    n, bins, patches = ax.hist(df["rainfall_mm"], bins=30, color=BLUE, edgecolor=CARD_BG2, linewidth=0.5, alpha=0.85)
    # Color gradient
    for i, p in enumerate(patches):
        p.set_facecolor(plt.cm.cool(i / len(patches)))

    ax.set_xlabel("Curah Hujan (mm)")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi Frekuensi Curah Hujan")

    # Median & Mean lines
    ax.axvline(df["rainfall_mm"].mean(),   color=AMBER, linestyle="--", linewidth=1.4, label=f'Mean: {df["rainfall_mm"].mean():.1f} mm')
    ax.axvline(df["rainfall_mm"].median(), color=ROSE,  linestyle=":",  linewidth=1.4, label=f'Median: {df["rainfall_mm"].median():.1f} mm')
    ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    st.pyplot(fig, use_container_width=True)

    skew_val = df["rainfall_mm"].skew()
    skew_dir = "kanan (positif)" if skew_val > 0 else "kiri (negatif)"
    interpret(
        f"Distribusi curah hujan bersifat <strong>skewed {skew_dir}</strong> (skewness = {skew_val:.2f}), "
        f"artinya sebagian besar kejadian memiliki curah hujan rendah hingga sedang, "
        f"namun terdapat sejumlah kejadian ekstrem dengan nilai sangat tinggi. "
        f"Nilai <strong>mean ({df['rainfall_mm'].mean():.1f} mm) > median ({df['rainfall_mm'].median():.1f} mm)</strong> "
        f"mengonfirmasi distribusi tidak simetris."
    )


# ============================================================
# PAGE: ANALISIS TREN
# ============================================================

elif menu == "📈 Analisis Tren":

    st.markdown("""
    <h1 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:2rem;
               font-weight:800;color:#f1f5f9;margin-bottom:4px;">
        📈 Analisis Tren
    </h1>
    <p style="color:#64748b;margin-bottom:24px;">
        Tren temporal curah hujan dan uji statistik Mann-Kendall.
    </p>
    """, unsafe_allow_html=True)

    df_tren = load_tren()
    x = np.arange(len(df_tren))
    z = np.polyfit(x, df_tren["rata_curah_hujan"], 1)
    p = np.poly1d(z)

    # Mann-Kendall
    result = mk.original_test(df_tren["rata_curah_hujan"].values)
    trend_label = result.trend

    # Trend badge
    badge_class = "badge-up" if trend_label == "increasing" else ("badge-down" if trend_label == "decreasing" else "badge-neutral")
    badge_text  = "⬆ Meningkat" if trend_label == "increasing" else ("⬇ Menurun" if trend_label == "decreasing" else "➡ Stabil")

    # --- Tren bulanan ---
    st.markdown('<div class="section-title">📉 Tren Curah Hujan Bulanan</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    apply_dark_style(fig, ax)

    ax.fill_between(df_tren["bulan_dt"], df_tren["rata_curah_hujan"],
                    alpha=0.18, color=CYAN)
    ax.plot(df_tren["bulan_dt"], df_tren["rata_curah_hujan"],
            color=CYAN, linewidth=2, marker="o", markersize=4, label="Rata-rata Bulanan")
    ax.plot(df_tren["bulan_dt"], p(x),
            color=AMBER, linewidth=2.2, linestyle="--", label=f"Tren Linear (slope={z[0]:+.2f})")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.set_ylabel("Curah Hujan (mm)")
    ax.set_title("Tren Rata-rata Curah Hujan Bulanan")
    ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    st.pyplot(fig, use_container_width=True)

    interpret(
        f"Grafik tren menunjukkan fluktuasi bulanan yang signifikan. "
        f"Garis tren linear memiliki <strong>slope {z[0]:+.4f} mm/bulan</strong>, "
        f"menandakan tren <strong>{'meningkat' if z[0]>0 else 'menurun'}</strong> secara keseluruhan. "
        f"Pola musiman terlihat jelas dengan puncak curah hujan pada bulan-bulan tertentu."
    )

    # --- Mann-Kendall ---
    st.markdown('<div class="section-title">🧪 Uji Statistik Mann-Kendall</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tren",    badge_text)
    m2.metric("p-value", f"{result.p:.4f}")
    m3.metric("Tau (τ)", f"{result.Tau:.4f}")
    m4.metric("S statistic", f"{result.s}")

    sig = result.p < 0.05
    interpret(
        f"Berdasarkan Uji Mann-Kendall, tren curah hujan secara keseluruhan bersifat "
        f"<strong>{trend_label}</strong> dengan nilai p = <strong>{result.p:.4f}</strong>. "
        f"{'✅ Tren ini <strong>signifikan secara statistik</strong> (p < 0,05), menunjukkan perubahan yang nyata, bukan acak.' if sig else '⚠️ Tren ini <strong>tidak signifikan secara statistik</strong> (p ≥ 0,05), sehingga perubahan masih dalam batas variabilitas alami.'} "
        f"Nilai Tau = <strong>{result.Tau:.4f}</strong> menunjukkan {'korelasi positif' if result.Tau > 0 else 'korelasi negatif'} antara waktu dan curah hujan."
    )

    # --- Tren per wilayah ---
    st.markdown('<div class="section-title">🗺️ Tren per Wilayah (Top 10)</div>', unsafe_allow_html=True)

    df_tw = load_tren_wilayah()

    fig, ax = plt.subplots(figsize=(13, 5))
    apply_dark_style(fig, ax)

    for i, (wil, data) in enumerate(df_tw.groupby("wilayah")):
        ax.plot(data["bulan_dt"], data["rata_curah_hujan"],
                color=PALETTE[i % len(PALETTE)], linewidth=1.6,
                alpha=0.85, label=wil)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.set_ylabel("Curah Hujan (mm)")
    ax.set_title("Tren Curah Hujan 10 Wilayah Teratas")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    st.pyplot(fig, use_container_width=True)

    interpret(
        "Tren per wilayah memperlihatkan <strong>heterogenitas spasial</strong> yang nyata. "
        "Beberapa wilayah menunjukkan pola curah hujan yang relatif stabil, "
        "sementara wilayah lain memiliki fluktuasi ekstrem pada periode tertentu. "
        "Perbedaan ini dapat dipengaruhi oleh faktor topografi, jarak dari laut, "
        "dan kondisi lokal masing-masing kabupaten/kota."
    )


# ============================================================
# PAGE: VISUALISASI
# ============================================================

elif menu == "🗺️ Visualisasi":

    st.markdown("""
    <h1 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:2rem;
               font-weight:800;color:#f1f5f9;margin-bottom:4px;">
        🗺️ Visualisasi Spasial & Musiman
    </h1>
    <p style="color:#64748b;margin-bottom:24px;">
        Pola distribusi curah hujan antar wilayah dan antar musim.
    </p>
    """, unsafe_allow_html=True)

    # --- Top Wilayah (rata-rata) ---
    st.markdown('<div class="section-title">🏆 Top 10 Wilayah — Rata-rata Curah Hujan Tertinggi</div>',
                unsafe_allow_html=True)

    df_wa = load_wilayah_avg()

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(fig, ax)

    bars = ax.barh(df_wa["district_city"], df_wa["rata_rata"],
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(df_wa))],
                   edgecolor=CARD_BG2, linewidth=0.5, height=0.65)
    ax.invert_yaxis()

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2,
                f"{w:.1f} mm", va="center", ha="left", color=TEXT, fontsize=8.5)

    ax.set_xlabel("Rata-rata Curah Hujan (mm)")
    ax.set_title("Top 10 Wilayah Curah Hujan Tertinggi")
    ax.set_xlim(0, df_wa["rata_rata"].max() * 1.15)

    st.pyplot(fig, use_container_width=True)

    top1 = df_wa.iloc[0]
    interpret(
        f"<strong>{top1['district_city']}</strong> mencatat rata-rata curah hujan tertinggi "
        f"sebesar <strong>{top1['rata_rata']:.1f} mm</strong>. "
        f"Wilayah-wilayah dengan curah hujan tinggi umumnya terletak di zona pegunungan "
        f"atau berhadapan langsung dengan angin monsun. Informasi ini penting untuk "
        f"perencanaan infrastruktur drainase dan mitigasi bencana banjir."
    )

    # --- Top Wilayah (frekuensi) ---
    st.markdown('<div class="section-title">📊 Top 10 Wilayah — Frekuensi Kejadian Hujan</div>',
                unsafe_allow_html=True)

    df_wc = load_wilayah_count()

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(fig, ax)

    colors_wc = [TEAL if i < 3 else MUTED for i in range(len(df_wc))]
    ax.barh(df_wc["district_city"], df_wc["total_kejadian"],
            color=colors_wc, edgecolor=CARD_BG2, linewidth=0.5, height=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("Jumlah Kejadian")
    ax.set_title("Frekuensi Kejadian Hujan per Wilayah")

    for i, (_, row) in enumerate(df_wc.iterrows()):
        ax.text(row["total_kejadian"] + 1, i,
                str(int(row["total_kejadian"])), va="center", color=TEXT, fontsize=8.5)

    st.pyplot(fig, use_container_width=True)

    interpret(
        f"Wilayah dengan frekuensi kejadian terbanyak adalah "
        f"<strong>{df_wc.iloc[0]['district_city']}</strong> "
        f"({int(df_wc.iloc[0]['total_kejadian']):,} kejadian). "
        f"Perlu dibedakan antara wilayah dengan <em>intensitas tinggi</em> (rata-rata mm besar) "
        f"dan wilayah dengan <em>frekuensi tinggi</em> (sering hujan). "
        f"Wilayah yang dominan di kedua kategori memerlukan perhatian khusus dalam manajemen risiko."
    )

    st.markdown("---")

    # --- Pola Musiman ---
    st.markdown('<div class="section-title">🌦️ Pola Musiman</div>', unsafe_allow_html=True)

    df_mus  = load_musiman()
    df_heat = load_heatmap()

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        apply_dark_style(fig, ax)

        bar_colors = [BLUE if v >= df_mus["rata_rata"].mean() else MUTED
                      for v in df_mus["rata_rata"]]
        ax.bar(df_mus["nama_bulan"], df_mus["rata_rata"],
               color=bar_colors, edgecolor=CARD_BG2, linewidth=0.5)

        mean_line = df_mus["rata_rata"].mean()
        ax.axhline(mean_line, color=AMBER, linestyle="--", linewidth=1.2,
                   label=f"Rata-rata: {mean_line:.1f} mm")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Curah Hujan (mm)")
        ax.set_title("Rata-rata Curah Hujan per Bulan")
        ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

        st.pyplot(fig, use_container_width=True)

        peak_month = df_mus.loc[df_mus["rata_rata"].idxmax(), "nama_bulan"]
        dry_month  = df_mus.loc[df_mus["rata_rata"].idxmin(), "nama_bulan"]
        interpret(
            f"Pola musiman jelas terlihat dengan puncak curah hujan pada bulan "
            f"<strong>{peak_month}</strong> dan curah hujan terendah di bulan "
            f"<strong>{dry_month}</strong>. Bulan-bulan berwarna biru menunjukkan "
            f"curah hujan di atas rata-rata tahunan, mengindikasikan musim hujan dominan."
        )

    with col2:
        pivot = df_heat.pivot(index="bulan", columns="tahun", values="rata_rata")

        fig, ax = plt.subplots(figsize=(7, 4))
        apply_dark_style(fig, ax)

        sns.heatmap(
            pivot,
            annot=True, fmt=".0f",
            cmap="Blues",
            linewidths=0.5, linecolor=CARD_BG,
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Heatmap Curah Hujan (Bulan × Tahun)")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Bulan")
        ax.tick_params(colors=TEXT, labelsize=8)

        st.pyplot(fig, use_container_width=True)

        interpret(
            "Heatmap memperlihatkan variasi curah hujan setiap bulan lintas tahun. "
            "Warna <strong>biru gelap</strong> menandakan curah hujan sangat tinggi. "
            "Pola vertikal yang konsisten (bulan tertentu selalu gelap) mengonfirmasi "
            "<strong>musiman yang kuat</strong>. Pola horizontal (tahun tertentu lebih gelap) "
            "dapat mengindikasikan pengaruh fenomena iklim seperti La Niña atau El Niño."
        )
