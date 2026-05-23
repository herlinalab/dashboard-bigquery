import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pymannkendall as mk

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dashboard Curah Hujan Jawa Barat",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CSS — LIGHT THEME + LOCKED SIDEBAR
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Palette ─────────────────────────────── */
:root {
    --bg:          #f0f6ff;
    --bg-card:     #ffffff;
    --bg-card2:    #f8faff;
    --sidebar-bg:  #1a56db;
    --sidebar-top: #1e40af;
    --accent:      #1a56db;
    --accent2:     #0ea5e9;
    --accent3:     #06b6d4;
    --success:     #059669;
    --warning:     #d97706;
    --danger:      #dc2626;
    --text:        #1e293b;
    --text-mid:    #475569;
    --text-muted:  #94a3b8;
    --border:      #dbeafe;
    --shadow:      0 2px 12px rgba(30,86,219,.10);
}

/* ── Base ─────────────────────────────────── */
html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ────────────────── */
header[data-testid="stHeader"]          { display: none !important; }
footer                                   { display: none !important; }
#MainMenu                                { display: none !important; }

/* ── LOCK SIDEBAR — hide collapse button ─── */
button[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Keep sidebar always visible & styled */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    min-width: 240px !important;
    max-width: 240px !important;
    width:     240px !important;
    transform: none !important;
    visibility: visible !important;
    border-right: none !important;
    box-shadow: 3px 0 16px rgba(30,64,175,.15);
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* Sidebar text colours */
section[data-testid="stSidebar"] * {
    color: #e0eaff !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #c7d9ff !important;
    font-size: .9rem !important;
    font-weight: 500 !important;
    padding: 6px 0 !important;
}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #c7d9ff !important;
}

/* Radio active item highlight */
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(255,255,255,.15) !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
}

/* ── Metric cards ─────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: var(--shadow);
    transition: box-shadow .2s, border-color .2s;
}
div[data-testid="metric-container"]:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 20px rgba(30,86,219,.15);
}
div[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: .75rem !important;
    font-weight: 700 !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
}

/* ── Section title ───────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin: 28px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, var(--border), transparent);
    margin-left: 8px;
}

/* ── Interpretation box ──────────────────── */
.interpretation-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border: 1px solid #bfdbfe;
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: .875rem;
    line-height: 1.75;
    color: #334155;
}
.interpretation-box strong { color: var(--accent); }

/* ── DataFrames ──────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Divider ─────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Nav label ───────────────────────────── */
.nav-label {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: rgba(255,255,255,.5) !important;
    padding: 18px 4px 6px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB LIGHT THEME
# ============================================================

PLOT_BG   = "#ffffff"
PLOT_BG2  = "#f8faff"
BLUE      = "#1a56db"
CYAN      = "#0ea5e9"
TEAL      = "#0d9488"
AMBER     = "#d97706"
ROSE      = "#e11d48"
VIOLET    = "#7c3aed"
LIME      = "#65a30d"
ORANGE    = "#ea580c"
TEXT_PL   = "#1e293b"
MUTED_PL  = "#64748b"
GRID_PL   = "#e2e8f0"

PALETTE = [BLUE, CYAN, TEAL, AMBER, ROSE, VIOLET, LIME, ORANGE, "#0891b2", "#9333ea"]


def apply_light_style(fig, ax):
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG2)
    ax.tick_params(colors=MUTED_PL, labelsize=9)
    ax.xaxis.label.set_color(MUTED_PL)
    ax.yaxis.label.set_color(MUTED_PL)
    ax.title.set_color(TEXT_PL)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_PL)
    ax.grid(color=GRID_PL, linewidth=0.7, linestyle="--", alpha=0.9)
    ax.set_axisbelow(True)


# ============================================================
# BIGQUERY CONNECTION
# ============================================================

PROJECT_ID   = "st-project-488506"
DATASET_NAME = "uas_adbc"
TABLE_NAME   = "dataset_bersih"
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
# SIDEBAR — ALWAYS VISIBLE
# ============================================================

with st.sidebar:
    # Header
    st.markdown("""
    <div style="background:rgba(0,0,0,.18);padding:22px 16px 18px;
                margin:-16px -16px 0;border-bottom:1px solid rgba(255,255,255,.15);">
        <div style="font-size:2rem;line-height:1;">🌧️</div>
        <div style="font-size:1rem;font-weight:800;color:#fff;margin-top:6px;
                    letter-spacing:.01em;line-height:1.3;">
            Dashboard<br>Curah Hujan
        </div>
        <div style="font-size:.72rem;color:rgba(255,255,255,.6);margin-top:4px;">
            Jawa Barat · Analisis Hidrologis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Menu Navigasi</div>', unsafe_allow_html=True)

    menu = st.radio(
        label="menu",
        options=["📊  Data", "📈  Analisis Tren", "🗺️  Visualisasi"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.7rem;color:rgba(255,255,255,.4);line-height:1.6;padding:0 2px;">
        Sumber: BigQuery · BMKG Jawa Barat<br>
        Model: Mann-Kendall
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATA QUERIES (cached)
# ============================================================

@st.cache_data(ttl=600)
def load_clean_data():
    sql = f"""
    WITH dedup AS (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY event_id ORDER BY event_date) rn
        FROM {FULL_TABLE}
        WHERE event_id IS NOT NULL AND event_date IS NOT NULL
          AND district_city IS NOT NULL AND rainfall_mm >= 0
    )
    SELECT event_date, INITCAP(TRIM(district_city)) AS district_city, rainfall_mm
    FROM dedup WHERE rn = 1
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
               ROUND(AVG(rainfall_mm), 2) AS rata_curah_hujan
        FROM {FULL_TABLE} GROUP BY bulan_dt ORDER BY bulan_dt
    """)
    df["bulan_dt"] = pd.to_datetime(df["bulan_dt"])
    return df

@st.cache_data(ttl=600)
def load_wilayah_avg():
    return run_query(f"""
        SELECT INITCAP(TRIM(district_city)) AS district_city,
               ROUND(AVG(rainfall_mm), 2) AS rata_rata
        FROM {FULL_TABLE} GROUP BY district_city ORDER BY rata_rata DESC LIMIT 10
    """)

@st.cache_data(ttl=600)
def load_wilayah_count():
    return run_query(f"""
        SELECT INITCAP(TRIM(district_city)) AS district_city,
               COUNT(*) AS total_kejadian
        FROM {FULL_TABLE} GROUP BY district_city ORDER BY total_kejadian DESC LIMIT 10
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
               EXTRACT(YEAR FROM event_date) AS tahun,
               ROUND(AVG(rainfall_mm), 2) AS rata_rata
        FROM {FULL_TABLE} WHERE rainfall_mm >= 0
        GROUP BY bulan, tahun ORDER BY bulan, tahun
    """)

@st.cache_data(ttl=600)
def load_tren_wilayah():
    df = run_query(f"""
        WITH top_wil AS (
            SELECT INITCAP(TRIM(district_city)) AS wilayah
            FROM {FULL_TABLE} GROUP BY wilayah ORDER BY COUNT(*) DESC LIMIT 10
        )
        SELECT DATE_TRUNC(event_date, MONTH) AS bulan_dt,
               INITCAP(TRIM(t.district_city)) AS wilayah,
               ROUND(AVG(t.rainfall_mm), 2) AS rata_curah_hujan
        FROM {FULL_TABLE} t
        JOIN top_wil tw ON INITCAP(TRIM(t.district_city)) = tw.wilayah
        GROUP BY bulan_dt, wilayah ORDER BY bulan_dt, wilayah
    """)
    df["bulan_dt"] = pd.to_datetime(df["bulan_dt"])
    return df


# ============================================================
# HELPERS
# ============================================================

def section(icon_title: str):
    st.markdown(f'<div class="section-title">{icon_title}</div>', unsafe_allow_html=True)

def interpret(text: str):
    st.markdown(
        f'<div class="interpretation-box">💡 <strong>Interpretasi:</strong> {text}</div>',
        unsafe_allow_html=True
    )

def page_header(title: str, subtitle: str):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a56db 0%,#0ea5e9 100%);
                border-radius:16px;padding:28px 32px;margin-bottom:28px;
                box-shadow:0 4px 24px rgba(30,86,219,.2);">
        <h1 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.75rem;
                   font-weight:800;color:#fff;margin:0 0 6px;">{title}</h1>
        <p style="color:rgba(255,255,255,.75);margin:0;font-size:.9rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE: DATA
# ============================================================

if menu == "📊  Data":

    page_header("📊 Ringkasan Dataset",
                "Tinjauan menyeluruh dataset curah hujan Jawa Barat yang tersimpan di BigQuery.")

    df_str = load_struktur()
    tgl_awal  = pd.to_datetime(df_str["tanggal_awal"][0]).strftime("%d %b %Y")
    tgl_akhir = pd.to_datetime(df_str["tanggal_akhir"][0]).strftime("%d %b %Y")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🗂️ Total Observasi",  f'{int(df_str["jumlah_observasi"][0]):,}')
    c2.metric("📍 Jumlah Wilayah",   int(df_str["jumlah_wilayah"][0]))
    c3.metric("🔑 Unique Event ID",  int(df_str["unique_event_id"][0]))
    c4.metric("📅 Tahun Data",       f"{tgl_awal} – {tgl_akhir}")

    # Preview
    section("🔍 Preview Data (10 baris pertama)")
    df_preview = run_query(f"SELECT * FROM {FULL_TABLE} LIMIT 10")
    st.dataframe(df_preview, use_container_width=True)

    # Statistik deskriptif
    section("📐 Statistik Deskriptif Curah Hujan")
    df = load_clean_data()
    desc = df["rainfall_mm"].describe().rename({
        "count": "Jumlah Data", "mean": "Rata-rata (mm)",
        "std": "Std. Deviasi",  "min": "Minimum",
        "25%": "Kuartil 1 (25%)", "50%": "Median (50%)",
        "75%": "Kuartil 3 (75%)", "max": "Maksimum",
    })
    st.dataframe(desc.to_frame("Nilai").style.format("{:.2f}"), use_container_width=True)

    interpret(
        f"Dataset mencakup <strong>{int(df_str['jumlah_observasi'][0]):,}</strong> observasi dari "
        f"<strong>{int(df_str['jumlah_wilayah'][0])}</strong> wilayah di Jawa Barat "
        f"(rentang {tgl_awal} – {tgl_akhir}). "
        f"Rata-rata curah hujan <strong>{df['rainfall_mm'].mean():.1f} mm</strong> dengan "
        f"nilai maksimum <strong>{df['rainfall_mm'].max():.1f} mm</strong>, "
        f"menunjukkan variasi yang tinggi antar-kejadian."
    )

    # Distribusi
    section("📊 Distribusi Frekuensi Curah Hujan")

    fig, ax = plt.subplots(figsize=(12, 4))
    apply_light_style(fig, ax)

    n, bins, patches = ax.hist(df["rainfall_mm"], bins=30,
                                edgecolor="white", linewidth=0.4)
    # Gradient coloring
    cmap = plt.cm.Blues
    for i, p in enumerate(patches):
        p.set_facecolor(cmap(0.35 + 0.6 * i / len(patches)))

    ax.axvline(df["rainfall_mm"].mean(),   color=AMBER, lw=1.6, ls="--",
               label=f'Mean: {df["rainfall_mm"].mean():.1f} mm')
    ax.axvline(df["rainfall_mm"].median(), color=ROSE,  lw=1.6, ls=":",
               label=f'Median: {df["rainfall_mm"].median():.1f} mm')
    ax.set_xlabel("Curah Hujan (mm)")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi Frekuensi Curah Hujan")
    ax.legend(fontsize=9)

    st.pyplot(fig, use_container_width=True)

    skew_val = df["rainfall_mm"].skew()
    interpret(
        f"Distribusi bersifat <strong>skewed {'kanan (positif)' if skew_val>0 else 'kiri (negatif)'}</strong> "
        f"(skewness = {skew_val:.2f}). "
        f"Mayoritas kejadian memiliki curah hujan rendah–sedang, namun terdapat ekor panjang ke kanan "
        f"akibat kejadian ekstrem. "
        f"Nilai <strong>mean ({df['rainfall_mm'].mean():.1f} mm) > median ({df['rainfall_mm'].median():.1f} mm)</strong> "
        f"mengonfirmasi distribusi tidak simetris."
    )


# ============================================================
# PAGE: ANALISIS TREN
# ============================================================

elif menu == "📈  Analisis Tren":

    page_header("📈 Analisis Tren",
                "Tren temporal curah hujan bulanan dan uji statistik Mann-Kendall.")

    df_tren = load_tren()
    x = np.arange(len(df_tren))
    z = np.polyfit(x, df_tren["rata_curah_hujan"], 1)
    p = np.poly1d(z)
    result = mk.original_test(df_tren["rata_curah_hujan"].values)
    trend_label = result.trend
    badge_text = "⬆ Meningkat" if trend_label == "increasing" else \
                 ("⬇ Menurun" if trend_label == "decreasing" else "➡ Stabil")

    # Tren bulanan
    section("📉 Tren Curah Hujan Bulanan")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    apply_light_style(fig, ax)

    ax.fill_between(df_tren["bulan_dt"], df_tren["rata_curah_hujan"],
                    alpha=0.12, color=BLUE)
    ax.plot(df_tren["bulan_dt"], df_tren["rata_curah_hujan"],
            color=BLUE, lw=2, marker="o", markersize=4, label="Rata-rata Bulanan")
    ax.plot(df_tren["bulan_dt"], p(x),
            color=ROSE, lw=2.2, ls="--", label=f"Tren Linear (slope={z[0]:+.2f})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.set_ylabel("Curah Hujan (mm)")
    ax.set_title("Tren Rata-rata Curah Hujan Bulanan")
    ax.legend(fontsize=9)

    st.pyplot(fig, use_container_width=True)

    interpret(
        f"Garis tren linear memiliki slope <strong>{z[0]:+.4f} mm/bulan</strong>, "
        f"menandakan tren <strong>{'meningkat' if z[0]>0 else 'menurun'}</strong> secara keseluruhan. "
        f"Fluktuasi bulanan yang besar mencerminkan pola musiman yang kuat di wilayah Jawa Barat."
    )

    # Mann-Kendall
    section("🧪 Uji Statistik Mann-Kendall")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Arah Tren",     badge_text)
    m2.metric("p-value",       f"{result.p:.4f}")
    m3.metric("Tau (τ)",       f"{result.Tau:.4f}")
    m4.metric("S statistic",   f"{result.s}")

    sig = result.p < 0.05
    interpret(
        f"Uji Mann-Kendall mengindikasikan tren <strong>{trend_label}</strong> "
        f"dengan p = <strong>{result.p:.4f}</strong>. "
        f"{'✅ <strong>Signifikan secara statistik</strong> (p < 0,05) — perubahan bersifat nyata, bukan acak.' if sig else '⚠️ <strong>Tidak signifikan secara statistik</strong> (p ≥ 0,05) — masih dalam batas variabilitas alami.'} "
        f"Nilai Tau = <strong>{result.Tau:.4f}</strong> menunjukkan "
        f"{'korelasi positif' if result.Tau>0 else 'korelasi negatif'} antara waktu dan curah hujan."
    )

    # Tren per wilayah
    section("🗺️ Tren per Wilayah (Top 10)")

    df_tw = load_tren_wilayah()

    fig, ax = plt.subplots(figsize=(13, 5))
    apply_light_style(fig, ax)

    for i, (wil, data) in enumerate(df_tw.groupby("wilayah")):
        ax.plot(data["bulan_dt"], data["rata_curah_hujan"],
                color=PALETTE[i % len(PALETTE)], lw=1.8, alpha=0.9, label=wil)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.set_ylabel("Curah Hujan (mm)")
    ax.set_title("Tren Curah Hujan 10 Wilayah Teratas")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.95, edgecolor=GRID_PL, fontsize=8)

    st.pyplot(fig, use_container_width=True)

    interpret(
        "Tren per wilayah memperlihatkan <strong>heterogenitas spasial</strong> yang nyata. "
        "Beberapa wilayah menunjukkan curah hujan stabil, sementara yang lain berfluktuasi ekstrem. "
        "Perbedaan ini dipengaruhi oleh topografi, jarak dari laut, dan kondisi lokal masing-masing daerah."
    )


# ============================================================
# PAGE: VISUALISASI
# ============================================================

elif menu == "🗺️  Visualisasi":

    page_header("🗺️ Visualisasi Spasial & Musiman",
                "Distribusi curah hujan antar wilayah dan pola musiman lintas tahun.")

    # Top wilayah (rata-rata)
    section("🏆 Top 10 Wilayah — Rata-rata Curah Hujan Tertinggi")
    df_wa = load_wilayah_avg()

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_light_style(fig, ax)

    colors_wa = [PALETTE[i % len(PALETTE)] for i in range(len(df_wa))]
    bars = ax.barh(df_wa["district_city"], df_wa["rata_rata"],
                   color=colors_wa, edgecolor="white", linewidth=0.5, height=0.65)
    ax.invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f} mm", va="center", ha="left", color=MUTED_PL, fontsize=8.5)
    ax.set_xlabel("Rata-rata Curah Hujan (mm)")
    ax.set_title("Top 10 Wilayah Curah Hujan Tertinggi")
    ax.set_xlim(0, df_wa["rata_rata"].max() * 1.15)

    st.pyplot(fig, use_container_width=True)

    top1 = df_wa.iloc[0]
    interpret(
        f"<strong>{top1['district_city']}</strong> mencatat rata-rata curah hujan tertinggi "
        f"sebesar <strong>{top1['rata_rata']:.1f} mm</strong>. "
        f"Wilayah dengan curah hujan tinggi umumnya berada di zona pegunungan atau berhadapan "
        f"langsung dengan angin monsun, penting untuk perencanaan drainase dan mitigasi banjir."
    )

    # Top wilayah (frekuensi)
    section("📊 Top 10 Wilayah — Frekuensi Kejadian Hujan")
    df_wc = load_wilayah_count()

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_light_style(fig, ax)

    clr_wc = [TEAL if i < 3 else "#93c5fd" for i in range(len(df_wc))]
    ax.barh(df_wc["district_city"], df_wc["total_kejadian"],
            color=clr_wc, edgecolor="white", linewidth=0.5, height=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("Jumlah Kejadian")
    ax.set_title("Frekuensi Kejadian Hujan per Wilayah")
    for i, (_, row) in enumerate(df_wc.iterrows()):
        ax.text(row["total_kejadian"] + 1, i,
                str(int(row["total_kejadian"])), va="center", color=MUTED_PL, fontsize=8.5)

    st.pyplot(fig, use_container_width=True)

    interpret(
        f"Wilayah dengan frekuensi kejadian terbanyak adalah "
        f"<strong>{df_wc.iloc[0]['district_city']}</strong> "
        f"({int(df_wc.iloc[0]['total_kejadian']):,} kejadian). "
        f"Perlu dibedakan antara wilayah berintensitas tinggi (mm besar) dan berfrekuensi tinggi (sering hujan). "
        f"Wilayah dominan di kedua kategori memerlukan perhatian khusus dalam manajemen risiko."
    )

    st.markdown("---")

    # Pola musiman
    section("🌦️ Pola Musiman Curah Hujan")
    df_mus  = load_musiman()
    df_heat = load_heatmap()

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        apply_light_style(fig, ax)

        bar_clr = [BLUE if v >= df_mus["rata_rata"].mean() else "#93c5fd"
                   for v in df_mus["rata_rata"]]
        ax.bar(df_mus["nama_bulan"], df_mus["rata_rata"],
               color=bar_clr, edgecolor="white", linewidth=0.5)
        mean_val = df_mus["rata_rata"].mean()
        ax.axhline(mean_val, color=AMBER, lw=1.4, ls="--",
                   label=f"Rata-rata: {mean_val:.1f} mm")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Curah Hujan (mm)")
        ax.set_title("Rata-rata Curah Hujan per Bulan")
        ax.legend(fontsize=8)

        st.pyplot(fig, use_container_width=True)

        peak_m = df_mus.loc[df_mus["rata_rata"].idxmax(), "nama_bulan"]
        dry_m  = df_mus.loc[df_mus["rata_rata"].idxmin(), "nama_bulan"]
        interpret(
            f"Puncak curah hujan terjadi pada bulan <strong>{peak_m}</strong> dan "
            f"terendah pada <strong>{dry_m}</strong>. Batang berwarna biru tua menandakan "
            f"bulan-bulan di atas rata-rata — musim hujan dominan Jawa Barat."
        )

    with col2:
        pivot = df_heat.pivot(index="bulan", columns="tahun", values="rata_rata")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG2)

        sns.heatmap(
            pivot, annot=True, fmt=".0f",
            cmap="YlOrRd",
            linewidths=0.4, linecolor="#f8faff",
            ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Heatmap Curah Hujan (Bulan × Tahun)")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Bulan")
        ax.tick_params(labelsize=8)

        st.pyplot(fig, use_container_width=True)

        interpret(
            "Warna <strong>merah/oranye gelap</strong> menandakan curah hujan sangat tinggi. "
            "Pola vertikal konsisten (bulan tertentu selalu gelap) mengonfirmasi <strong>musiman kuat</strong>. "
            "Pola horizontal (tahun tertentu lebih gelap) dapat mengindikasikan pengaruh "
            "La Niña atau El Niño pada tahun tersebut."
        )
