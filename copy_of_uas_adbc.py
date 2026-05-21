import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pymannkendall as mk

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Dashboard Curah Hujan",
    layout="wide"
)

st.title("🌧 Dashboard Analisis Curah Hujan Jawa Barat")

# ======================
# BIGQUERY CONNECTION
# ======================

PROJECT_ID="st-project-488506"
DATASET_NAME="uas_adbc"
TABLE_NAME="dataset_uas"

FULL_TABLE=f"`{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}`"

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

client = bigquery.Client(
    credentials=credentials,
    project=PROJECT_ID
)

# ======================
# FUNCTION QUERY
# ======================

@st.cache_data
def run_query(query):

    return client.query(query).to_dataframe()


# ======================
# PREVIEW DATA
# ======================

query_preview=f"""
SELECT *
FROM {FULL_TABLE}
LIMIT 10
"""

df_preview=run_query(query_preview)

st.subheader("Preview Dataset")

st.dataframe(df_preview)

# ======================
# STRUKTUR DATA
# ======================

sql_struktur=f"""
SELECT
COUNT(*) AS jumlah_observasi,
COUNT(DISTINCT event_id) AS unique_event_id,
COUNT(DISTINCT district_city) AS jumlah_wilayah,
MIN(event_date) AS tanggal_awal,
MAX(event_date) AS tanggal_akhir
FROM {FULL_TABLE}
"""

df_struktur=run_query(sql_struktur)

st.subheader("Informasi Dataset")

col1,col2,col3=st.columns(3)

col1.metric(
    "Jumlah Observasi",
    int(df_struktur["jumlah_observasi"][0])
)

col2.metric(
    "Jumlah Wilayah",
    int(df_struktur["jumlah_wilayah"][0])
)

col3.metric(
    "Unique Event",
    int(df_struktur["unique_event_id"][0])
)

# ======================
# CLEAN DATA
# ======================

sql_bersih=f"""
WITH deduplicated AS (

SELECT
*,
ROW_NUMBER() OVER(
PARTITION BY event_id
ORDER BY event_date
) rn

FROM {FULL_TABLE}

WHERE
event_id IS NOT NULL
AND event_date IS NOT NULL
AND district_city IS NOT NULL
AND rainfall_mm>=0

)

SELECT
event_date,
INITCAP(TRIM(district_city)) district_city,
rainfall_mm
FROM deduplicated
WHERE rn=1
"""

df=run_query(sql_bersih)

df["event_date"]=pd.to_datetime(df["event_date"])

# ======================
# HISTOGRAM
# ======================

st.subheader("Distribusi Curah Hujan")

fig,ax=plt.subplots()

ax.hist(
    df["rainfall_mm"],
    bins=20
)

ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Frekuensi")

st.pyplot(fig)

# ======================
# TOP WILAYAH
# ======================

sql_wilayah=f"""
SELECT
district_city,
AVG(rainfall_mm) rata_rata

FROM {FULL_TABLE}

GROUP BY district_city
ORDER BY rata_rata DESC
LIMIT 10
"""

df_wilayah=run_query(sql_wilayah)

st.subheader(
"Top 10 Wilayah Curah Hujan Tertinggi"
)

fig,ax=plt.subplots(figsize=(10,5))

ax.barh(
    df_wilayah["district_city"],
    df_wilayah["rata_rata"]
)

ax.invert_yaxis()

st.pyplot(fig)

# ======================
# TREN BULANAN
# ======================

sql_tren=f"""
SELECT

DATE_TRUNC(event_date,MONTH)
AS bulan_dt,

ROUND(
AVG(rainfall_mm),2
)
AS rata_curah_hujan

FROM {FULL_TABLE}

GROUP BY bulan_dt
ORDER BY bulan_dt
"""

df_tren=run_query(sql_tren)

df_tren["bulan_dt"]=pd.to_datetime(
    df_tren["bulan_dt"]
)

x=np.arange(
    len(df_tren)
)

z=np.polyfit(
    x,
    df_tren["rata_curah_hujan"],
    1
)

st.subheader(
"Tren Curah Hujan"
)

fig,ax=plt.subplots(
figsize=(12,5)
)

ax.plot(
df_tren["bulan_dt"],
df_tren["rata_curah_hujan"],
marker='o'
)

ax.plot(
df_tren["bulan_dt"],
np.poly1d(z)(x),
linestyle="--"
)

ax.xaxis.set_major_formatter(
mdates.DateFormatter(
'%b\n%Y'
)
)

st.pyplot(fig)

# ======================
# MANN KENDALL
# ======================

series=df_tren[
"rata_curah_hujan"
].values

result=mk.original_test(
series
)

st.subheader(
"Hasil Uji Mann-Kendall"
)

st.write(
f"Tren : {result.trend}"
)

st.write(
f"p-value : {result.p:.4f}"
)

st.write(
f"Tau : {result.Tau:.4f}"
)

# ======================
# POLA MUSIMAN
# ======================

st.subheader("Pola Musiman Curah Hujan")

sql_musiman=f"""
SELECT
EXTRACT(MONTH FROM event_date) AS bulan,

FORMAT_DATE(
'%b',
DATE(
2024,
CAST(EXTRACT(MONTH FROM event_date) AS INT64),
1
)
) AS nama_bulan,

ROUND(
AVG(rainfall_mm),2
) AS rata_rata

FROM {FULL_TABLE}

WHERE rainfall_mm>=0

GROUP BY bulan,nama_bulan
ORDER BY bulan
"""

df_musiman=run_query(
sql_musiman
)

sql_heatmap=f"""
SELECT

EXTRACT(MONTH FROM event_date) AS bulan,

EXTRACT(YEAR FROM event_date) AS tahun,

ROUND(
AVG(rainfall_mm),
2
) AS rata_rata

FROM {FULL_TABLE}

WHERE rainfall_mm>=0

GROUP BY bulan,tahun
ORDER BY bulan,tahun
"""

df_heatmap=run_query(
sql_heatmap
)

col1,col2=st.columns(2)

with col1:

    fig,ax=plt.subplots(
    figsize=(8,4)
    )

    ax.bar(
        df_musiman["nama_bulan"],
        df_musiman["rata_rata"]
    )

    ax.set_title(
    "Rata-rata Curah Hujan per Bulan"
    )

    ax.tick_params(
        axis='x',
        rotation=45
    )

    st.pyplot(fig)


with col2:

    pivot=df_heatmap.pivot(
        index='bulan',
        columns='tahun',
        values='rata_rata'
    )

    fig,ax=plt.subplots(
    figsize=(8,4)
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax
    )

    ax.set_title(
    "Heatmap Curah Hujan"
    )

    st.pyplot(fig)


# ======================
# TOP WILAYAH BERDASARKAN JUMLAH KEJADIAN
# ======================

st.subheader(
"Top 10 Wilayah Berdasarkan Jumlah Kejadian"
)

query_top=f"""
SELECT

district_city,

COUNT(*)
AS total_kejadian

FROM {FULL_TABLE}

GROUP BY district_city

ORDER BY total_kejadian DESC

LIMIT 10
"""

top_wilayah=run_query(
query_top
)

fig,ax=plt.subplots(
figsize=(10,5)
)

sns.barplot(
data=top_wilayah,
x='total_kejadian',
y='district_city',
ax=ax
)

st.pyplot(fig)


# ======================
# TREN PER WILAYAH
# ======================

st.subheader(
"Tren Curah Hujan per Wilayah"
)

sql_tren_wil=f"""

WITH top_wilayah AS(

SELECT
INITCAP(
TRIM(district_city)
) wilayah

FROM {FULL_TABLE}

GROUP BY wilayah

ORDER BY COUNT(*) DESC

LIMIT 10
)

SELECT

DATE_TRUNC(
event_date,
MONTH
) AS bulan_dt,

INITCAP(
TRIM(t.district_city)
) wilayah,

ROUND(
AVG(
t.rainfall_mm
),
2
) AS rata_curah_hujan

FROM {FULL_TABLE} t

JOIN top_wilayah tw

ON INITCAP(
TRIM(
t.district_city
)
)=tw.wilayah

GROUP BY
bulan_dt,
wilayah

ORDER BY
bulan_dt,
wilayah
"""

df_tren_wil=run_query(
sql_tren_wil
)

df_tren_wil[
"bulan_dt"
]=pd.to_datetime(
df_tren_wil[
"bulan_dt"
]
)

fig,ax=plt.subplots(
figsize=(14,6)
)

for wilayah,data in df_tren_wil.groupby(
'wilayah'
):

    ax.plot(
        data['bulan_dt'],
        data['rata_curah_hujan'],
        label=wilayah
    )

ax.legend(
bbox_to_anchor=(1.02,1)
)

ax.xaxis.set_major_formatter(
mdates.DateFormatter(
'%b\n%Y'
)
)

st.pyplot(fig)
