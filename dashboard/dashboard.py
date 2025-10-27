import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import plotly.express as px
import os  # Diperlukan untuk path & exists

# ==============================
# 🔧 KONFIGURASI DASBOR
# ==============================
st.set_page_config(
    page_title="Dashboard Analisis Penyewaan Sepeda",
    layout="wide",
)
st.title("🚲 Dashboard Analisis Penyewaan Sepeda")

# ==============================
# 🗺️ KONSTANTA & LABEL
# ==============================
SEASON_MAP = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
WEATHER_LABELS = {1: 'Cerah/Berawan Tipis', 2: 'Berkabut/Mendung', 3: 'Hujan/Salju Ringan', 4: 'Hujan/Salju Lebat'}
WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 🎨 Palet warna satu-tone (biru lembut) sesuai #58A9E6
PALETTE = ["#1E7ACB", "#3795DE", "#58A9E6", "#84C4F0", "#B5E1FA"]

# ==============================
# 📁 LOAD DATA
# ==============================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/inayahayudeswita/bike-sharing-dashboard/main/dashboard/main_data.csv"
    df = pd.read_csv(url)
    df['dteday'] = pd.to_datetime(df['dteday'])
    day_df = df.drop_duplicates(subset=['dteday'])[['dteday', 'season', 'weathersit_x', 'cnt_x', 'Level_deman_x']]
    hour_df = df[['dteday', 'season', 'hr', 'weathersit_y', 'cnt_y', 'Level_deman_x']]
    # Normalisasi nama kolom agar konsisten dengan sisa kode
    day_df = day_df.rename(columns={'weathersit_x':'weathersit','cnt_x':'cnt','Level_deman_x':'level'})
    hour_df = hour_df.rename(columns={'weathersit_y':'weathersit','cnt_y':'cnt','Level_deman_x':'level'})
    # Kolom turunan untuk analitik
    day_df['weekday_name'] = day_df['dteday'].dt.day_name()
    hour_df['weekday_name'] = hour_df['dteday'].dt.day_name()
    return day_df, hour_df

# ==============================
# ⚙️ LOAD SCALER DAN MODEL (Lokal, satu folder dengan dashboard.py)
# ==============================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 📝 Daftar path file yang diperlukan (sesuai nama file unggahan)
    file_paths = {
        "scaler_day": os.path.join(BASE_DIR, "day_scaler.pkl"),
        "scaler_hour": os.path.join(BASE_DIR, "hour_scaler.pkl"),
        "model_day": os.path.join(BASE_DIR, "rf_day_model.pkl"),
        "model_hour": os.path.join(BASE_DIR, "rf_hour_model.pkl")
    }
    
    loaded_files = {}
    
    # Loop untuk memeriksa keberadaan file dan memuatnya
    for key, path in file_paths.items():
        if not os.path.exists(path):
            # Jika file tidak ada, hentikan aplikasi dan beri pesan error
            file_name = os.path.basename(path)
            raise FileNotFoundError(
                f"File **{file_name}** tidak ditemukan. "
                f"Pastikan file tersebut berada di folder yang sama dengan dashboard.py."
            )
        # Muat file jika ada
        loaded_files[key] = joblib.load(path)

    # Mengembalikan Scaler dan Model
    return (
        loaded_files["scaler_day"],
        loaded_files["scaler_hour"],
        loaded_files["model_day"],
        loaded_files["model_hour"]
    )

# ==============================
# 🔄 INIT DATA & MODEL
# ==============================
try:
    # 📝 Menangkap 4 output dari load_models()
    scaler_day, scaler_hour, model_day, model_hour = load_models()
    day_df, hour_df = load_data()
    st.sidebar.success("✅ Data, Scaler, dan Model berhasil dimuat!")
    data_loaded = True
except Exception as e:
    st.error(f"Error saat memuat data, scaler atau model: {e}")
    st.info("Pastikan semua file (`day_scaler.pkl`, `hour_scaler.pkl`, `rf_day_model.pkl`, `rf_hour_model.pkl`) tersedia di folder yang sama dengan dashboard.py.")
    data_loaded = False

# ==============================
# 🧭 FILTER (SIDEBAR)
# ==============================
if data_loaded:
    with st.sidebar:
        st.header("Filter")
        min_date, max_date = day_df["dteday"].min().date(), day_df["dteday"].max().date()
        start_date, end_date = st.date_input(
            "Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        seasons = st.multiselect(
            "Musim",
            options=sorted(SEASON_MAP.keys()),
            default=sorted(SEASON_MAP.keys()),
            format_func=lambda x: SEASON_MAP[x],
        )
        weathers = st.multiselect(
            "Cuaca",
            options=sorted(day_df["weathersit"].unique()),
            default=sorted(day_df["weathersit"].unique()),
            format_func=lambda x: WEATHER_LABELS.get(x, str(x)),
        )

    mask_day = (
        (day_df["dteday"].dt.date >= start_date)
        & (day_df["dteday"].dt.date <= end_date)
        & (day_df["season"].isin(seasons))
        & (day_df["weathersit"].isin(weathers))
    )
    mask_hour = (
        (hour_df["dteday"].dt.date >= start_date)
        & (hour_df["dteday"].dt.date <= end_date)
        & (hour_df["season"].isin(seasons))
        & (hour_df["weathersit"].isin(weathers))
    )
    md = day_df.loc[mask_day].copy()
    mh = hour_df.loc[mask_hour].copy()


# ==============================
# 🧩 TABS
# ==============================
if data_loaded:
    tab1, tab2 = st.tabs(["📊 Analitik Data", "🤖 Prediksi"])

    # =========================
    # 📊 TAB 1 - RINGKASAN DATA
    # =========================
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Average Rentals by Weekday (horizontal bar)")
            weekday_avg = md.groupby("weekday_name", as_index=False)["cnt"].mean()
            weekday_avg["weekday_name"] = pd.Categorical(
                weekday_avg["weekday_name"], categories=WEEKDAY_ORDER, ordered=True
            )
            weekday_avg = weekday_avg.sort_values("weekday_name", ascending=True)
            fig_wd = px.bar(
                weekday_avg,
                y="weekday_name",
                x="cnt",
                orientation="h",
                labels={"weekday_name": "Hari", "cnt": "Rata-rata Penyewaan"},
                template="plotly_white",
                title="Rata-rata Penyewaan per Hari (Horizontal)",
                color_discrete_sequence=PALETTE,
            )
            fig_wd.update_layout(
                yaxis=dict(categoryorder="array", categoryarray=WEEKDAY_ORDER),
                margin=dict(l=80, r=40, t=60, b=40),
            )
            st.plotly_chart(fig_wd, use_container_width=True, theme=None)

        with col2:
            st.markdown("### Hourly Rental Trend (bar)")
            hourly_avg = mh.groupby("hr", as_index=False)["cnt"].mean().sort_values("hr")
            fig_hr = px.bar(
                hourly_avg,
                x="hr",
                y="cnt",
                template="plotly_white",
                title="Rata-rata Penyewaan per Jam",
                color_discrete_sequence=PALETTE,
            )
            fig_hr.update_xaxes(dtick=1, tick0=0)
            st.plotly_chart(fig_hr, use_container_width=True, theme=None)

        # Monthly Trend
        st.markdown("### Monthly Rental Trend (line)")
        monthly = md.set_index("dteday")["cnt"].resample("M").sum().reset_index()
        monthly["period"] = monthly["dteday"].dt.to_period("M").astype(str)
        fig_month = px.line(
            monthly,
            x="period",
            y="cnt",
            markers=True,
            title="Monthly Rental Trend",
            template="plotly_white",
            color_discrete_sequence=PALETTE,
        )
        fig_month.update_xaxes(tickangle=45)
        st.plotly_chart(fig_month, use_container_width=True, theme=None)

        # Heatmap Weekday × Hour
        st.markdown("### Heatmap — Avg Rentals (Weekday × Hour)")
        mh_disp = mh.copy()
        mh_disp["Hari"] = mh_disp["weekday_name"]
        heat = mh_disp.pivot_table(index="Hari", columns="hr", values="cnt", aggfunc="mean")
        heat = heat.reindex(WEEKDAY_ORDER)  # pastikan 7 hari
        fig_heat = px.imshow(
            heat,
            aspect="auto",
            origin="lower",
            labels=dict(x="Jam", y="Hari", color="Avg cnt"),
            title="Heatmap Rata-rata Penyewaan (Jam × Hari)",
            color_continuous_scale=PALETTE,
        )
        st.plotly_chart(fig_heat, use_container_width=True, theme=None)

        # Weather & Season
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Rentals by Weather (day)")
            wday = (
                md.groupby("weathersit", as_index=False)["cnt"].sum()
                  .assign(Cuaca=lambda d: d["weathersit"].map(WEATHER_LABELS))
            )
            fig_wday = px.bar(
                wday,
                x="Cuaca",
                y="cnt",
                template="plotly_white",
                title="Penyewaan berdasarkan Cuaca (day)",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(fig_wday, use_container_width=True, theme=None)
        with col4:
            st.markdown("### Rentals by Season (day)")
            sday = (
                md.groupby("season", as_index=False)["cnt"].sum()
                  .assign(Musim=lambda d: d["season"].map(SEASON_MAP))
            )
            fig_sday = px.pie(
                sday,
                names="Musim",
                values="cnt",
                hole=0.35,
                title="Distribusi Penyewaan berdasarkan Musim (day)",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(fig_sday, use_container_width=True, theme=None)

        st.caption(
            "Filter bekerja untuk semua visual. Bulan menggunakan resample('M'); Weekday 7 hari; Hourly bar; Heatmap lengkap; semua grafik satu-tone merah."
        )

    # =========================
    # 🤖 TAB 2 - PREDIKSI
    # =========================
    with tab2:
        st.header("Prediksi Jumlah Penyewaan Sepeda")
        mode = st.radio("Pilih Mode Prediksi", ["Harian (day)", "Per Jam (hour)"])

        if mode == "Harian (day)":
            st.subheader("Masukkan Parameter Prediksi (Day)")
            col1, col2, col3 = st.columns(3)
            with col1:
                season = st.selectbox("Musim", [1, 2, 3, 4], format_func=lambda x: SEASON_MAP[x])
                yr = st.selectbox("Tahun (0=2011, 1=2012)", [0, 1])
                mnth = st.slider("Bulan", 1, 12, 6)
            with col2:
                holiday = st.selectbox("Hari Libur", [0, 1])
                weekday = st.slider("Hari ke-", 0, 6, 3)
                workingday = st.selectbox("Hari Kerja", [0, 1])
            with col3:
                weathersit = st.selectbox("Cuaca", [1, 2, 3, 4], format_func=lambda x: {1:'Cerah',2:'Berawan',3:'Hujan',4:'Badai'}[x])
                temp = st.slider("Suhu", 0.0, 1.0, 0.5)
                atemp = st.slider("Suhu Terasa", 0.0, 1.0, 0.5)
                hum = st.slider("Kelembaban", 0.0, 1.0, 0.5)
                windspeed = st.slider("Kecepatan Angin", 0.0, 1.0, 0.2)
            
            # Daftar fitur yang digunakan oleh model harian
            day_features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

            if st.button("Prediksi Jumlah Penyewaan (Day)"):
                input_data = pd.DataFrame([
                    {
                        'season': season, 'yr': yr, 'mnth': mnth, 'holiday': holiday,
                        'weekday': weekday, 'workingday': workingday, 'weathersit': weathersit,
                        'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed,
                    }
                ])
                
                # 📝 LANGKAH PENSKALAAN: Skala data input sebelum prediksi
                input_scaled = scaler_day.transform(input_data[day_features])
                input_scaled_df = pd.DataFrame(input_scaled, columns=day_features)

                # Prediksi menggunakan data yang sudah diskalakan
                pred_day = model_day.predict(input_scaled_df)[0]
                st.success(f"🔮 Prediksi Jumlah Penyewaan Harian: **{int(pred_day):,} sepeda**")

        else:
            st.subheader("Masukkan Parameter Prediksi (Hour)")
            col1, col2, col3 = st.columns(3)
            with col1:
                season = st.selectbox("Musim", [1, 2, 3, 4], key="season_hour", format_func=lambda x: SEASON_MAP[x])
                yr = st.selectbox("Tahun", [0, 1], key="yr_hour")
                mnth = st.slider("Bulan", 1, 12, 6, key="mnth_hour")
            with col2:
                hr = st.slider("Jam", 0, 23, 12, key="hr_hour")
                holiday = st.selectbox("Hari Libur", [0, 1], key="hol_hour")
                weekday = st.slider("Hari ke-", 0, 6, 3, key="weekday_hour")
                workingday = st.selectbox("Hari Kerja", [0, 1], key="work_hour")
            with col3:
                weathersit = st.selectbox("Cuaca", [1, 2, 3, 4], key="weather_hour", format_func=lambda x: {1:'Cerah',2:'Berawan',3:'Hujan',4:'Badai'}[x])
                temp = st.slider("Suhu", 0.0, 1.0, 0.5, key="temp_hour")
                atemp = st.slider("Suhu Terasa", 0.0, 1.0, 0.5, key="atemp_hour")
                hum = st.slider("Kelembaban", 0.0, 1.0, 0.5, key="hum_hour")
                windspeed = st.slider("Kecepatan Angin", 0.0, 1.0, 0.2, key="wind_hour")

            # Daftar fitur yang digunakan oleh model per jam
            hour_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

            if st.button("Prediksi Jumlah Penyewaan (Hour)"):
                input_data = pd.DataFrame([
                    {
                        'season': season, 'yr': yr, 'mnth': mnth, 'hr': hr,
                        'holiday': holiday, 'weekday': weekday, 'workingday': workingday,
                        'weathersit': weathersit, 'temp': temp, 'atemp': atemp,
                        'hum': hum, 'windspeed': windspeed,
                    }
                ])

                # 📝 LANGKAH PENSKALAAN: Skala data input sebelum prediksi
                input_scaled = scaler_hour.transform(input_data[hour_features])
                input_scaled_df = pd.DataFrame(input_scaled, columns=hour_features)
                
                # Prediksi menggunakan data yang sudah diskalakan
                pred_hour = model_hour.predict(input_scaled_df)[0]
                st.success(f"🔮 Prediksi Jumlah Penyewaan Per Jam: **{int(pred_hour):,} sepeda**")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(["Prediksi"], [pred_hour], color=PALETTE[2])
                ax.set_ylabel("Jumlah Penyewaan")
                ax.set-title("Visualisasi Hasil Prediksi")
                st.pyplot(fig)