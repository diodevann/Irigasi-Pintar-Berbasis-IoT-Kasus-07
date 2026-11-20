from __future__ import annotations

import os
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import AMBANG_AGRONOMI, build_single_feature_row, validate_single_input
from src.train import MODEL_PATH, train_and_save_artifacts

st.set_page_config(
    page_title="Smart Irrigation IoT - Greenhouse Lembang",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.environ.get("IRRIGATION_API_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), MODEL_PATH.name
        except Exception:
            pass
    with st.spinner("Menyelaraskan artifact model dengan versi scikit-learn lokal..."):
        return train_and_save_artifacts(), MODEL_PATH.name


@st.cache_data
def load_sample_data():
    sample_path = PROJECT_ROOT / "data" / "sample_greenhouse_16petak.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return None


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False


def run_inference(payload: dict, bundel: dict, use_api: bool) -> dict:
    if use_api:
        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=3.0)
            data = resp.json()
            data["source"] = f"FastAPI ({API_BASE_URL}/predict)"
            data["http_status"] = resp.status_code
            return data
        except Exception:
            pass

    is_valid, errors, cleaned = validate_single_input(payload)
    if not is_valid:
        return {
            "status": "error",
            "message": "Masukan sensor tidak valid atau di luar rentang fisik.",
            "errors": errors,
            "source": "src/features.py",
            "http_status": 422,
        }

    df_row, summary = build_single_feature_row(cleaned)
    prob_rf = float(bundel["model_rf"].predict_proba(df_row)[0, 1])
    prob_hgb = float(bundel["model_hgb"].predict_proba(df_row)[0, 1])

    if cleaned["model_type"] == "ensemble":
        prob_ml = 0.5 * prob_rf + 0.5 * prob_hgb
        threshold = float(bundel["threshold_ensemble"])
        model_used = "Ensemble RF + HistGB"
    else:
        prob_ml = prob_rf
        threshold = float(bundel["threshold_rf"])
        model_used = "Random Forest"

    keputusan_ml = int(prob_ml >= threshold)
    keputusan_baseline = int(float(cleaned["kelembaban_tanah_pct"]) < 35.0)

    return {
        "status": "success",
        "model_used": model_used,
        "model_version": bundel.get("model_version", "1.0.0"),
        "keputusan_ml": keputusan_ml,
        "label_ml": "SIRAM (1)" if keputusan_ml == 1 else "TIDAK PERLU DISIRAM (0)",
        "probabilitas_ml": round(prob_ml, 4),
        "threshold": threshold,
        "keputusan_baseline": keputusan_baseline,
        "label_baseline": "SIRAM (1)" if keputusan_baseline == 1 else "JANGAN SIRAM (0)",
        "ringkasan_agronomi": summary,
        "source": "artifacts/model.joblib",
        "http_status": 200,
    }


bundel, artifact_name = load_model_bundle()
df_sample = load_sample_data()
api_active = check_api_health()

st.title("Sistem Keputusan Irigasi Pintar Berbasis IoT (Kasus 07)")
st.markdown(
    "**Klien:** Pengelola Rumah Kaca (*Greenhouse*) Sayuran di Lembang (16 Petak: `P01` s.d. `P16`)  \n"
    "Sistem ini memprediksi secara *real-time* setiap jam apakah suatu petak **perlu disiram sekarang (`1`) atau tidak (`0`)**, "
    "dengan memperhitungkan jenis tanaman, jenis tanah, sensor iklim mikro, serta **menahan penyiraman apabila sebentar lagi akan turun hujan**."
)

st.sidebar.header("Panel Kontrol Sensor & Petak")

preset = st.sidebar.selectbox(
    "Pilih Skenario Demo Cepat:",
    [
        "Skenario 1: Tanah Kering & Cuaca Cerah (Perlu Disiram)",
        "Skenario 2: Tanah Kering Tapi Akan Hujan Lebat (Tunda Siram)",
        "Skenario 3: Selada di Tanah Pasir Mulai Kering (Baseline Gagal, Model Berhasil)",
        "Skenario 4: Kelembapan Tanah Ideal (Pompa Mati)",
        "Kustom (Atur Bebas Seluruh Sensor)",
    ],
)

if preset == "Skenario 1: Tanah Kering & Cuaca Cerah (Perlu Disiram)":
    def_petak, def_tanaman, def_tanah, def_umur = "P01", "cabai", "lempung", 45
    def_kt, def_kt1, def_kt2 = 30.5, 31.2, 32.0
    def_suhu, def_rh, def_lux = 24.5, 70.0, 58000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.0, 13
elif preset == "Skenario 2: Tanah Kering Tapi Akan Hujan Lebat (Tunda Siram)":
    def_petak, def_tanaman, def_tanah, def_umur = "P03", "tomat", "lempung", 60
    def_kt, def_kt1, def_kt2 = 33.0, 33.8, 34.5
    def_suhu, def_rh, def_lux = 21.0, 86.0, 22000.0
    def_hujan, def_prakiraan, def_jam = 0.2, 8.5, 14
elif preset == "Skenario 3: Selada di Tanah Pasir Mulai Kering (Baseline Gagal, Model Berhasil)":
    def_petak, def_tanaman, def_tanah, def_umur = "P06", "selada", "pasir", 35
    def_kt, def_kt1, def_kt2 = 38.5, 39.2, 40.1
    def_suhu, def_rh, def_lux = 23.8, 74.0, 52000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.0, 11
elif preset == "Skenario 4: Kelembapan Tanah Ideal (Pompa Mati)":
    def_petak, def_tanaman, def_tanah, def_umur = "P05", "selada", "lempung", 50
    def_kt, def_kt1, def_kt2 = 54.0, 54.5, 55.0
    def_suhu, def_rh, def_lux = 19.5, 84.0, 35000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.5, 10
else:
    def_petak, def_tanaman, def_tanah, def_umur = "P01", "cabai", "lempung", 45
    def_kt, def_kt1, def_kt2 = 32.0, 32.8, 33.5
    def_suhu, def_rh, def_lux = 23.0, 78.0, 45000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.0, 12

daftar_petak = [f"P{i:02d}" for i in range(1, 17)]
daftar_tanaman = ["cabai", "tomat", "selada"]
daftar_tanah = ["pasir", "lempung", "liat"]

petak = st.sidebar.selectbox("Identitas Petak:", daftar_petak, index=daftar_petak.index(def_petak))
tanaman = st.sidebar.selectbox("Jenis Tanaman:", daftar_tanaman, index=daftar_tanaman.index(def_tanaman))
jenis_tanah = st.sidebar.selectbox("Jenis Tanah:", daftar_tanah, index=daftar_tanah.index(def_tanah))
umur_tanaman_hari = st.sidebar.slider("Umur Tanaman (Hari):", min_value=10, max_value=130, value=def_umur)
jam = st.sidebar.slider("Jam Pengukuran (0-23 WIB):", min_value=0, max_value=23, value=def_jam)

mode_model = st.sidebar.radio(
    "Pilih Model Inferensi:",
    ["Random Forest (F1 = 0.8287)", "Ensemble RF + HistGB (F1 = 0.8313)"],
    index=0,
)
model_type_key = "rf" if mode_model.startswith("Random Forest") else "ensemble"

st.sidebar.markdown("---")
st.sidebar.caption(f"Artifact: `artifacts/{artifact_name}`")
st.sidebar.caption(f"Backend: {'FastAPI Aktif' if api_active else 'Lokal (Direct Artifact)'}")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Live Prediction & Simulasi Sensor",
    "2. Monitor Serentak 16 Petak (P01-P16)",
    "3. Evaluasi Model Baseline",
    "4. Biaya Bisnis",
])

with tab1:
    st.subheader("Input Pembacaan Sensor IoT & Prakiraan Cuaca")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Sensor Kelembapan Tanah**")
        kelembaban_tanah_pct = st.slider(
            "Kelembapan Tanah Saat Ini (%)", min_value=15.0, max_value=80.0, value=float(def_kt), step=0.1
        )
        kt_1jam_lalu = st.slider(
            "Kelembapan Tanah 1 Jam Lalu (%)", min_value=15.0, max_value=80.0, value=float(def_kt1), step=0.1
        )
        kt_2jam_lalu = st.slider(
            "Kelembapan Tanah 2 Jam Lalu (%)", min_value=15.0, max_value=80.0, value=float(def_kt2), step=0.1
        )

    with col2:
        st.markdown("**2. Sensor Iklim Mikro Greenhouse**")
        suhu_udara_C = st.slider(
            "Suhu Udara (°C)", min_value=12.0, max_value=35.0, value=float(def_suhu), step=0.1
        )
        kelembaban_udara_pct = st.slider(
            "Kelembapan Udara (%)", min_value=40.0, max_value=100.0, value=float(def_rh), step=1.0
        )
        cahaya_lux = st.number_input(
            "Intensitas Cahaya (Lux)", min_value=0.0, max_value=100000.0, value=float(def_lux), step=500.0
        )

    with col3:
        st.markdown("**3. Sensor Curah Hujan & Prakiraan**")
        curah_hujan_mm = st.number_input(
            "Curah Hujan Jam Ini (mm)", min_value=0.0, max_value=30.0, value=float(def_hujan), step=0.1
        )
        prakiraan_hujan_6jam_mm = st.number_input(
            "Prakiraan Hujan 6 Jam ke Depan (mm)", min_value=0.0, max_value=50.0, value=float(def_prakiraan), step=0.1
        )
        ambang_tanaman = AMBANG_AGRONOMI[(tanaman, jenis_tanah)]
        st.info(
            f"**Referensi Agronomi:**  \n"
            f"Tanaman **{tanaman.capitalize()}** pada tanah **{jenis_tanah}** memiliki ambang kritis "
            f"kelembapan tanah sekitar **`{ambang_tanaman:.0f}%`**."
        )

    payload_utama = {
        "petak": petak,
        "tanaman": tanaman,
        "jenis_tanah": jenis_tanah,
        "umur_tanaman_hari": umur_tanaman_hari,
        "kelembaban_tanah_pct": kelembaban_tanah_pct,
        "kt_1jam_lalu": kt_1jam_lalu,
        "kt_2jam_lalu": kt_2jam_lalu,
        "suhu_udara_C": suhu_udara_C,
        "kelembaban_udara_pct": kelembaban_udara_pct,
        "cahaya_lux": cahaya_lux,
        "curah_hujan_mm": curah_hujan_mm,
        "prakiraan_hujan_6jam_mm": prakiraan_hujan_6jam_mm,
        "jam": jam,
        "model_type": model_type_key,
    }

    hasil = run_inference(payload_utama, bundel, use_api=api_active)

    if hasil.get("status") != "success":
        st.error("Validasi masukan gagal: " + "; ".join(hasil.get("errors", ["Masukan tidak valid."])))
    else:
        prob_ml = float(hasil["probabilitas_ml"])
        threshold_dipakai = float(hasil["threshold"])
        keputusan_ml = int(hasil["keputusan_ml"])
        keputusan_baseline = int(hasil["keputusan_baseline"])
        summary = hasil["ringkasan_agronomi"]
        kt_diff1 = float(summary["kt_diff1"])
        kt_min3 = float(summary["kt_min_3h"])
        total_air_6jam = float(summary["total_air_6jam"])

        st.divider()
        st.subheader(f"Hasil Keputusan Irigasi - Petak `{petak}` ({tanaman.capitalize()} / Tanah {jenis_tanah.capitalize()})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kelembapan Tanah", f"{kelembaban_tanah_pct:.1f}%", f"{kt_diff1:+.1f}% vs 1 jam lalu")
        m2.metric("Selisih dari Ambang Tanaman", f"{kelembaban_tanah_pct - ambang_tanaman:+.1f}%", f"Ambang: {ambang_tanaman:.0f}%")
        m3.metric("Prakiraan Hujan 6 Jam", f"{prakiraan_hujan_6jam_mm:.1f} mm", f"Saat ini: {curah_hujan_mm:.1f} mm")
        m4.metric("Probabilitas Model", f"{prob_ml * 100:.1f}%", f"Threshold: {threshold_dipakai * 100:.0f}%")

        col_res1, col_res2 = st.columns([3, 2])

        with col_res1:
            if keputusan_ml == 1:
                st.error("### KEPUTUSAN: PERLU DISIRAM SEKARANG (POMPA MENYALA)")
                st.markdown(
                    f"**Alasan Keputusan Agronomi:**  \n"
                    f"1. **Defisit Kelembapan Tanah:** Kelembapan tanah pada petak **{petak}** saat ini (**{kelembaban_tanah_pct:.1f}%**, "
                    f"minimum 3 jam terakhir: **{kt_min3:.1f}%**) berada di bawah ambang tanaman **{tanaman}** "
                    f"pada tanah **{jenis_tanah}** (~**{ambang_tanaman:.0f}%**).  \n"
                    f"2. **Prakiraan Cuaca Aman:** Curah hujan saat ini (**{curah_hujan_mm:.1f} mm**) dan prakiraan hujan "
                    f"6 jam ke depan (**{prakiraan_hujan_6jam_mm:.1f} mm**) rendah.  \n"
                    f"3. **Mitigasi Risiko:** Penyiraman mencegah tanaman **{tanaman}** mengalami stres kekeringan."
                )
            else:
                st.success("### KEPUTUSAN: TIDAK PERLU DISIRAM (POMPA MATI / DITAHAN)")
                if prakiraan_hujan_6jam_mm >= 1.5 or curah_hujan_mm >= 0.5:
                    st.markdown(
                        f"**Alasan Keputusan Agronomi (Penundaan karena Hujan):**  \n"
                        f"1. **Antisipasi Hujan:** Kelembapan tanah berada di angka **{kelembaban_tanah_pct:.1f}%**, "
                        f"namun prakiraan hujan 6 jam ke depan sebesar **{prakiraan_hujan_6jam_mm:.1f} mm** "
                        f"(total suplai air hujan: **{total_air_6jam:.1f} mm**).  \n"
                        f"2. **Efisiensi & Kesehatan Akar:** Menahan penyiraman mencegah pemborosan air dan risiko busuk akar."
                    )
                else:
                    st.markdown(
                        f"**Alasan Keputusan Agronomi (Kelembapan Mencukupi):**  \n"
                        f"1. **Cadangan Air Cukup:** Kelembapan tanah saat ini (**{kelembaban_tanah_pct:.1f}%**) masih memenuhi "
                        f"kebutuhan tanaman **{tanaman}** pada tanah **{jenis_tanah}** (ambang kritis ~**{ambang_tanaman:.0f}%**).  \n"
                        f"2. **Efisiensi Operasional:** Menahan pompa tetap mati menghemat air dan menjaga aerasi akar."
                    )
            st.caption(f"Sumber inferensi: `{hasil.get('source')}` | Versi model: `{hasil.get('model_version')}`")

        with col_res2:
            st.markdown("#### Perbandingan terhadap Baseline (`< 35%`)")
            label_base = "SIRAM (1)" if keputusan_baseline == 1 else "JANGAN SIRAM (0)"
            label_ai = "SIRAM (1)" if keputusan_ml == 1 else "JANGAN SIRAM (0)"

            st.table(pd.DataFrame([
                {"Metode": "Baseline (Kelembapan < 35%)", "Keputusan": label_base},
                {"Metode": "Model Machine Learning", "Keputusan": label_ai},
            ]))

            if keputusan_ml != keputusan_baseline:
                if keputusan_baseline == 1 and keputusan_ml == 0:
                    st.warning(
                        "Aturan `< 35%` menyalakan pompa hanya karena kelembapan di bawah 35%, "
                        "padahal akan turun hujan atau tanah masih aman bagi tanaman."
                    )
                else:
                    st.warning(
                        f"Aturan `< 35%` gagal mendeteksi bahwa tanaman **{tanaman}** memiliki ambang kritis lebih tinggi "
                        f"(~**{ambang_tanaman:.0f}%**)."
                    )
            else:
                st.caption("Pada kondisi sensor ini, keputusan model dan aturan baseline sejalan.")

    st.divider()
    with st.expander("Uji Validasi Masukan Tidak Wajar (Input Kosong, Mustahil, atau di Luar Rentang)"):
        u_col1, u_col2, u_col3 = st.columns(3)
        with u_col1:
            uji_kt = st.text_input("Kelembapan Tanah Uji (%)", value="-15")
            uji_suhu = st.text_input("Suhu Udara Uji (°C)", value="500")
        with u_col2:
            uji_hujan = st.text_input("Curah Hujan Uji (mm)", value="-5")
            uji_lux = st.text_input("Cahaya Lux Uji", value="")
        with u_col3:
            uji_tanaman = st.text_input("Jenis Tanaman Uji", value="cabai")
            uji_tanah = st.text_input("Jenis Tanah Uji", value="lempung")

        if st.button("Jalankan Uji Validasi Masukan", type="secondary"):
            payload_uji = {
                "petak": petak,
                "tanaman": uji_tanaman,
                "jenis_tanah": uji_tanah,
                "umur_tanaman_hari": umur_tanaman_hari,
                "kelembaban_tanah_pct": uji_kt,
                "kt_1jam_lalu": kt_1jam_lalu,
                "kt_2jam_lalu": kt_2jam_lalu,
                "suhu_udara_C": uji_suhu,
                "kelembaban_udara_pct": kelembaban_udara_pct,
                "cahaya_lux": uji_lux,
                "curah_hujan_mm": uji_hujan,
                "prakiraan_hujan_6jam_mm": prakiraan_hujan_6jam_mm,
                "jam": jam,
                "model_type": model_type_key,
            }
            res_uji = run_inference(payload_uji, bundel, use_api=api_active)
            if res_uji.get("status") == "error":
                st.warning(
                    f"**Masukan Ditolak Secara Aman (HTTP {res_uji.get('http_status', 422)}):**  \n"
                    + "\n".join([f"- {err}" for err in res_uji.get("errors", [])])
                )
            else:
                st.success(
                    f"Masukan valid. Prediksi: **{res_uji['label_ml']}** "
                    f"(Probabilitas: {res_uji['probabilitas_ml'] * 100:.1f}%)."
                )

with tab2:
    st.subheader("Pantauan Serentak 16 Petak Rumah Kaca (`P01` - `P16`)")
    if df_sample is None or "kt_lag1" not in df_sample.columns:
        st.info("Data sampel 16 petak tidak tersedia.")
    else:
        daftar_waktu = sorted(df_sample["waktu"].unique(), reverse=True)
        waktu_pilih = st.selectbox("Pilih Jam Rekaman Sensor Historis (Data Uji Oktober 2025):", daftar_waktu, index=0)

        sub_df = df_sample[df_sample["waktu"] == waktu_pilih].sort_values("petak").reset_index(drop=True)
        X_sub = sub_df[bundel["features"]].astype(np.float32)

        p_rf_sub = bundel["model_rf"].predict_proba(X_sub)[:, 1]
        p_hgb_sub = bundel["model_hgb"].predict_proba(X_sub)[:, 1]
        if mode_model.startswith("Random Forest"):
            prob_sub = p_rf_sub
            th_sub = float(bundel["threshold_rf"])
        else:
            prob_sub = 0.5 * p_rf_sub + 0.5 * p_hgb_sub
            th_sub = float(bundel["threshold_ensemble"])

        sub_df["Probabilitas (%)"] = (prob_sub * 100).round(1)
        sub_df["Prediksi Model"] = (prob_sub >= th_sub).astype(int)
        sub_df["Baseline (<35%)"] = (sub_df["kelembaban_tanah_pct"] < 35.0).astype(int)

        jml_siram_ml = int(sub_df["Prediksi Model"].sum())
        jml_siram_asli = int(sub_df["perlu_siram"].sum())
        curah_jam_ini = float(sub_df["curah_hujan_mm"].iloc[0])
        prakiraan_6j = float(sub_df["prakiraan_hujan_6jam_mm"].iloc[0])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Waktu Pantauan", waktu_pilih)
        c2.metric("Petak Perlu Disiram (Model)", f"{jml_siram_ml} / 16 Petak")
        c3.metric("Keputusan Asli Agronom", f"{jml_siram_asli} / 16 Petak")
        c4.metric("Prakiraan Hujan 6 Jam", f"{prakiraan_6j:.1f} mm", f"Hujan Jam Ini: {curah_jam_ini:.1f} mm")

        st.markdown("#### Peta Status 16 Petak Rumah Kaca")
        for row_idx in range(4):
            cols_grid = st.columns(4)
            for col_idx in range(4):
                idx = row_idx * 4 + col_idx
                if idx < len(sub_df):
                    r = sub_df.iloc[idx]
                    status_label = "SIRAM (1)" if r["Prediksi Model"] == 1 else "AMAN (0)"

                    bg_color = "#FFEBEB" if r["Prediksi Model"] == 1 else "#EBF9F1"
                    border_color = "#FF4B4B" if r["Prediksi Model"] == 1 else "#21C354"
                    text_color = "#8A0000" if r["Prediksi Model"] == 1 else "#005A20"

                    with cols_grid[col_idx]:
                        st.markdown(
                            f"""
                            <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                                <h4 style="margin-top: 0px; margin-bottom: 10px; color: {text_color}; text-align: center;">{r['petak']}</h4>
                                <div style="font-size: 14px; color: #333333;">
                                    <b>Tanaman:</b> {r['tanaman'].capitalize()} ({r['jenis_tanah']})<br>
                                    <b>Status:</b> <span style="color: {border_color}; font-weight: bold;">{status_label}</span><br>
                                    <b>Probabilitas:</b> {r['Probabilitas (%)']:.1f}%<br>
                                    <b>Kelembapan:</b> {r['kelembaban_tanah_pct']:.1f}% / <i>{r['ambang_estimasi']:.0f}%</i><br>
                                    <hr style="margin: 8px 0;">
                                    <span style="font-size: 12px; color: #666;">Label Asli: {'Siram' if r['perlu_siram']==1 else 'Aman'}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.markdown("#### Tabel Detail Pembacaan Sensor 16 Petak")
        tampil_cols = [
            "petak", "tanaman", "jenis_tanah", "umur_tanaman_hari",
            "kelembaban_tanah_pct", "kt_lag1", "suhu_udara_C",
            "prakiraan_hujan_6jam_mm", "Probabilitas (%)", "Prediksi Model", "Baseline (<35%)", "perlu_siram",
        ]
        st.dataframe(sub_df[tampil_cols], use_container_width=True)

with tab3:
    st.subheader("Evaluasi Performa Model pada Validation & Test Set")
    mt = bundel["metrics_test"]
    mv = bundel.get("metrics_val", {})

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("F1-Score Random Forest (Test)", f"{mt['f1_ml_rf']:.4f}", f"+{mt['f1_ml_rf'] - mt['f1_baseline']:.4f} vs Baseline")
    k2.metric("Precision Random Forest", f"{mt['prec_ml_rf']:.4f}", "Minim Pemborosan Air")
    k3.metric("Recall Random Forest", f"{mt['rec_ml_rf']:.4f}", "Minim Tanaman Stres")
    k4.metric("F1-Score Baseline (< 35%)", f"{mt['f1_baseline']:.4f}", "Aturan Tanpa ML")

    if mv:
        st.markdown("#### Perbandingan F1-Score (Validation vs Test Set)")
        st.table(pd.DataFrame([
            {"Model": "Baseline (kelembaban_tanah_pct < 35%)", "F1 Validation": f"{mv['f1_baseline']:.4f}", "F1 Test": f"{mt['f1_baseline']:.4f}"},
            {"Model": "Random Forest (th = 0.61)", "F1 Validation": f"{mv['f1_ml_rf']:.4f}", "F1 Test": f"{mt['f1_ml_rf']:.4f}"},
            {"Model": "HistGradientBoosting (th = 0.47)", "F1 Validation": f"{mv['f1_ml_hgb']:.4f}", "F1 Test": f"{mt['f1_ml_hgb']:.4f}"},
            {"Model": "Ensemble RF + HistGB (th = 0.54)", "F1 Validation": f"{mv['f1_ml_ensemble']:.4f}", "F1 Test": f"{mt['f1_ml_ensemble']:.4f}"},
        ]))

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.markdown("#### Perbandingan F1-Score per Jenis Tanaman (Test Set)")
        st.table(pd.DataFrame(mt["crop_metrics"]))
        st.caption(
            "Pada tanaman Selada, baseline (< 35%) memperoleh F1 = 0.1081 karena ambang selada sekitar 41-43%. "
            "Model Random Forest mencapai F1 = 0.8933."
        )

    with col_ev2:
        st.markdown("#### Top 10 Feature Importance (Random Forest)")
        df_imp = pd.DataFrame(mt["top_features"]).head(10)
        fig_imp, ax_imp = plt.subplots(figsize=(6, 3.8))
        sns.barplot(data=df_imp, x="Importance", y="Fitur", color="#2A9D8F", ax=ax_imp)
        ax_imp.set_title("Fitur Penentu Keputusan Irigasi")
        plt.tight_layout()
        st.pyplot(fig_imp)

with tab4:
    st.subheader("Kalkulator Pertimbangan Biaya Operasional & Agronomi")
    st.markdown(
        "Terlambat menyiram (*False Negative*) menyebabkan tanaman stres kekeringan, "
        "sedangkan menyiram tanpa perlu (*False Positive*) memboroskan air/listrik dan meningkatkan risiko busuk akar."
    )

    cb1, cb2 = st.columns(2)
    with cb1:
        biaya_fp = st.number_input(
            "Estimasi Biaya False Positive (per Kejadian - Rp):",
            min_value=1000, max_value=500000, value=15000, step=1000,
        )
    with cb2:
        biaya_fn = st.number_input(
            "Estimasi Biaya False Negative (per Kejadian - Rp):",
            min_value=1000, max_value=500000, value=45000, step=1000,
        )

    cm_b = mt["cm_baseline"]
    cm_r = mt["cm_rf"]
    cm_e = mt["cm_ensemble"]

    fp_b, fn_b = cm_b[0][1], cm_b[1][0]
    fp_r, fn_r = cm_r[0][1], cm_r[1][0]
    fp_e, fn_e = cm_e[0][1], cm_e[1][0]

    rug_b = fp_b * biaya_fp + fn_b * biaya_fn
    rug_r = fp_r * biaya_fp + fn_r * biaya_fn
    rug_e = fp_e * biaya_fp + fn_e * biaya_fn

    df_cost = pd.DataFrame([
        {
            "Metode / Model": "1. Baseline (kelembaban_tanah < 35%)",
            "False Positive": fp_b,
            "False Negative": fn_b,
            "Total Estimasi Kerugian (17 Hari Uji)": f"Rp {rug_b:,.0f}",
            "Penghematan vs Baseline": "Rp 0",
        },
        {
            "Metode / Model": "2. Ensemble RF + HistGB",
            "False Positive": fp_e,
            "False Negative": fn_e,
            "Total Estimasi Kerugian (17 Hari Uji)": f"Rp {rug_e:,.0f}",
            "Penghematan vs Baseline": f"Rp {rug_b - rug_e:,.0f}",
        },
        {
            "Metode / Model": "3. Random Forest (Model Final)",
            "False Positive": fp_r,
            "False Negative": fn_r,
            "Total Estimasi Kerugian (17 Hari Uji)": f"Rp {rug_r:,.0f}",
            "Penghematan vs Baseline": f"Rp {rug_b - rug_r:,.0f}",
        },
    ])
    st.table(df_cost)
