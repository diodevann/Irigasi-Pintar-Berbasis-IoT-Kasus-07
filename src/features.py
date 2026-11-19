from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numpy as np
import pandas as pd

TANAMAN_DICT: Dict[str, int] = {"cabai": 0, "tomat": 1, "selada": 2}
TANAH_DICT: Dict[str, int] = {"pasir": 0, "lempung": 1, "liat": 2}
VALID_PETAK: List[str] = [f"P{i:02d}" for i in range(1, 17)]

AMBANG_AGRONOMI: Dict[Tuple[str, str], float] = {
    ("cabai", "pasir"): 32.0,
    ("cabai", "lempung"): 34.0,
    ("cabai", "liat"): 35.0,
    ("tomat", "pasir"): 35.0,
    ("tomat", "lempung"): 37.0,
    ("tomat", "liat"): 38.0,
    ("selada", "pasir"): 41.0,
    ("selada", "lempung"): 42.0,
    ("selada", "liat"): 43.0,
}

PEMETAAN_TANAMAN: Dict[str, str] = {
    "cabai": "cabai",
    "Cabai": "cabai",
    "CABAI": "cabai",
    "cabe": "cabai",
    "tomat": "tomat",
    "Tomat": "tomat",
    "tomat ": "tomat",
    "selada": "selada",
    "Selada": "selada",
    "lettuce": "selada",
}

FEATURE_COLUMNS: List[str] = [
    "umur_tanaman_hari",
    "kelembaban_tanah_pct",
    "suhu_udara_C",
    "kelembaban_udara_pct",
    "cahaya_lux",
    "curah_hujan_mm",
    "prakiraan_hujan_6jam_mm",
    "tanaman_enc",
    "jenis_tanah_enc",
    "petak_enc",
    "kt_lag1",
    "kt_lag2",
    "kt_lag3",
    "kt_diff1",
    "kt_diff2",
    "kt_min_3h",
    "kt_mean_3h",
    "hujan_lag1",
    "prakiraan_lag1",
    "jam",
    "jam_sin",
    "jam_cos",
    "evap_index",
    "total_air_6jam",
    "ambang_estimasi",
    "defisit_kt_saat_ini",
    "defisit_kt_min3h",
    "lonjakan_siram_tanpa_hujan",
]

NUMERIC_RANGES: Dict[str, Tuple[float, float, str]] = {
    "umur_tanaman_hari": (1.0, 365.0, "Umur tanaman (hari)"),
    "kelembaban_tanah_pct": (1.0, 99.0, "Kelembaban tanah saat ini (%)"),
    "kt_1jam_lalu": (1.0, 99.0, "Kelembaban tanah 1 jam lalu (%)"),
    "kt_2jam_lalu": (1.0, 99.0, "Kelembaban tanah 2 jam lalu (%)"),
    "suhu_udara_C": (5.0, 50.0, "Suhu udara (C)"),
    "kelembaban_udara_pct": (10.0, 100.0, "Kelembaban udara (%)"),
    "cahaya_lux": (0.0, 150000.0, "Intensitas cahaya (Lux)"),
    "curah_hujan_mm": (0.0, 200.0, "Curah hujan jam ini (mm)"),
    "prakiraan_hujan_6jam_mm": (0.0, 300.0, "Prakiraan hujan 6 jam (mm)"),
    "jam": (0.0, 23.0, "Jam pengukuran (0-23 WIB)"),
}


def clean_raw_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = raw_df.copy()
    stats: Dict[str, int] = {}

    df["waktu_dt"] = pd.to_datetime(df["waktu"])

    stats["duplikat_waktu_petak"] = int(df.duplicated(subset=["waktu", "petak"], keep="first").sum())
    df = df.drop_duplicates(subset=["waktu", "petak"], keep="first")
    df = df.sort_values(["petak", "waktu_dt"]).reset_index(drop=True)

    stats["inkonsistensi_label_tanaman"] = int((~df["tanaman"].isin(TANAMAN_DICT.keys())).sum())
    df["tanaman"] = df["tanaman"].map(PEMETAAN_TANAMAN)

    mask_lux_neg = df["cahaya_lux"] < 0
    stats["cahaya_lux_negatif"] = int(mask_lux_neg.sum())
    df.loc[mask_lux_neg, "cahaya_lux"] = -(df.loc[mask_lux_neg, "cahaya_lux"] + 1.0)

    mask_kt_anomali = (df["kelembaban_tanah_pct"] <= 0.0) | (df["kelembaban_tanah_pct"] >= 100.0)
    stats["kelembaban_tanah_anomali"] = int(mask_kt_anomali.sum())
    df.loc[mask_kt_anomali, "kelembaban_tanah_pct"] = np.nan

    stats["suhu_udara_hilang"] = int(df["suhu_udara_C"].isna().sum())
    stats["prakiraan_hujan_hilang"] = int(df["prakiraan_hujan_6jam_mm"].isna().sum())

    for col in ["kelembaban_tanah_pct", "suhu_udara_C", "prakiraan_hujan_6jam_mm"]:
        df[col] = df.groupby("petak")[col].ffill().bfill()

    return df, stats


def build_batch_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()
    if "waktu_dt" not in df.columns:
        df["waktu_dt"] = pd.to_datetime(df["waktu"])
    df = df.sort_values(["petak", "waktu_dt"]).reset_index(drop=True)

    grp = df.groupby("petak")
    df["kt_lag1"] = grp["kelembaban_tanah_pct"].shift(1).bfill()
    df["kt_lag2"] = grp["kelembaban_tanah_pct"].shift(2).bfill()
    df["kt_lag3"] = grp["kelembaban_tanah_pct"].shift(3).bfill()

    df["kt_diff1"] = df["kelembaban_tanah_pct"] - df["kt_lag1"]
    df["kt_diff2"] = df["kt_lag1"] - df["kt_lag2"]
    df["kt_min_3h"] = df[["kelembaban_tanah_pct", "kt_lag1", "kt_lag2"]].min(axis=1)
    df["kt_mean_3h"] = df[["kelembaban_tanah_pct", "kt_lag1", "kt_lag2"]].mean(axis=1)

    df["hujan_lag1"] = grp["curah_hujan_mm"].shift(1).fillna(0.0)
    df["prakiraan_lag1"] = grp["prakiraan_hujan_6jam_mm"].shift(1).bfill()
    df["total_air_6jam"] = df["curah_hujan_mm"] + df["prakiraan_hujan_6jam_mm"]

    df["jam"] = df["waktu_dt"].dt.hour
    df["jam_sin"] = np.sin(2.0 * np.pi * df["jam"] / 24.0)
    df["jam_cos"] = np.cos(2.0 * np.pi * df["jam"] / 24.0)
    df["evap_index"] = df["suhu_udara_C"] * (100.0 - df["kelembaban_udara_pct"]) / 100.0

    df["tanaman_enc"] = df["tanaman"].map(TANAMAN_DICT)
    df["jenis_tanah_enc"] = df["jenis_tanah"].map(TANAH_DICT)
    df["petak_enc"] = df["petak"].str.slice(1).astype(int)

    df["ambang_estimasi"] = [
        AMBANG_AGRONOMI[(t, j)] for t, j in zip(df["tanaman"], df["jenis_tanah"])
    ]
    df["defisit_kt_saat_ini"] = df["ambang_estimasi"] - df["kelembaban_tanah_pct"]
    df["defisit_kt_min3h"] = df["ambang_estimasi"] - df["kt_min_3h"]
    df["lonjakan_siram_tanpa_hujan"] = (
        (df["kt_diff1"] > 10.0) & (df["curah_hujan_mm"] < 0.5)
    ).astype(int)

    return df


def validate_single_input(payload: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    cleaned: Dict[str, Any] = {}

    if not isinstance(payload, dict):
        return False, ["Format masukan tidak valid."], {}

    petak_raw = payload.get("petak")
    if petak_raw is None or str(petak_raw).strip() == "":
        errors.append("Identitas petak tidak boleh kosong (P01-P16).")
    else:
        petak_val = str(petak_raw).strip().upper()
        if petak_val not in VALID_PETAK:
            errors.append(f"Petak '{petak_raw}' tidak valid (gunakan P01-P16).")
        else:
            cleaned["petak"] = petak_val

    tanaman_raw = payload.get("tanaman")
    if tanaman_raw is None or str(tanaman_raw).strip() == "":
        errors.append("Jenis tanaman tidak boleh kosong (cabai, tomat, selada).")
    else:
        t_str = str(tanaman_raw).strip().lower()
        t_mapped = PEMETAAN_TANAMAN.get(t_str, PEMETAAN_TANAMAN.get(str(tanaman_raw).strip()))
        if t_mapped not in TANAMAN_DICT:
            errors.append(f"Tanaman '{tanaman_raw}' tidak valid (cabai, tomat, selada).")
        else:
            cleaned["tanaman"] = t_mapped

    tanah_raw = payload.get("jenis_tanah")
    if tanah_raw is None or str(tanah_raw).strip() == "":
        errors.append("Jenis tanah tidak boleh kosong (pasir, lempung, liat).")
    else:
        j_str = str(tanah_raw).strip().lower()
        if j_str not in TANAH_DICT:
            errors.append(f"Jenis tanah '{tanah_raw}' tidak valid (pasir, lempung, liat).")
        else:
            cleaned["jenis_tanah"] = j_str

    for field, (min_v, max_v, label) in NUMERIC_RANGES.items():
        val = payload.get(field)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            errors.append(f"{label} tidak boleh kosong.")
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            errors.append(f"{label} harus berupa angka (diterima: '{val}').")
            continue

        if math.isnan(num) or math.isinf(num):
            errors.append(f"{label} tidak boleh NaN atau tak hingga.")
            continue

        if num < min_v or num > max_v:
            errors.append(f"{label} di luar rentang wajar ({min_v:g}-{max_v:g}, diterima: {num:g}).")
            continue

        cleaned[field] = int(round(num)) if field in ("umur_tanaman_hari", "jam") else num

    model_type = str(payload.get("model_type", "rf")).strip().lower()
    cleaned["model_type"] = model_type if model_type in ("rf", "ensemble") else "rf"

    return (len(errors) == 0), errors, cleaned


def build_single_feature_row(cleaned: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    petak = cleaned["petak"]
    tanaman = cleaned["tanaman"]
    jenis_tanah = cleaned["jenis_tanah"]
    umur = float(cleaned["umur_tanaman_hari"])
    kt = float(cleaned["kelembaban_tanah_pct"])
    kt1 = float(cleaned["kt_1jam_lalu"])
    kt2 = float(cleaned["kt_2jam_lalu"])
    suhu = float(cleaned["suhu_udara_C"])
    rh = float(cleaned["kelembaban_udara_pct"])
    lux = float(cleaned["cahaya_lux"])
    hujan = float(cleaned["curah_hujan_mm"])
    prakiraan = float(cleaned["prakiraan_hujan_6jam_mm"])
    jam = int(cleaned["jam"])

    ambang = AMBANG_AGRONOMI[(tanaman, jenis_tanah)]
    kt_min3 = min(kt, kt1, kt2)
    kt_mean3 = (kt + kt1 + kt2) / 3.0
    kt_diff1 = kt - kt1
    kt_diff2 = kt1 - kt2
    evap_index = suhu * (100.0 - rh) / 100.0
    total_air_6jam = hujan + prakiraan

    fitur_dict = {
        "umur_tanaman_hari": umur,
        "kelembaban_tanah_pct": kt,
        "suhu_udara_C": suhu,
        "kelembaban_udara_pct": rh,
        "cahaya_lux": lux,
        "curah_hujan_mm": hujan,
        "prakiraan_hujan_6jam_mm": prakiraan,
        "tanaman_enc": TANAMAN_DICT[tanaman],
        "jenis_tanah_enc": TANAH_DICT[jenis_tanah],
        "petak_enc": int(petak.replace("P", "")),
        "kt_lag1": kt1,
        "kt_lag2": kt2,
        "kt_lag3": kt2,
        "kt_diff1": kt_diff1,
        "kt_diff2": kt_diff2,
        "kt_min_3h": kt_min3,
        "kt_mean_3h": kt_mean3,
        "hujan_lag1": hujan,
        "prakiraan_lag1": prakiraan,
        "jam": jam,
        "jam_sin": float(np.sin(2.0 * np.pi * jam / 24.0)),
        "jam_cos": float(np.cos(2.0 * np.pi * jam / 24.0)),
        "evap_index": evap_index,
        "total_air_6jam": total_air_6jam,
        "ambang_estimasi": ambang,
        "defisit_kt_saat_ini": ambang - kt,
        "defisit_kt_min3h": ambang - kt_min3,
        "lonjakan_siram_tanpa_hujan": int((kt_diff1 > 10.0) and (hujan < 0.5)),
    }

    df_row = pd.DataFrame([fitur_dict])[FEATURE_COLUMNS].astype(np.float32)
    summary = {
        "ambang_estimasi": ambang,
        "kt_min_3h": kt_min3,
        "kt_diff1": kt_diff1,
        "total_air_6jam": total_air_6jam,
        "evap_index": evap_index,
    }
    return df_row, summary
