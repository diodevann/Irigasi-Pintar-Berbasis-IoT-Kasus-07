# Smart Irrigation IoT - Greenhouse Lembang · Tim [Nama Tim]

**Kasus:** 07 - Irigasi Pintar Berbasis IoT · **Aplikasi:** [tautan aplikasi] · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| [Nama Anggota 1] | [NIM 1] | EDA, pembersihan data, dan rekayasa fitur (`src/features.py`) |
| [Nama Anggota 2] | [NIM 2] | Pelatihan model, tuning threshold, dan evaluasi (`src/train.py`, `notebooks/`) |
| [Nama Anggota 3] | [NIM 3] | Pengembangan API (`app/api/main.py`) dan antarmuka Streamlit (`app/ui/streamlit_app.py`) |
| [Nama Anggota 4] | [NIM 4] | Validasi masukan, deployment, laporan, dan video demo |

## Masalah dan pengguna

Pengguna sistem ini adalah pengelola rumah kaca (*greenhouse*) sayuran di Lembang yang mengelola 16 petak (`P01` sampai `P16`) dengan kombinasi tanaman (cabai, tomat, selada) dan jenis tanah (pasir, lempung, liat). Model membantu keputusan penyiraman per jam (`1` = perlu disiram, `0` = tidak perlu disiram) untuk mencegah tanaman stres kekeringan sekaligus menghindari pemborosan air dan busuk akar saat hujan akan turun.

## Data dan pembersihan

Dataset mentah (`data/raw.csv.gz`) berisi 35.452 baris dan 12 kolom rekaman sensor per jam selama 3 bulan (Agustus - Oktober 2025). Pembagian data dilakukan secara berbasis waktu (*time-based split*): *Train* 23.307 baris (1 Agu - 30 Sep), *Validation* 5.353 baris (1 - 14 Okt), dan *Test* 6.495 baris (15 - 31 Okt).

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat pada pasangan (`waktu`, `petak`) | 297 | Dihapus dengan `drop_duplicates(subset=["waktu", "petak"], keep="first")` |
| Inkonsistensi label `tanaman` (`Cabai`, `CABAI`, `cabe`, `Tomat`, `tomat `, `Selada`, `lettuce`) | 1.055 | Distandarisasi menjadi `cabai`, `tomat`, dan `selada` |
| Nilai `cahaya_lux` negatif | 70 | Dikoreksi dengan `-(x + 1.0)` |
| Anomali `kelembaban_tanah_pct` (`<= 0%` atau `>= 100%`) | 103 | Diubah menjadi `NaN`, lalu diimputasi dengan `ffill` dan `bfill` per petak |
| Nilai hilang pada `suhu_udara_C` | 527 | Diimputasi dengan `ffill` dan `bfill` per petak |
| Nilai hilang pada `prakiraan_hujan_6jam_mm` | 351 | Diimputasi dengan `ffill` dan `bfill` per petak |

## Hasil

| Model | F1-Score *validation* | F1-Score *test* |
|---|---|---|
| *Baseline*: `kelembaban_tanah_pct < 35%` | 0.1783 | 0.1575 |
| HistGradientBoostingClassifier (`threshold = 0.47`) | 0.8581 | 0.8179 |
| Ensemble RF + HistGB (`threshold = 0.54`) | 0.8621 | 0.8291 |
| **RandomForestClassifier (`threshold = 0.61`, Model Final)** | **0.8562** | **0.8391** |

*Threshold* yang dipakai: **0.61**, dipilih dari *validation set* untuk menyeimbangkan *precision* tinggi (**0.9366** pada *test set*, menekan pemborosan air dan risiko busuk akar) dengan *recall* (**0.7600**, mencegah tanaman stres kekeringan).

## Cara menjalankan

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload       # API
streamlit run app/ui/streamlit_app.py   # aplikasi
```

## Struktur repositori

```text
notebooks/      notebook dari EDA sampai penyimpanan model
data/           data mentah dan data sampel 16 petak
src/features.py fungsi pembersihan, validasi, dan pembuat fitur (pelatihan & prediksi)
src/train.py    skrip pelatihan ulang untuk menghasilkan artifacts/
artifacts/      model.joblib + metadata.json
app/            API (FastAPI) dan antarmuka (Streamlit)
```

## Keterbatasan

- Data pelatihan mencakup 3 bulan (Agustus - Oktober 2025) sehingga belum mewakili variasi cuaca sepanjang tahun penuh.
- Model tidak mendeteksi pergeseran kalibrasi sensor (*sensor drift*) yang masih berada di dalam rentang normal.
- Belum mencakup variabel agronomi tambahan seperti pH tanah, EC, dan kapasitas pompa aktual.

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [x] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [x] *Baseline* wajib dilaporkan dan dibandingkan
- [x] Data *test* hanya dibuka satu kali, di akhir
- [x] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [ ] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [x] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
