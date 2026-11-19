# Irigasi Pintar Berbasis IoT · Tim [DK (Dio Kindi]

**Kasus:** 07 - Irigasi Pintar Berbasis IoT · **Aplikasi:** https://irigasi-pintar-berbasis-iot-kasus-07-dio-kindi.streamlit.app/ · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| [I Gede Dio Devan Satria Prawira ] | [2401871] | Menyiapkan lingkungan kerja dan `requirements.txt`. Membuat API FastAPI (`app/api/main.py`) beserta validasi masukan yang tidak wajar. Men-*deploy* backend ke PythonAnywhere dan frontend ke Streamlit Community Cloud, lalu menghubungkan keduanya dan mengujinya dari ujung ke ujung. Menulis README ini serta bagian Aplikasi dan Keterbatasan di laporan. |
| [Reifana Muhamad Al Kindi] | [2400867] | Memeriksa dan membersihkan data mentah, EDA, membuat fitur (`src/features.py`). Melatih, men-*tuning*, dan mengevaluasi model di *notebook*, lalu menyimpan model dan metadata (`src/train.py`, `artifacts/`). Membuat antarmuka Streamlit (`app/ui/streamlit_app.py`). Menulis bagian Masalah, Data, Metode, dan Hasil di laporan. |

Video demo dibuat berdua, dan masing-masing menjelaskan bagian yang dikerjakannya sendiri.

## Masalah dan pengguna

Penggunanya pengelola rumah kaca sayuran di Lembang yang punya 16 petak (P01 sampai P16) dengan tanaman cabai, tomat, atau selada di atas tanah pasir, lempung, atau liat. Agronomnya tidak bisa berjaga 24 jam, jadi model ini setiap jam memberi tahu petak mana yang perlu disiram (`1`) dan mana yang tidak (`0`). Menyiram terlambat membuat tanaman stres, sedangkan menyiram tanpa perlu (apalagi menjelang hujan) membuang air dan bisa membusukkan akar.

## Data dan pembersihan

Data mentah (`data/raw.csv.gz`) berisi 35.452 baris dan 12 kolom, satu baris untuk satu petak per jam, Agustus sampai Oktober 2025. Setelah dibersihkan tersisa 35.155 baris. Hanya sekitar 4% baris yang berlabel perlu disiram, jadi kelasnya sangat timpang.

Data dibagi berdasarkan waktu, bukan diacak:

| Bagian | Periode | Jumlah baris |
|---|---|---|
| *Train* | 1 Agustus - 30 September | 23.307 |
| *Validation* | 1 - 14 Oktober | 5.353 |
| *Test* | 15 - 31 Oktober | 6.495 |

Masalah yang kami temukan (jumlah dihitung pada data mentah, sebelum duplikat dihapus):

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat pada pasangan `waktu` dan `petak` | 297 | Dihapus dengan `drop_duplicates`, baris pertama dipertahankan |
| Penulisan `tanaman` tidak seragam (`Cabai`, `CABAI`, `cabe`, `Tomat`, `tomat ` dengan spasi, `Selada`, `lettuce`) | 1.063 | Dipetakan ke tiga nilai: `cabai`, `tomat`, `selada` |
| `cahaya_lux` bernilai negatif | 70 | Dikembalikan dengan rumus `-(x + 1)`, karena polanya mirip kesalahan pembacaan sensor |
| `kelembaban_tanah_pct` bernilai 0%, 100%, atau 127,5% (sensor putus atau meluap) | 106 | Diubah menjadi `NaN`, lalu diisi nilai jam sebelumnya di petak yang sama (`ffill`) |
| `suhu_udara_C` kosong | 528 | Diisi dengan `ffill` per petak |
| `prakiraan_hujan_6jam_mm` kosong | 354 | Diisi dengan `ffill` per petak |

Pengisian hanya memakai data jam sebelumnya supaya tidak ada informasi dari masa depan yang ikut masuk. `bfill` hanya dipakai kalau baris pertama sebuah petak kosong.

Dua temuan EDA yang paling berpengaruh:

1. Ambang kering tiap tanaman berbeda. Selada perlu disiram saat kelembapan tanah turun ke sekitar 40-43%, tomat 34-38%, cabai 30-35%. Karena itu aturan tunggal "kurang dari 35%" tidak cocok untuk semuanya. Tanah liat juga menahan air lebih lama daripada lempung dan pasir.
2. Sekitar seperempat label `perlu_siram = 1` tercatat pada jam ketika penyiraman baru saja berlangsung, sehingga sensor sudah melonjak (kira-kira dari 31% ke 52%) padahal tidak ada hujan. Kalau model hanya melihat jam itu saja, tanahnya tampak sudah basah. Maka kami menambahkan fitur dari jam-jam sebelumnya.

Dari situ dibuat 28 fitur, di antaranya kelembapan tanah 1-3 jam sebelumnya, selisih dan nilai terendah 3 jam terakhir, selisih terhadap ambang tanaman dan tanah, penanda lonjakan tanpa hujan, jam (sin/cos), indeks penguapan, serta total air 6 jam (hujan sekarang + prakiraan). Semuanya hanya memakai data sampai jam saat ini.

## Hasil

Metrik evaluasi utama yang ditetapkan adalah F1-Score pada kelas positif (`perlu_siram = 1`). Model dievaluasi pada *Validation Set* untuk optimasi *threshold*, lalu diuji satu kali pada *Test Set*.

| Model | F1 *Validation* | F1 *Test* | *Precision* | *Recall* | *Accuracy* |
|---|---|---|---|---|---|
| *Baseline* (`kelembaban_tanah_pct` < 35%) | 0,1783 | 0,1575 | 0,1159 | 0,2457 | 0,9292 |
| HistGradientBoosting (*threshold* 0,47) | 0,8581 | 0,8179 | - | - | - |
| *Ensemble* RF + HistGB (*threshold* 0,54) | 0,8621 | 0,8291 | 0,9291 | 0,7486 | 0,9917 |
| **RandomForestClassifier (*threshold* 0,61, Final)** | 0,8562 | 0,8391 | 0,9366 | 0,7600 | 0,9921 |

Model final Random Forest (*threshold* = 0,61) berhasil meningkatkan F1-Score pada data *Test* dari 0,1575 (*baseline*) menjadi 0,8391. Pada data *Test* (6.495 sampel), jumlah penyiraman sia-sia (*False Positive*) turun drastis dari 328 menjadi hanya 9 kejadian, dan tanaman yang terlambat disiram (*False Negative*) turun dari 132 menjadi 42 kejadian.

Dengan asumsi biaya agronomi Rp15.000 untuk setiap FP (pemborosan air dan risiko busuk akar) serta Rp45.000 untuk setiap FN (tanaman stres), estimasi kerugian selama 17 hari uji berkurang dari Rp10.860.000 menjadi Rp2.025.000. Ini memberikan penghematan biaya sebesar Rp8.835.000 atau 81,4%.

Peningkatan performa F1-Score juga terlihat merata di semua jenis tanaman dibandingkan aturan *baseline*:
- **Selada**: 0,1081 menjadi 0,8933
- **Cabai**: 0,1353 menjadi 0,7957
- **Tomat**: 0,2947 menjadi 0,7838

Fitur yang paling berpengaruh pada keputusan model Random Forest adalah selisih kelembapan tanah terhadap ambang batas dalam 3 jam terakhir (`defisit_kt_min3h`, 21,2%), disusul kelembapan tanah 1 jam sebelumnya (`kt_lag1`, 10,0%), nilai terendah 3 jam terakhir (`kt_min_3h`, 8,2%), selisih saat ini (`defisit_kt_saat_ini`, 7,2%), `kt_lag2` (6,0%), dan pola waktu harian `jam_cos` (5,9%). Dominasi fitur ini masuk akal secara agronomi karena keputusan penyiraman ditentukan oleh seberapa jauh kelembapan tanah sudah turun di bawah batas aman tanamannya selama beberapa jam terakhir serta pola waktu harian.

## Cara menjalankan

Jalankan dari folder utama repositori.

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload       # API
streamlit run app/ui/streamlit_app.py   # aplikasi
python -m src.train                     # (opsional) melatih ulang dan membuat ulang artifacts/
```

**Arsitektur dan *deployment*.** Backend dan tampilan sengaja dipisah:

```
Streamlit (Streamlit Community Cloud)  --HTTP-->  FastAPI (PythonAnywhere)  -->  artifacts/model.joblib
```

API untuk sensor di lapangan ada di PythonAnywhere ([https://diodevann.pythonanywhere.com/docs#/]), sedangkan dashboard ada di Streamlit Community Cloud. Jadi kalau dashboard sedang *maintenance* atau ramai pengunjung, API tetap berjalan. Alamat API dibaca dari variabel lingkungan `IRRIGATION_API_URL`. Jika API tidak bisa dihubungi, dashboard memakai model di `artifacts/` secara langsung.

**Masukan dan keluaran.** Masukannya tanaman, jenis tanah, petak, umur tanaman, jam, kelembapan tanah sekarang dan 1-2 jam sebelumnya, suhu, kelembapan udara, cahaya, curah hujan, dan prakiraan hujan 6 jam. Keluarannya keputusan siram atau tidak, probabilitas, *threshold* yang dipakai, pembanding dari *baseline*, dan ringkasan agronomi. Masukan yang kosong, bukan angka, atau di luar rentang wajar ditolak dengan HTTP 422 dan daftar pesan galat, sehingga aplikasi tidak berhenti mendadak.

## Struktur repositori

```
notebooks/      notebook dari EDA sampai penyimpanan model
data/           data mentah (raw.csv.gz) dan sampel 16 petak
src/features.py pembersihan, validasi masukan, dan pembuat fitur (dipakai saat pelatihan dan prediksi)
src/train.py    melatih ulang model dan menulis artifacts/
artifacts/      model.joblib + metadata.json
app/            API (FastAPI) dan antarmuka (Streamlit)
```

## Keterbatasan

- Data hanya tiga bulan (Agustus - Oktober 2025), jadi belum mencakup musim lain atau cuaca ekstrem. Kejadian perlu siram di data *test* hanya 175, sehingga F1 bisa bergeser cukup jauh kalau beberapa kejadian saja berubah.
- Model masih melewatkan sekitar seperempat kejadian perlu siram (*recall* 0,76). Kalau bagi klien tanaman layu lebih mahal daripada air terbuang, *threshold* sebaiknya diturunkan.
- Prakiraan hujan hanya berpengaruh kecil di model (kepentingan fitur sekitar 0,02), jadi aturan "jangan siram kalau mau hujan" belum dipelajari model sekuat yang diminta klien.
- Aplikasi hanya meminta kelembapan tanah 1 dan 2 jam sebelumnya. Fitur jam ke-3 dan data hujan jam sebelumnya diisi dengan pendekatan sederhana, sehingga hasilnya bisa sedikit berbeda dari kondisi saat pelatihan.
- Model tidak bisa mendeteksi sensor yang pelan-pelan melenceng (*drift*) selama nilainya masih tampak wajar.
- Belum ada data pH tanah, EC, atau kapasitas pompa. Jika dipakai sungguhan, yang perlu dipantau adalah perubahan pola data, penurunan F1 dibanding keputusan agronom, dan kalibrasi sensor.

## Daftar periksa sebelum dikumpulkan

- [x] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [x] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [x] *Baseline* wajib dilaporkan dan dibandingkan
- [x] Data *test* hanya dibuka satu kali, di akhir
- [x] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [x] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [x] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
