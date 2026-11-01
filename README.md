# Deteksi dan Klasifikasi Merek Sepatu

Aplikasi Streamlit untuk mengenali merek sepatu **Adidas, Converse, dan Nike** dari foto. Dibuat untuk UTS Praktikum Pemrograman Big Data.

**Aplikasi online:** https://uts-thariq.streamlit.app

**Nama:** Thariq Arkan  
**NPM:** 2308108010065

## Fitur

| Mode | Model | Hasil |
|---|---|---|
| Deteksi | YOLO (`model/best.pt`) | Kotak pada tiap sepatu, beserta merek dan tingkat keyakinan. Ambang keyakinan bisa diatur. |
| Klasifikasi | MobileNetV2 (`model/classifier_model.h5`) | Satu merek untuk seluruh gambar, beserta keyakinan tiap kelas. |

Aplikasi menerima berkas JPG dan PNG. Contoh gambar ada di folder `sample_images/`.

## Menjalankan di komputer

Sudah diuji dengan Python 3.13.

```bash
git clone https://github.com/thariqar23-beep/UTS-Streamlit.git
cd UTS-Streamlit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Untuk memastikan kedua model termuat dan bisa memprediksi:

```bash
python test_app.py
```

Perintah ini mencetak `OK` dan `OK klasifikasi` jika berhasil.

## Deploy ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io), lalu masuk dengan akun GitHub.
2. Pilih **Create app**, lalu isi repo ini, branch `main`, dan berkas `app.py`.
3. Di **Advanced settings**, pilih Python 3.13.

`packages.txt` memasang pustaka sistem yang dibutuhkan OpenCV. Aplikasi memuat TensorFlow dan PyTorch sekaligus, jadi pemakaian memorinya cukup besar. Aplikasi sudah berjalan di Streamlit Community Cloud.

## Dataset

Dataset Roboflow berisi 114 gambar dan 3 kelas, dibagi menjadi 79 latih, 23 validasi, dan 12 test. Dataset tidak disertakan di repo ini.

Model klasifikasi dilatih dari potongan tiap objek ditambah gambar utuh yang hanya berisi satu objek, dengan 158 sampel latih dan 46 sampel validasi.

## Hasil evaluasi

**Deteksi (YOLO)**

| Data | mAP50 |
|---|---|
| Validasi | 0,770 |
| Test | 0,638 |

**Klasifikasi (MobileNetV2)**

| Data | Potongan objek | Gambar utuh |
|---|---|---|
| Validasi | 18/27 (66,7%) | 11/19 (57,9%) |
| Test | 11/13 (84,6%) | 6/11 (54,5%) |

Aplikasi memakai gambar utuh, jadi kolom itu yang paling mendekati pemakaian sebenarnya.

## Keterbatasan

- Datanya sedikit, jadi prediksi bisa keliru dan aplikasi menampilkan peringatan saat keyakinan di bawah 60%.
- Data test hanya memuat 1 objek Adidas, sehingga nilai mAP50 kelas Adidas (0,199) belum bisa dijadikan patokan.
- Pada data validasi, recall kelas Nike hanya 0,231.
- YOLO bisa memberi dua kotak dengan merek berbeda pada satu sepatu. Menaikkan ambang keyakinan membantu mengurangi hal ini.

## Melatih ulang model klasifikasi

Buka `Latih_Klasifikasi_h5.ipynb` di Google Colab, jalankan semua sel, lalu unggah `Hasil_roboflow.zip` saat diminta. Letakkan `classifier_model.h5` hasil unduhannya di folder `model/`. Notebook ini disimpan tanpa output, dan angka hasilnya bisa sedikit berbeda dari tabel di atas.

## Struktur proyek

```
├── app.py                      Aplikasi Streamlit
├── test_app.py                 Cek cepat kedua model
├── Latih_Klasifikasi_h5.ipynb  Notebook pelatihan model .h5
├── model/
│   ├── best.pt                 Bobot YOLO
│   └── classifier_model.h5     Model klasifikasi
├── sample_images/              Contoh gambar
├── requirements.txt            Pustaka Python
├── packages.txt                Pustaka sistem
└── .streamlit/config.toml      Tema aplikasi
```
