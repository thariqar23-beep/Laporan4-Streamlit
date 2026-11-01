#thrq
# Aplikasi Streamlit: deteksi objek (YOLOv8) dan klasifikasi gambar (MobileNetV2)
import base64
import csv
import io
import os
import time

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

JALUR_H5_THRQ = "model/classifier_model.h5"
FOLDER_CONTOH_THRQ = "sample_images"
AMBANG_RENDAH_THRQ = 0.6                            #thrq di bawah ini hasil diberi peringatan
MEREK_LOGO_THRQ = ["Adidas", "Converse", "Nike"]
# Angka uji sungguhan: UTS (YOLO) dan Latih_Klasifikasi_h5.ipynb (klasifikasi)
BUKTI_THRQ = {
    "Deteksi": "mAP50 pada data test 0,638. Dilatih dari 114 gambar, jadi hasilnya bisa keliru.",
    "Klasifikasi": "Benar 6 dari 11 gambar utuh pada data test. Dilatih dari data kecil, jadi hasilnya bisa keliru.",
}

GAYA_THRQ = """<style>
div[data-testid="stElementContainer"]:has(style) {display: none;}
.stApp {background: #14213d;}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1200px; padding: 0 32px 24px;}
.kepala-thrq {display: flex; align-items: center; gap: 16px; background: #ffffff; width: 100vw; margin: 0 calc(50% - 50vw) 20px;
  padding: 18px max(32px, calc(50vw - 568px)); border-bottom: 3px solid #c9a227;}
[data-testid="stMain"] {overflow-x: clip;}
.kepala-thrq .usk-thrq {height: 64px; width: auto; flex: none;}
h1.judul-thrq {font-size: 24px !important; font-weight: 600 !important; color: #14213d; line-height: 1.2; padding: 0 !important; margin: 0;}
.kepala-thrq p {color: #4a5365; font-size: 14px; margin: 2px 0 0;}
.merek-logo-thrq {margin-left: auto; display: flex; align-items: center; gap: 28px;}
.merek-logo-thrq img {height: 32px; width: auto; transition: opacity .2s;}
.logo-Nike-thrq {height: 20px !important;}
[class*="st-key-panel"] {background: #ffffff; border-radius: 12px; padding: 24px;}
[data-testid="stButtonGroup"] button {background: transparent !important; border: 0 !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; color: #4a5365 !important; box-shadow: none !important;}
[data-testid="stButtonGroup"] button * {color: inherit !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"] {border-bottom-color: #14213d !important; color: #14213d !important; font-weight: 600;}
h2.bagian-thrq {font-size: 16px !important; font-weight: 600 !important; padding: 0 !important; margin: 0 0 4px;}
.slot-thrq {display: flex; align-items: center; justify-content: center; height: 420px; width: 100%;
  background: #eef0f3; border-radius: 10px; color: #4a5365; font-size: 14px; text-align: center; padding: 16px;}
.st-key-kanvas_thrq div[data-testid="stImage"] img {width: 100%; height: 420px; object-fit: contain; background: #eef0f3; border-radius: 10px;}
[data-testid="stFileUploader"] label {font-size: 13px; color: #4a5365;}
[data-testid="stFileUploaderDropzone"] {padding: 10px 14px; background: #eef0f3;}
.st-key-contoh_thrq div[data-testid="stImage"] img {width: 100%; height: 96px; object-fit: contain; background: #eef0f3; border-radius: 6px;}
.st-key-contoh_thrq [data-testid="stElementToolbar"] {display: none;}
.st-key-contoh_thrq div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap;}
.st-key-contoh_thrq div[data-testid="stColumn"] {min-width: 0; flex: 1 1 0;}
.st-key-contoh_thrq button {background: transparent; border: 1px solid #d9dde4; box-shadow: none; color: #0f172a;}
.st-key-contoh_thrq button:hover {background: #eef0f3; border-color: #d9dde4; color: #0f172a;}
.st-key-contoh_thrq button[kind="primary"] {background: #14213d; border-color: #14213d; color: #ffffff;}
.st-key-contoh_thrq button[kind="primary"] * {color: #ffffff; font-weight: 600;}
.jawab-thrq {font-size: 28px; font-weight: 600; line-height: 1.2; color: #14213d; margin: 2px 0 2px;}
.sub-thrq {font-size: 14px; color: #4a5365; margin-bottom: 14px;}
table.tabel-thrq {width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums;}
table.tabel-thrq, table.tabel-thrq tr, table.tabel-thrq th, table.tabel-thrq td {border: 0 !important; background: transparent;}
table.tabel-thrq th {text-align: left; font-size: 12px; font-weight: 500; color: #4a5365; padding: 0 10px 6px 0; border-bottom: 1px solid #d9dde4 !important;}
table.tabel-thrq td {padding: 9px 10px 9px 0; border-bottom: 1px solid #eef0f3 !important; vertical-align: middle;}
table.tabel-thrq .kanan-thrq {text-align: right; padding-right: 0;}
table.tabel-thrq tr.utama-thrq td {font-weight: 600;}
table.tabel-thrq img {display: block; width: 40px; height: 40px; object-fit: cover; border-radius: 4px;}
.ukuran-thrq {font-size: 12px; color: #4a5365; font-weight: 400; line-height: 1.4; white-space: nowrap;}
.jalur-thrq {height: 4px; border-radius: 2px; background: #eef0f3; margin-top: 5px;}
.jalur-thrq i {display: block; height: 4px; border-radius: 2px; background: #8a93a3;}
tr.utama-thrq .jalur-thrq i {background: #14213d;}
.catatan-thrq {font-size: 13px; color: #92400e; background: #fffbeb; border-radius: 6px; padding: 8px 10px; margin-top: 12px;}
.bukti-thrq {font-size: 13px; color: #4a5365; line-height: 1.5; margin: 12px 0 4px;}
.status-thrq {display: flex; flex-wrap: wrap; gap: 4px 24px; font-size: 12px; color: #c5cbd8; margin-top: 4px; font-variant-numeric: tabular-nums;}
.status-thrq b {font-weight: 500; color: #ffffff;}
.status-thrq .nama-thrq {margin-left: auto;}
@media (max-width: 640px) {
  .block-container {padding: 0 16px 24px;}
  .kepala-thrq {flex-wrap: wrap; gap: 12px; padding: 40px 16px 16px;}
  .kepala-thrq .usk-thrq {height: 52px;}
  h1.judul-thrq {font-size: 20px !important;}
  .merek-logo-thrq {margin-left: 0; width: 100%;}
  [class*="st-key-panel"] {padding: 16px;}
  .slot-thrq {height: 220px;}
  .st-key-kanvas_thrq div[data-testid="stImage"] img {height: 300px;}
  .status-thrq .nama-thrq {margin-left: 0;}
}
</style>"""


@st.cache_resource(show_spinner="Memuat model deteksi...")
def muat_yolo_thrq():
    return YOLO("model/best.pt")                    #thrq bobot terbaik dari UTS


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def muat_klasifikasi_thrq():
    import tensorflow as tf                         #thrq berat, dimuat hanya saat dipakai
    return tf.keras.models.load_model(JALUR_H5_THRQ, compile=False)


def potongan_thrq(img_thrq, kotak_thrq):
    """Potongan sepatu sebagai data URI, untuk kolom pertama tabel deteksi."""
    pot_thrq = img_thrq.crop(tuple(float(v) for v in kotak_thrq)).convert("RGB")
    pot_thrq.thumbnail((96, 96))
    buf_thrq = io.BytesIO()
    pot_thrq.save(buf_thrq, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf_thrq.getvalue()).decode()


def logo_thrq(nama_thrq):
    with open(f"assets/logo/{nama_thrq}.svg", "rb") as berkas_thrq:
        return "data:image/svg+xml;base64," + base64.b64encode(berkas_thrq.read()).decode()


def gaya_logo_thrq(merk_thrq):
    """Logo merek selain jawaban teratas diredupkan."""
    redup_thrq = "".join(f".logo-{m}-thrq {{opacity: .3;}}" for m in MEREK_LOGO_THRQ if merk_thrq and m != merk_thrq)
    return f"<style>{redup_thrq}</style>"


def nilai_thrq(nilai):
    """Keyakinan dengan bar kecil. None berarti belum ada hasil."""
    if nilai is None:
        return "&ndash;"
    return f'{nilai:.0%}<div class="jalur-thrq"><i style="width:{nilai * 100:.1f}%"></i></div>'


def tabel_deteksi_thrq(baris_list_thrq):
    """baris: (merek, keyakinan, data URI potongan, kotak xyxy). Kosong: tiga merek bertanda strip."""
    kepala_thrq = "<tr><th></th><th>Merek</th><th>Keyakinan</th><th class='kanan-thrq'>Kotak (px)</th></tr>"
    isi_thrq = ""
    for i_thrq, (merk_thrq, p_thrq, foto_thrq, kotak_thrq) in enumerate(baris_list_thrq):
        gambar_thrq = f'<img src="{foto_thrq}" alt="Potongan sepatu {merk_thrq}">' if foto_thrq else ""
        ukuran_thrq = "&ndash;"
        if kotak_thrq is not None:
            x1, y1, x2, y2 = (int(v) for v in kotak_thrq)
            ukuran_thrq = f'<div class="ukuran-thrq">x {x1}, y {y1}<br>{x2 - x1} &times; {y2 - y1}</div>'
        kelas_thrq = "utama-thrq" if i_thrq == 0 and p_thrq is not None else ""
        isi_thrq += (f'<tr class="{kelas_thrq}"><td>{gambar_thrq}</td><td>{merk_thrq}</td>'
                     f'<td>{nilai_thrq(p_thrq)}</td><td class="kanan-thrq">{ukuran_thrq}</td></tr>')
    return f'<table class="tabel-thrq">{kepala_thrq}{isi_thrq}</table>'


def tabel_klasifikasi_thrq(baris_list_thrq):
    """baris: (merek, keyakinan), terurut dari tertinggi."""
    isi_thrq = "".join(
        f'<tr class="{"utama-thrq" if i == 0 and p is not None else ""}"><td>{m}</td><td style="width:60%">{nilai_thrq(p)}</td></tr>'
        for i, (m, p) in enumerate(baris_list_thrq))
    return f'<table class="tabel-thrq"><tr><th>Merek</th><th>Keyakinan</th></tr>{isi_thrq}</table>'


def tulis_hasil_thrq(judul_thrq, sub_thrq, tabel_thrq, merk_thrq=None, catatan_thrq=None):
    slot_logo_thrq.markdown(gaya_logo_thrq(merk_thrq), unsafe_allow_html=True)
    st.session_state.merk_thrq = merk_thrq
    catatan_html_thrq = f'<div class="catatan-thrq">{catatan_thrq}</div>' if catatan_thrq else ""
    st.markdown(
        f'<h2 class="bagian-thrq">Hasil {menu_thrq.lower()}</h2><div class="jawab-thrq">{judul_thrq}</div>'
        f'<div class="sub-thrq">{sub_thrq}</div>{tabel_thrq}{catatan_html_thrq}'
        f'<div class="bukti-thrq">{BUKTI_THRQ[menu_thrq]}</div>',
        unsafe_allow_html=True,
    )


def tombol_unduh_thrq(gambar_thrq, kolom_thrq, baris_thrq):
    """Unduh foto hasil (PNG) dan tabel hasil (CSV). on_click="ignore" supaya halaman tidak dimuat ulang."""
    png_thrq = io.BytesIO()
    gambar_thrq.save(png_thrq, "PNG")
    teks_thrq = io.StringIO()
    tulis_thrq = csv.writer(teks_thrq)
    tulis_thrq.writerow(kolom_thrq)
    tulis_thrq.writerows(baris_thrq)
    nama_thrq = f"hasil_{menu_thrq.lower()}"
    kiri_unduh_thrq, kanan_unduh_thrq = st.columns(2, gap="small")
    kiri_unduh_thrq.download_button("Unduh foto", png_thrq.getvalue(), f"{nama_thrq}.png", "image/png",
                                    key="unduh_foto_thrq", on_click="ignore", width="stretch")
    kanan_unduh_thrq.download_button("Unduh tabel (CSV)", teks_thrq.getvalue(), f"{nama_thrq}.csv", "text/csv",
                                     key="unduh_tabel_thrq", on_click="ignore", width="stretch")


def tabel_kosong_thrq():
    if menu_thrq == "Deteksi":
        return tabel_deteksi_thrq([(k, None, None, None) for k in kelas_thrq])
    return tabel_klasifikasi_thrq([(k, None) for k in kelas_thrq])


def catatan_hasil_thrq(merk_thrq, nilai):
    """Catatan jujur: foto contoh punya label, jadi jawaban yang salah bisa disebut."""
    catatan_thrq = []
    if contoh_thrq and merk_thrq != contoh_thrq:
        catatan_thrq.append(f"Foto contoh ini berlabel {contoh_thrq}, jadi jawaban model keliru.")
    if nilai < AMBANG_RENDAH_THRQ:
        catatan_thrq.append("Keyakinan model rendah, jadi hasil ini sebaiknya tidak langsung dipercaya.")
    return " ".join(catatan_thrq) or None


def pilih_contoh_thrq(nama_thrq):
    """Klik contoh yang sama melepasnya. Uploader dikosongkan lewat kunci baru."""
    st.session_state.contoh_thrq = None if st.session_state.contoh_thrq == nama_thrq else nama_thrq
    st.session_state.kunci_unggah_thrq += 1


st.set_page_config(page_title="Deteksi Merk Sepatu", layout="wide")
st.session_state.setdefault("contoh_thrq", None)
st.session_state.setdefault("kunci_unggah_thrq", 0)
st.session_state.setdefault("merk_thrq", None)
slot_logo_thrq = st.empty()                         #thrq redup logo terakhir dulu, diganti setelah hasil diketahui
slot_logo_thrq.markdown(gaya_logo_thrq(st.session_state.merk_thrq), unsafe_allow_html=True)
st.markdown(GAYA_THRQ, unsafe_allow_html=True)

logo_merek_thrq = "".join(f'<img class="logo-{m}-thrq" src="{logo_thrq(m.lower())}" alt="Logo {m}">' for m in MEREK_LOGO_THRQ)
st.markdown(
    f'<div class="kepala-thrq"><img class="usk-thrq" src="{logo_thrq("usk")}" alt="Logo Universitas Syiah Kuala"><div>'
    '<h1 class="judul-thrq">Deteksi Merk Sepatu</h1><p>UTS Praktikum Pemrograman Big Data</p></div>'
    f'<div class="merek-logo-thrq">{logo_merek_thrq}</div></div>',
    unsafe_allow_html=True,
)

model_thrq = muat_yolo_thrq()
kelas_thrq = list(model_thrq.names.values())        #thrq Adidas, Converse, Nike

kiri_thrq, kanan_thrq = st.columns([7, 5], gap="small")
panel_kanan_thrq = kanan_thrq.container(key="panel_kanan_thrq", height="stretch")
menu_thrq = panel_kanan_thrq.segmented_control("Mode", ["Deteksi", "Klasifikasi"], default="Deteksi",
                                               required=True, label_visibility="collapsed")
kotak_hasil_thrq = panel_kanan_thrq.container()
kotak_ambang_thrq = panel_kanan_thrq.container()

with kiri_thrq.container(key="panel_kiri_thrq", height="stretch"):
    slot_gambar_thrq = st.container(key="kanvas_thrq").empty()
    kunci_thrq = st.session_state.kunci_unggah_thrq
    kamera_aktif_thrq = st.toggle("Ambil foto dari kamera", key="kamera_aktif_thrq")
    if kamera_aktif_thrq:                           #thrq kamera baru diminta setelah sakelar dinyalakan
        unggah_thrq = st.camera_input("Foto sepatu dari kamera", key=f"kamera_thrq_{kunci_thrq}")
    else:
        unggah_thrq = st.file_uploader("Foto sepatu (JPG atau PNG)", type=["jpg", "jpeg", "png"],
                                       key=f"unggah_thrq_{kunci_thrq}")
    if unggah_thrq is not None:
        st.session_state.contoh_thrq = None
    contoh_thrq = st.session_state.contoh_thrq
    with st.container(key="contoh_thrq"):
        st.markdown('<h2 class="bagian-thrq">Atau coba foto contoh</h2>', unsafe_allow_html=True)
        for kolom_thrq, nama_thrq in zip(st.columns(3, gap="small"), MEREK_LOGO_THRQ):
            kolom_thrq.image(f"{FOLDER_CONTOH_THRQ}/{nama_thrq}.jpg", width="stretch")
            kolom_thrq.button(nama_thrq, key=f"tombol_{nama_thrq}_thrq", width="stretch",
                              type="primary" if contoh_thrq == nama_thrq else "secondary",
                              on_click=pilih_contoh_thrq, args=(nama_thrq,))

conf_thrq = 0.25
if menu_thrq == "Deteksi":
    conf_thrq = kotak_ambang_thrq.slider("Ambang keyakinan", 0.05, 0.95, 0.25, 0.05,
                                         help="Kotak dengan keyakinan di bawah nilai ini tidak ditampilkan.")
ambang_teks_thrq = f"{conf_thrq:.2f}".replace(".", ",")

img_thrq = None
gagal_thrq = False
sumber_thrq = unggah_thrq or (f"{FOLDER_CONTOH_THRQ}/{contoh_thrq}.jpg" if contoh_thrq else None)
if sumber_thrq is not None:
    try:
        img_thrq = Image.open(sumber_thrq).convert("RGB")   #thrq PNG bisa punya kanal alfa
    except (UnidentifiedImageError, OSError):
        gagal_thrq = True
keterangan_thrq = f"foto contoh {contoh_thrq}" if contoh_thrq else ("foto dari kamera" if kamera_aktif_thrq else "foto yang diunggah")

if img_thrq is None:
    slot_gambar_thrq.markdown('<div class="slot-thrq">Unggah foto atau pilih foto contoh</div>', unsafe_allow_html=True)

model_nama_thrq = "best.pt" if menu_thrq == "Deteksi" else "classifier_model.h5"
status_thrq = [("Model", model_nama_thrq), ("Ukuran foto", "&ndash;"), ("Waktu prediksi", "&ndash;")]
if img_thrq is not None:
    status_thrq[1] = ("Ukuran foto", f"{img_thrq.width} &times; {img_thrq.height} px")

with kotak_hasil_thrq:
    if gagal_thrq:
        tulis_hasil_thrq("Berkas tidak terbaca", "Berkas ini bukan gambar yang utuh. Coba unggah foto lain.", tabel_kosong_thrq())

    elif img_thrq is None:
        tulis_hasil_thrq("Belum ada foto", "Hasil muncul setelah foto dipilih.", tabel_kosong_thrq())

    elif menu_thrq == "Deteksi":
        hasil_thrq = model_thrq(img_thrq, conf=conf_thrq, verbose=False)[0]
        status_thrq[2] = ("Waktu prediksi", f"{hasil_thrq.speed['inference']:.0f} ms")
        # plot() mengembalikan array BGR, tanpa channels="BGR" warnanya terbalik
        slot_gambar_thrq.image(hasil_thrq.plot(), channels="BGR", width="stretch",
                               caption=f"Hasil deteksi pada {keterangan_thrq}")

        # Urutkan dari keyakinan tertinggi
        urut_thrq = sorted(zip(hasil_thrq.boxes.cls, hasil_thrq.boxes.conf, hasil_thrq.boxes.xyxy),
                           key=lambda b_thrq: -float(b_thrq[1]))
        if not urut_thrq:
            tulis_hasil_thrq("Tidak ada sepatu terdeteksi", f"0 kotak pada ambang {ambang_teks_thrq}. Coba turunkan ambang atau pakai foto lain.",
                             tabel_kosong_thrq())
        else:
            merk_thrq = list(dict.fromkeys(hasil_thrq.names[int(b[0])] for b in urut_thrq))
            nilai_atas_thrq = float(urut_thrq[0][1])
            tulis_hasil_thrq(
                " dan ".join(merk_thrq),
                f"{len(urut_thrq)} kotak pada ambang {ambang_teks_thrq}",
                tabel_deteksi_thrq([(hasil_thrq.names[int(c)], float(p), potongan_thrq(img_thrq, k), k) for c, p, k in urut_thrq]),
                merk_thrq[0],
                catatan_hasil_thrq(merk_thrq[0], nilai_atas_thrq),
            )
            tombol_unduh_thrq(
                Image.fromarray(hasil_thrq.plot()[..., ::-1]),      #thrq plot() BGR, PIL butuh RGB
                ["merek", "keyakinan", "x", "y", "lebar", "tinggi"],
                [[hasil_thrq.names[int(c)], round(float(p), 4), int(k[0]), int(k[1]), int(k[2]) - int(k[0]), int(k[3]) - int(k[1])]
                 for c, p, k in urut_thrq],
            )

    elif not os.path.exists(JALUR_H5_THRQ):
        slot_gambar_thrq.image(img_thrq, width="stretch", caption=keterangan_thrq[0].upper() + keterangan_thrq[1:])
        tulis_hasil_thrq("Model belum ada", f"Berkas {JALUR_H5_THRQ} belum ada. Latih dulu lewat Latih_Klasifikasi_h5.ipynb.", tabel_kosong_thrq())

    else:
        slot_gambar_thrq.image(img_thrq, width="stretch", caption=keterangan_thrq[0].upper() + keterangan_thrq[1:])
        x_thrq = np.array(img_thrq.resize((224, 224)), dtype="float32")[None] / 255.0
        model_kl_thrq = muat_klasifikasi_thrq()
        mulai_thrq = time.perf_counter()
        prob_thrq = model_kl_thrq.predict(x_thrq, verbose=0)[0]
        status_thrq[2] = ("Waktu prediksi", f"{(time.perf_counter() - mulai_thrq) * 1000:.0f} ms")
        urutan_thrq = prob_thrq.argsort()[::-1]
        terbaik_thrq = int(urutan_thrq[0])
        tulis_hasil_thrq(
            kelas_thrq[terbaik_thrq],
            "Satu merek untuk seluruh gambar, masukan 224 &times; 224 px",
            tabel_klasifikasi_thrq([(kelas_thrq[i], float(prob_thrq[i])) for i in urutan_thrq]),
            kelas_thrq[terbaik_thrq],
            catatan_hasil_thrq(kelas_thrq[terbaik_thrq], float(prob_thrq[terbaik_thrq])),
        )
        tombol_unduh_thrq(img_thrq, ["merek", "keyakinan"],
                          [[kelas_thrq[i], round(float(prob_thrq[i]), 4)] for i in urutan_thrq])

isi_status_thrq = "".join(f"<span>{k} <b>{v}</b></span>" for k, v in status_thrq)
st.markdown(f'<div class="status-thrq">{isi_status_thrq}<span class="nama-thrq">Thariq Arkan, 2308108010065</span></div>',
            unsafe_allow_html=True)
