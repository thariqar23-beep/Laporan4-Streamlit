#thrq
# Aplikasi Streamlit: deteksi objek (YOLOv8) dan klasifikasi gambar (MobileNetV2)
import base64
import io
import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

JALUR_H5_THRQ = "model/classifier_model.h5"
AMBANG_RENDAH_THRQ = 0.6                            #thrq di bawah ini hasil diberi peringatan
# Angka uji sungguhan: Laporan 4 (YOLO) dan Latih_Klasifikasi_h5.ipynb (klasifikasi)
BUKTI_THRQ = {
    "Deteksi": "mAP50 pada data test 0,638. Dilatih dari 114 gambar, jadi hasilnya bisa keliru.",
    "Klasifikasi": "Benar 6 dari 11 gambar utuh pada data test. Dilatih dari data kecil, jadi hasilnya bisa keliru.",
}

GAYA_THRQ = """<style>
.block-container {max-width: 1120px; padding: 32px 24px 48px;}
.atas-thrq {display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px;}
.atas-thrq h1 {font-size: 24px !important; font-weight: 500; letter-spacing: -0.01em; padding: 0 !important; margin: 0;}
.meta-thrq {margin-left: auto; color: #6b6b6b; font-size: 12px;}
.label-thrq {color: #6b6b6b; font-size: 12px; margin-bottom: 2px;}
.jawab-thrq {font-size: 30px; font-weight: 500; letter-spacing: -0.02em; line-height: 1.15;}
.jawab-thrq.kosong-thrq {font-size: 16px; color: #6b6b6b; font-weight: 400;}
.sub-thrq {font-size: 14px; color: #6b6b6b; margin: 4px 0 16px;}
.baris-thrq {display: flex; align-items: center; gap: 12px; padding: 10px 0; border-top: 1px solid #f0f0f0;}
.baris-thrq img {width: 56px; height: 56px; object-fit: cover; border-radius: 6px; flex: none;}
.isi-thrq {flex: 1; min-width: 0;}
.nama-thrq {display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 6px;}
.nama-thrq b {font-weight: 500; font-variant-numeric: tabular-nums;}
.jalur-thrq {height: 6px; border-radius: 3px; background: #eeeeee;}
.jalur-thrq i {display: block; height: 6px; border-radius: 3px; background: #8a8a8a;}
.jalur-thrq i.utama-thrq {background: #1d5fbf;}
.catatan-thrq {font-size: 12px; color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 8px 10px; margin-top: 12px;}
.bukti-thrq {font-size: 12px; color: #6b6b6b; line-height: 1.5; margin-top: 16px; padding-bottom: 4px;}
.slot-thrq {display: flex; align-items: center; justify-content: center; aspect-ratio: 1; max-height: 420px; width: 100%;
  border: 1px dashed #d4d4d4; border-radius: 6px; color: #6b6b6b; font-size: 14px; text-align: center; padding: 16px; margin-bottom: 16px;}
div[data-testid="stImage"] img {border-radius: 6px;}
[class*="st-key-kartu_thrq"] {background: #ffffff;}
@media (max-width: 640px) {
  .meta-thrq {margin-left: 0; width: 100%;}
  div[data-testid="stImage"] img {width: 100%; max-height: 240px; object-fit: contain;}
  .slot-thrq {aspect-ratio: auto; height: 96px;}
}
</style>"""

TANDA_THRQ = """<svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true"><path fill="#141414"
d="M2 7.5c0-1.2 1.5-1.6 2.2-.6l1.6 2.3c.9 1.2 2.6 1.4 3.7.4L11.6 8l6.6 4.6c2 .5 3.8 1.5 3.8 3.4v1H2z"/>
<rect x="2" y="18" width="20" height="2" rx="1" fill="#141414"/></svg>"""


@st.cache_resource(show_spinner="Memuat model deteksi...")
def muat_yolo_thrq():
    return YOLO("model/best.pt")                    #thrq bobot terbaik dari Laporan 4


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def muat_klasifikasi_thrq():
    import tensorflow as tf                         #thrq berat, dimuat hanya saat dipakai
    return tf.keras.models.load_model(JALUR_H5_THRQ, compile=False)


def potongan_thrq(img_thrq, kotak_thrq):
    """Potongan sepatu sebagai data URI, untuk ditampilkan di baris hasil."""
    pot_thrq = img_thrq.crop(tuple(float(v) for v in kotak_thrq)).convert("RGB")
    pot_thrq.thumbnail((112, 112))
    buf_thrq = io.BytesIO()
    pot_thrq.save(buf_thrq, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf_thrq.getvalue()).decode()


def baris_thrq(nama_thrq, nilai_thrq=None, utama_thrq=False, gambar_thrq=None):
    """Satu baris merk dan keyakinan. nilai_thrq None berarti belum ada hasil."""
    foto_thrq = f'<img src="{gambar_thrq}" alt="Potongan sepatu {nama_thrq}">' if gambar_thrq else ""
    if nilai_thrq is None:
        teks_thrq, lebar_thrq = '<span style="color:#6b6b6b">&ndash;</span>', 0
    else:
        teks_thrq, lebar_thrq = f"{nilai_thrq:.0%}", nilai_thrq * 100
    kelas_thrq_css = "utama-thrq" if utama_thrq else ""
    return (f'<div class="baris-thrq">{foto_thrq}<div class="isi-thrq">'
            f'<div class="nama-thrq"><span>{nama_thrq}</span><b>{teks_thrq}</b></div>'
            f'<div class="jalur-thrq"><i class="{kelas_thrq_css}" style="width:{lebar_thrq:.1f}%"></i></div>'
            f'</div></div>')


def tulis_hasil_thrq(judul_thrq, sub_thrq, baris_list_thrq, catatan_thrq=None, kosong_thrq=False):
    kelas_jawab_thrq = "jawab-thrq kosong-thrq" if kosong_thrq else "jawab-thrq"
    catatan_html_thrq = f'<div class="catatan-thrq">{catatan_thrq}</div>' if catatan_thrq else ""
    st.markdown(
        f'<div class="label-thrq">Hasil</div><div class="{kelas_jawab_thrq}">{judul_thrq}</div>'
        f'<div class="sub-thrq">{sub_thrq}</div>{"".join(baris_list_thrq)}{catatan_html_thrq}'
        f'<div class="bukti-thrq">{BUKTI_THRQ[menu_thrq]}</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Deteksi Merk Sepatu", layout="wide")
st.markdown(GAYA_THRQ, unsafe_allow_html=True)
st.markdown(
    f'<div class="atas-thrq">{TANDA_THRQ}<h1>Deteksi Merk Sepatu</h1>'
    f'<div class="meta-thrq">Thariq Arkan &middot; 2308108010065</div></div>',
    unsafe_allow_html=True,
)

model_thrq = muat_yolo_thrq()
kelas_thrq = list(model_thrq.names.values())        #thrq Adidas, Converse, Nike

kontrol_kiri_thrq, kontrol_kanan_thrq = st.columns(2, gap="medium")
menu_thrq = kontrol_kiri_thrq.segmented_control(
    "Mode", ["Deteksi", "Klasifikasi"], default="Deteksi", required=True,
    help="Deteksi menandai tiap sepatu dengan kotak. Klasifikasi menebak satu merk untuk seluruh gambar.",
)
conf_thrq = 0.25
if menu_thrq == "Deteksi":
    conf_thrq = kontrol_kanan_thrq.slider(
        "Ambang keyakinan", 0.05, 0.95, 0.25, 0.05,
        help="Kotak dengan keyakinan di bawah nilai ini tidak ditampilkan.",
    )

kiri_thrq, kanan_thrq = st.columns(2, gap="medium")

with kiri_thrq.container(border=True, height="stretch", key="kartu_thrq_foto"):
    unggah_thrq = st.file_uploader("Foto sepatu (JPG atau PNG)", type=["jpg", "jpeg", "png"])
    img_thrq = None
    gagal_thrq = False
    if unggah_thrq is not None:
        try:
            img_thrq = Image.open(unggah_thrq).convert("RGB")   #thrq PNG bisa punya kanal alfa
        except (UnidentifiedImageError, OSError):
            gagal_thrq = True
    slot_gambar_thrq = st.empty()
    if img_thrq is None:
        slot_gambar_thrq.markdown('<div class="slot-thrq">Foto yang diunggah tampil di sini</div>', unsafe_allow_html=True)

with kanan_thrq.container(border=True, height="stretch", key="kartu_thrq_hasil"):
    if gagal_thrq:
        tulis_hasil_thrq("Berkas tidak terbaca", "Berkas ini bukan gambar yang utuh. Coba unggah foto lain.",
                         [baris_thrq(k) for k in kelas_thrq], kosong_thrq=True)

    elif img_thrq is None:
        tulis_hasil_thrq("Belum ada foto", "Unggah foto sepatu untuk melihat merknya.",
                         [baris_thrq(k) for k in kelas_thrq], kosong_thrq=True)

    elif menu_thrq == "Deteksi":
        hasil_thrq = model_thrq(img_thrq, conf=conf_thrq, verbose=False)[0]
        # plot() mengembalikan array BGR, tanpa channels="BGR" warnanya terbalik
        slot_gambar_thrq.image(hasil_thrq.plot(), caption="Hasil deteksi", channels="BGR")

        # Urutkan dari keyakinan tertinggi, baris teratas memakai warna utama
        urut_thrq = sorted(zip(hasil_thrq.boxes.cls, hasil_thrq.boxes.conf, hasil_thrq.boxes.xyxy),
                           key=lambda b_thrq: -float(b_thrq[1]))
        if not urut_thrq:
            tulis_hasil_thrq("Tidak ada sepatu terdeteksi", "Coba turunkan ambang keyakinan atau pakai foto lain.",
                             [baris_thrq(k) for k in kelas_thrq], kosong_thrq=True)
        else:
            merk_thrq = list(dict.fromkeys(hasil_thrq.names[int(b[0])] for b in urut_thrq))
            tulis_hasil_thrq(
                " dan ".join(merk_thrq),
                f"{len(urut_thrq)} kotak terdeteksi",
                [baris_thrq(hasil_thrq.names[int(c)], float(p), i == 0, potongan_thrq(img_thrq, k))
                 for i, (c, p, k) in enumerate(urut_thrq)],
                "Keyakinan model rendah, jadi hasil ini sebaiknya tidak langsung dipercaya."
                if float(urut_thrq[0][1]) < AMBANG_RENDAH_THRQ else None,
            )

    elif not os.path.exists(JALUR_H5_THRQ):
        slot_gambar_thrq.image(img_thrq, caption="Foto yang diunggah")
        tulis_hasil_thrq("Model belum ada", f"Berkas {JALUR_H5_THRQ} belum ada. Latih dulu lewat Latih_Klasifikasi_h5.ipynb.",
                         [baris_thrq(k) for k in kelas_thrq], kosong_thrq=True)

    else:
        slot_gambar_thrq.image(img_thrq, caption="Foto yang diunggah")
        x_thrq = np.array(img_thrq.resize((224, 224)), dtype="float32")[None] / 255.0
        prob_thrq = muat_klasifikasi_thrq().predict(x_thrq, verbose=0)[0]
        terbaik_thrq = int(prob_thrq.argmax())
        tulis_hasil_thrq(
            kelas_thrq[terbaik_thrq],
            f"Keyakinan {prob_thrq[terbaik_thrq]:.0%} untuk seluruh gambar",
            [baris_thrq(kelas_thrq[i], float(prob_thrq[i]), int(i) == terbaik_thrq) for i in prob_thrq.argsort()[::-1]],
            "Keyakinan model rendah, jadi hasil ini sebaiknya tidak langsung dipercaya."
            if prob_thrq[terbaik_thrq] < AMBANG_RENDAH_THRQ else None,
        )
