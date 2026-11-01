#thrq
# Cek cepat: model termuat dengan 3 kelas dan prediksi berjalan. Jalankan: python test_app.py
from PIL import Image
from ultralytics import YOLO

m_thrq = YOLO("model/best.pt")
assert m_thrq.names == {0: "Adidas", 1: "Converse", 2: "Nike"}
assert m_thrq(Image.new("RGB", (640, 640), "white"), verbose=False)[0].boxes is not None
print("OK")

import numpy as np
import tensorflow as tf

k_thrq = tf.keras.models.load_model("model/classifier_model.h5", compile=False)
assert k_thrq.predict(np.zeros((1, 224, 224, 3)), verbose=0).shape == (1, 3)
print("OK klasifikasi")
