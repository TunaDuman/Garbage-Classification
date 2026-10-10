import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.set_page_config(page_title="Atık Türü Sınıflandırma", layout="centered")
st.title("Atık Türü Sınıflandırma Sistemi")

KAYIT_KLASORU = "gecmis_taramalar"
if not os.path.exists(KAYIT_KLASORU):
    os.makedirs(KAYIT_KLASORU)

@st.cache_resource
def load_garbage_model():
    return load_model('ayiricimodel.h5')

model = load_garbage_model()

class_names = [
    'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'
]

camera_image = st.camera_input("Bir atık fotoğrafı çekin")

if camera_image is not None:
    image = Image.open(camera_image)
    image_np = np.array(image)
    
    st.image(image, caption="Orijinal Görüntü", use_container_width=True)
    
    resized_frame = cv2.resize(image_np, (224, 224))
    img_array = img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    predicted_class_name = class_names[predicted_class_index]
    skor_yuzde = confidence * 100

    st.success(f"Tahmin Edilen Atık: **{predicted_class_name}**")
    st.info(f"Güven Skoru: %{skor_yuzde:.2f}")

    zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
    dosya_adi = f"{predicted_class_name}_{skor_yuzde:.1f}_{zaman_damgasi}.jpg"
    kayit_yolu = os.path.join(KAYIT_KLASORU, dosya_adi)
    image.save(kayit_yolu)

st.markdown("---")
st.subheader(" Geçmiş Taramalar")

dosya_yollari = [os.path.join(KAYIT_KLASORU, f) for f in os.listdir(KAYIT_KLASORU) if f.endswith(".jpg")]
dosya_yollari.sort(key=os.path.getctime, reverse=True)

if len(dosya_yollari) > 0:
    for i in range(0, len(dosya_yollari), 3):
        cols = st.columns(3)
        
        for j in range(3):
            if i + j < len(dosya_yollari):
                dosya_yolu = dosya_yollari[i + j]
                dosya_adi = os.path.basename(dosya_yolu)
                parcalar = dosya_adi.replace(".jpg", "").split("_")
                etiket = parcalar[0]
                skor = parcalar[1] if len(parcalar) >= 3 else "???"
                
                with cols[j]:
                    st.image(dosya_yolu, caption=f"{etiket} (%{skor})", use_container_width=True)
                    if st.button("Sil", key=dosya_yolu):
                        os.remove(dosya_yolu)
                        st.rerun()
else:
    st.write("Henüz kaydedilmiş bir atık taraması bulunmuyor.")