import streamlit as st
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ======================
# Load Model
# ======================
@st.cache_resource
def load_emotion_model():
    model = load_model("Emotion_Detection.h5")
    return model

model = load_emotion_model()

# Label Ekspresi sesuai model
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Bahagia', 'Netral', 'Sedih', 'Terkejut']

# ======================
# Face Detection
# ======================
# Gunakan file cascade yang diupload
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Deteksi Ekspresi Wajah", page_icon="😊", layout="wide")
st.title("😊 Deteksi Ekspresi Wajah Real-Time")
st.write("Model: **MobileNet** | Ekspresi: Marah, Jijik, Takut, Bahagia, Netral, Sedih, Terkejut")

# ======================
# Fungsi Prediksi
# ======================
def predict_emotion(face_img):
    # Resize sesuai input model
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype("float") / 255.0
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)

    preds = model.predict(face_img, verbose=0)[0]
    label = emotion_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return label, confidence

# ======================
# Video Processor (Kamera)
# ======================
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            label, confidence = predict_emotion(roi_gray)

            # Gambar kotak & label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({confidence:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ======================
# Mode Aplikasi
# ======================
mode = st.sidebar.radio("Pilih Mode:", ["📷 Kamera", "🖼️ Upload Gambar"])

if mode == "📷 Kamera":
    st.subheader("Deteksi Ekspresi via Kamera")
    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

elif mode == "🖼️ Upload Gambar":
    st.subheader("Deteksi Ekspresi via Upload Gambar")
    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("⚠️ Wajah tidak terdeteksi.")
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]

                label, confidence = predict_emotion(roi_gray)

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{label} ({confidence:.1f}%)", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Hasil Deteksi",
                     use_column_width=True)
