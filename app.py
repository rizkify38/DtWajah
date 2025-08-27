# app.py
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Deteksi Ekspresi Wajah", page_icon="🎭", layout="wide")

# =========================
# Konfigurasi & Utilities
# =========================

# Label emosi sesuai urutan output model Anda
# ⚠️ Pastikan jumlah label sesuai dengan output dari model best.h5
EMOTION_LABELS = ['Angry','Disgust','Fear', 'Happy','Neutral', 'Sad','Surprise']

@st.cache_resource(show_spinner=False)
def load_emotion_model(path: str = "mobileNet_emotion_recog.h5"):
    return load_model(path)

@st.cache_resource(show_spinner=False)
def load_face_detector(path: str = "haarcascade_frontalface_default.xml"):
    return cv2.CascadeClassifier(path)

def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    """
    Preproses ROI wajah untuk model: grayscale 48x48, normalisasi 0-1
    Output shape: (1, 48, 48, 1)
    """
    face = cv2.resize(gray_face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    return face

# =========================
# UI
# =========================
st.title("🎭 Deteksi Ekspresi Wajah — Real-time")
st.caption("Menggunakan model: `emotion_model.h5` + haarcascade_frontalface_default.xml")

col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.subheader("Pengaturan")
    draw_box = st.toggle("Tampilkan Bounding Box", value=True)
    show_conf = st.toggle("Tampilkan Confidence", value=True)
    min_face = st.slider("Ukuran minimum wajah (px)", 60, 200, 90, 10)
    scaleFactor = st.slider("Haar scaleFactor", 1.05, 1.50, 1.20, 0.01)
    minNeighbors = st.slider("Haar minNeighbors", 3, 10, 5, 1)

with col1:
    st.subheader("Kamera")
    st.info("Klik **Start** untuk mulai streaming video.")

# =========================
# Video Transformer
# =========================
class EmotionTransformer(VideoTransformerBase):
    def __init__(self, draw_box=True, show_conf=True, min_size=90, scaleFactor=1.2, minNeighbors=5):
        self.model = load_emotion_model()
        self.detector = load_face_detector("haarcascade_frontalface_default.xml")
        self.draw_box = draw_box
        self.show_conf = show_conf
        self.min_size = min_size
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.last_probs = None  # untuk panel probabilitas

    def predict_emotion(self, gray_face: np.ndarray):
        x = preprocess_face(gray_face)
        preds = self.model.predict(x, verbose=0)[0]  # shape: (n_class,)
        idx = int(np.argmax(preds))
        label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else f"Class {idx}"
        conf = float(np.max(preds))
        return label, conf, preds

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=(self.min_size, self.min_size)
        )

        best_probs = None
        faces = sorted(list(faces), key=lambda b: b[2] * b[3], reverse=True)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, conf, probs = self.predict_emotion(roi_gray)

            if best_probs is None:
                best_probs = probs

            if self.draw_box:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = label if not self.show_conf else f"{label} ({conf:.2f})"
            cv2.putText(
                img, text, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

        self.last_probs = best_probs
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# Jalankan WebRTC
# =========================
ctx = webrtc_streamer(
    key="emotion-rtc",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=lambda: EmotionTransformer(
        draw_box=draw_box,
        show_conf=show_conf,
        min_size=min_face,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
    ),
    async_processing=True,
)

# =========================
# Panel Probabilitas
# =========================
with col2:
    st.subheader("Probabilitas (wajah utama)")
    if ctx and ctx.video_transformer and ctx.video_transformer.last_probs is not None:
        probs = ctx.video_transformer.last_probs
        for label, p in zip(EMOTION_LABELS, probs):
            st.write(f"- **{label}**: {p:.3f}")
    else:
        st.write("Belum ada data. Mulai kamera untuk melihat probabilitas.")

st.markdown("---")
with st.expander("ℹ️ Catatan"):
    st.markdown(
        """
- Pastikan file **`best.h5`** dan **`haarcascade_frontalface_default.xml`** berada di direktori yang sama dengan `app.py`.  
- Urutan `EMOTION_LABELS` harus sesuai dengan urutan output model Anda.  
- Jika kamera tidak menyala, cek izin kamera di browser (harus HTTPS/localhost).  
        """
    )
