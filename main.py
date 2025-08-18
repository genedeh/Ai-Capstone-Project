import tempfile

import cv2
import streamlit as st

from face_blurrer import FaceBlurrer

st.title("😎 Face Blurring App")
st.write("Upload an image or video, or use your webcam to blur faces.")

blurrer = FaceBlurrer()

mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])

if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        out_path, result = blurrer.blur_image_file(tfile.name)

        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Blurred Image")
        with open(out_path, "rb") as f:
            st.download_button("Download Blurred Image", f, "blurred_image.png")

elif mode == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        out_path = blurrer.blur_video_file(tfile.name)

        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("Download Blurred Video", f, "blurred_video.mp4")

elif mode == "Webcam":
    st.write("Streaming webcam with face blur (no download).")
    run = st.checkbox("Start Webcam")

    if run:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


        class VideoTransformer(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="rgb24")
                img = blurrer._blur_frame(img)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
