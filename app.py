import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import tempfile
import io
import os
import torch

_orig_load = torch.load
def _safe_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(f, *args, **kwargs)
torch.load = _safe_load

st.set_page_config(page_title="VisionLens", layout="wide")
st.title("VisionLens")
st.markdown("Object Detection and Tracking using YOLOv8 + ByteTrack")

with st.sidebar:
    st.header("Detection Settings")
    model_name = st.selectbox("Choose Model", ["YOLOv8n", "YOLOv8s", "YOLOv8m"])
    confidence = st.slider("Confidence Threshold", 0.1, 0.95, 0.4, 0.05)

_MODEL_MAP = {"YOLOv8n": "yolov8n.pt", "YOLOv8s": "yolov8s.pt", "YOLOv8m": "yolov8m.pt"}

@st.cache_resource
def load_model(name: str) -> YOLO:
    return YOLO(_MODEL_MAP[name])

def get_detect_model(name: str) -> YOLO:
    m = load_model(name)
    cb_key = "on_predict_postprocess_end"
    if cb_key in m.callbacks:
        m.callbacks[cb_key] = [
            fn for fn in m.callbacks[cb_key]
            if getattr(fn, "__module__", "").find("trackers") == -1
        ]
    return m

def get_track_model(name: str) -> YOLO:
    return YOLO(_MODEL_MAP[name])

COLORS = [
    (255, 99, 132), (54, 162, 235), (255, 206, 86),
    (75, 192, 192), (153, 102, 255), (255, 159, 64),
]

def process_detections(image: np.ndarray, results, conf: float):
    output = image.copy()
    rows = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            score = float(box.conf[0])
            if score < conf:
                continue
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else None
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{label} | ID {track_id}" if track_id is not None else label
            cv2.putText(output, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            rows.append({
                "track_id": track_id,
                "Object": label,
                "Confidence": round(score, 2),
                "Width": x2 - x1,
                "Height": y2 - y1,
            })
    return output, rows

def show_img(container, img, caption=None):
    kw = {"width": "stretch"}
    if caption:
        kw["caption"] = caption
    container.image(img, **kw)

def show_df(df: pd.DataFrame):
    st.dataframe(df, width="stretch", hide_index=True)

def show_metrics_and_charts(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects Found", len(df))
    c2.metric("Classes Detected", df["Object"].nunique())
    c3.metric("Average Confidence", f"{df['Confidence'].mean():.0%}")
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Object Frequency")
        counts = Counter(df["Object"])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(list(counts.keys()), list(counts.values()))
        ax.set_xlabel("Count")
        st.pyplot(fig)
        plt.close()
    with right:
        st.subheader("Confidence Scores")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.barh(df["Object"], df["Confidence"])
        ax2.set_xlim(0, 1)
        st.pyplot(fig2)
        plt.close()

    st.subheader("Detection Details")
    show_df(df)
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Tracking", "Webcam Detection"])
with tab1:
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            show_img(col1, image)

        if st.button("Run Detection", key="btn_detect"):
            model = get_detect_model(model_name)
            results = model(image_np, verbose=False)
            detected_image, detections = process_detections(image_np, results, confidence)

            with col2:
                st.subheader("Detection Result")
                show_img(col2, detected_image)

            if detections:
                st.divider()
                show_metrics_and_charts(pd.DataFrame(detections))
                buf = io.BytesIO()
                Image.fromarray(detected_image).save(buf, format="PNG")
                st.download_button("⬇ Download Result", buf.getvalue(),
                                   "detection_result.png", "image/png")
            else:
                st.warning("No objects detected.")

with tab2:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    max_frames = st.slider("Frames To Process", 10, 100, 30)
    export_video = st.checkbox("Export annotated video", value=True)
    if uploaded_video and st.button("Run Tracking", key="btn_track"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded_video.read())
        tmp.close()
        model = get_track_model(model_name)
        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        skip = max(1, total_frames // max_frames)
        output_fps = max(1, fps / skip)

        out_path = None
        writer = None
        if export_video:
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     output_fps, (frame_w, frame_h))

        all_dets = []
        frame_num = 0
        processed = 0
        prev_h, prev_w = None, None
        progress = st.progress(0)
        preview = st.empty()
        while cap.isOpened() and processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_num % skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if prev_h is not None and (h != prev_h or w != prev_w):
                    model = get_track_model(model_name)
                prev_h, prev_w = h, w
                try:
                    results = model.track(rgb, persist=True, verbose=False)
                    out_frame, rows = process_detections(rgb, results, confidence)
                except (cv2.error, Exception):
                    model = get_track_model(model_name)
                    results = model(rgb, verbose=False)
                    out_frame, rows = process_detections(rgb, results, confidence)
                all_dets.extend(rows)
                preview.image(out_frame, caption=f"Frame {frame_num}", width="stretch")
                if writer is not None:
                    writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
                processed += 1
                progress.progress(processed / max_frames)
            frame_num += 1
        cap.release()
        if writer:
            writer.release()
        os.unlink(tmp.name)

        if export_video and out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button("⬇ Download Annotated Video", f.read(),
                                   "tracked_output.mp4", "video/mp4")
            os.unlink(out_path)

        if all_dets:
            df_v = pd.DataFrame(all_dets)
            unique_df = df_v.drop_duplicates(subset=["track_id", "Object"])
            summary = (unique_df
                       .groupby("Object")
                       .agg(Unique_Count=("track_id", "count"),
                            Avg_Confidence=("Confidence", "mean"))
                       .reset_index())
            summary["Avg_Confidence"] = summary["Avg_Confidence"].round(2)
            st.success("Tracking Completed")
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Tracked Objects", len(unique_df))
            c2.metric("Classes Detected", unique_df["Object"].nunique())
            c3.metric("Frame Detections", len(df_v))
            st.divider()

            left, right = st.columns(2)
            with left:
                st.subheader("Tracked Object Frequency")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(summary["Object"], summary["Unique_Count"])
                ax.set_xlabel("Objects")
                st.pyplot(fig)
                plt.close()
            with right:
                st.subheader("Tracking Summary")
                show_df(summary)

            st.subheader("Tracked Object Details")
            show_df(unique_df)
        else:
            st.warning("No objects detected in video.")

with tab3:
    st.subheader("Live Webcam Detection")
    st.info("Capture a photo — detection runs instantly on the snapshot.")

    snapshot = st.camera_input("Point your camera and capture")
    if snapshot:
        image = Image.open(snapshot).convert("RGB")
        image_np = np.array(image)

        model = get_detect_model(model_name)
        results = model(image_np, verbose=False)
        detected_image, detections = process_detections(image_np, results, confidence)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            show_img(col1, image)
        with col2:
            st.subheader("Detected")
            show_img(col2, detected_image)
        if detections:
            st.divider()
            show_metrics_and_charts(pd.DataFrame(detections))
            buf = io.BytesIO()
            Image.fromarray(detected_image).save(buf, format="PNG")
            st.download_button("⬇ Download Result", buf.getvalue(),
                               "webcam_detection.png", "image/png")
        else:
            st.warning("No objects detected.")