import os
import io
import json
from datetime import datetime
import gc

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ExifTags
import pandas as pd

# ===========================
# CONFIG
# ===========================
MODEL_PATH = os.getenv("OAK_WILT_MODEL_PATH", "models/oak_wilt_3.h5")

CLASSIFICATION_CATEGORIES = {
    "THIS PICTURE HAS OAK WILT": {"min_conf": 99.5, "color": "#FF0000"},
    "HIGH CHANCE OF OAK WILTS": {"min_conf": 90, "max_conf": 99.5, "color": "#FF6600"},
    "CHANGES OF COLORS ON TREE LEAVES": {"min_conf": 70, "max_conf": 90, "color": "#FFAA00"},
    "Not an Oak Wilt": {"max_conf": 70, "color": "#00AA00"}
}

IMG_SIZE = 256
RESULTS_DIR = "results"
MAX_UPLOAD = 300


# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(layout="wide", page_title="Oak-Wilt Detector")
st.title("Grand Haven Parks Oak-Wilt Detector")
st.markdown("Advanced 4-category classification system with GPS mapping")


# ===========================
# MODEL LOADING
# ===========================
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model = load_model()


# ===========================
# HELPER FUNCTIONS
# ===========================
def classify_prediction(confidence):
    confidence = confidence * 100

    if confidence > 99.5:
        return "THIS PICTURE HAS OAK WILT"
    elif 90 < confidence <= 99.5:
        return "HIGH CHANCE OF OAK WILTS"
    elif 70 < confidence <= 90:
        return "CHANGES OF COLORS ON TREE LEAVES"
    else:
        return "Not an Oak Wilt"


def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)


def get_gps_data(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        exif_data = img._getexif()

        if not exif_data:
            return None

        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)

            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                    lat = convert_to_degrees(gps_data["GPSLatitude"])
                    lon = convert_to_degrees(gps_data["GPSLongitude"])

                    if gps_data.get("GPSLatitudeRef") != "N":
                        lat = -lat
                    if gps_data.get("GPSLongitudeRef") != "E":
                        lon = -lon

                    return (lat, lon)

    except Exception:
        pass

    return None


def process_image(img_bytes):
    """Efficient image processing with minimal RAM usage"""

    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img, axis=0)

    prediction = model.predict(img_input, verbose=0)[0][0]

    classification = classify_prediction(prediction)
    gps = get_gps_data(img_bytes)

    del img_array, img, img_input
    gc.collect()

    return classification, prediction * 100, gps


def generate_csv(results):
    positive = [r for r in results if r["classification"] != "Not an Oak Wilt"]

    if not positive:
        return None

    df = pd.DataFrame(positive)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR,
        f"oak_wilt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    df.to_csv(path, index=False)
    return path


def generate_geojson(results):
    positive = [
        r for r in results
        if r["classification"] != "Not an Oak Wilt" and r["gps"]
    ]

    if not positive:
        return None

    geojson = {"type": "FeatureCollection", "features": []}

    for r in positive:
        lat, lon = r["gps"]

        geojson["features"].append({
            "type": "Feature",
            "properties": {
                "filename": r["filename"],
                "confidence": f"{r['confidence']:.2f}%",
                "classification": r["classification"]
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR,
        f"oak_wilt_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
    )

    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)

    return path


# ===========================
# SIDEBAR
# ===========================
with st.sidebar:

    st.header("Classification Info")

    for category, config in CLASSIFICATION_CATEGORIES.items():

        if "min_conf" in config and "max_conf" in config:
            conf_range = f"{config['min_conf']}-{config['max_conf']}%"
        elif "min_conf" in config:
            conf_range = f">{config['min_conf']}%"
        else:
            conf_range = f"≤{config['max_conf']}%"

        st.markdown(category)
        st.caption(f"Confidence: {conf_range}")
        st.markdown("---")


# ===========================
# MAIN UI
# ===========================
tab_upload, tab_folder = st.tabs(["Upload Images", "Scan Folder"])


# ===========================
# UPLOAD TAB
# ===========================
with tab_upload:

    files = st.file_uploader(
        "Upload JPG/PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files:

        if len(files) > MAX_UPLOAD:
            st.warning(f"Only first {MAX_UPLOAD} images will be processed.")
            files = files[:MAX_UPLOAD]

        seen = set()
        unique_files = []

        for f in files:
            if f.name not in seen:
                seen.add(f.name)
                unique_files.append(f)

        results = []

        progress = st.progress(0)

        with st.spinner("Analyzing images..."):

            for i, file in enumerate(unique_files):

                img_bytes = file.read()

                classification, confidence, gps = process_image(img_bytes)

                results.append({
                    "filename": file.name,
                    "classification": classification,
                    "confidence": confidence,
                    "gps": gps
                })

                col1, col2, col3, col4 = st.columns([2,1,1,1])

                with col1:
                    st.image(file, caption=file.name, use_container_width=True)

                with col2:
                    color = CLASSIFICATION_CATEGORIES[classification]["color"]
                    st.markdown(
                        f'<span style="color:{color};font-weight:bold">{classification}</span>',
                        unsafe_allow_html=True
                    )

                with col3:
                    st.write(f"{confidence:.2f}%")

                with col4:
                    if gps:
                        st.write(f"{gps[0]:.5f}, {gps[1]:.5f}")
                    else:
                        st.write("No GPS")

                st.markdown("---")

                progress.progress((i + 1) / len(unique_files))

                del img_bytes
                gc.collect()

        st.subheader("Export Results")

        csv_path = generate_csv(results)
        geojson_path = generate_geojson(results)

        if csv_path:
            st.download_button(
                "Download CSV",
                data=open(csv_path, "rb").read(),
                file_name=os.path.basename(csv_path),
                mime="text/csv"
            )

        if geojson_path:
            st.download_button(
                "Download GeoJSON",
                data=open(geojson_path, "rb").read(),
                file_name=os.path.basename(geojson_path),
                mime="application/geo+json"
            )


# ===========================
# FOLDER TAB
# ===========================
with tab_folder:

    img_dir = st.text_input("Folder to scan")

    if img_dir and os.path.isdir(img_dir):

        exts = {".jpg", ".jpeg", ".png"}

        paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]

        st.write(f"Found {len(paths)} images.")

        if st.button("Analyze Folder"):

            results = []

            with st.spinner("Analyzing images..."):

                for path in paths:

                    try:
                        with open(path, "rb") as f:
                            img_bytes = f.read()

                        classification, confidence, gps = process_image(img_bytes)

                        results.append({
                            "filename": os.path.basename(path),
                            "classification": classification,
                            "confidence": confidence,
                            "gps": gps
                        })

                        col1, col2, col3 = st.columns([1,2,2])

                        with col1:
                            st.image(path, use_container_width=True)

                        with col2:
                            color = CLASSIFICATION_CATEGORIES[classification]["color"]
                            st.markdown(f"**{classification}**")
                            st.markdown(
                                f'<span style="color:{color}">{confidence:.2f}%</span>',
                                unsafe_allow_html=True
                            )

                        with col3:
                            if gps:
                                st.write(f"{gps[0]:.6f}, {gps[1]:.6f}")
                            else:
                                st.write("No GPS")

                        st.markdown("---")

                        del img_bytes
                        gc.collect()

                    except Exception as e:
                        st.error(f"Error processing {os.path.basename(path)}: {e}")