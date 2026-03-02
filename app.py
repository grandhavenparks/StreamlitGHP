import os
import io
import json
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ExifTags
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# ===========================
# CONFIG
# ===========================
# Model path - use environment variable for deployment, fallback to local path
MODEL_PATH = os.getenv("OAK_WILT_MODEL_PATH", "models/oak_wilt_3.h5")

# Classification categories matching model.py
CLASSIFICATION_CATEGORIES = {
    "THIS PICTURE HAS OAK WILT": {"min_conf": 99.5, "color": "#FF0000"},
    "HIGH CHANCE OF OAK WILTS": {"min_conf": 90, "max_conf": 99.5, "color": "#FF6600"},
    "CHANGES OF COLORS ON TREE LEAVES": {"min_conf": 70, "max_conf": 90, "color": "#FFAA00"},
    "Not an Oak Wilt": {"max_conf": 70, "color": "#00AA00"}
}

IMG_SIZE = 256
MARKER_ICON_PATH = os.path.join(os.path.dirname(__file__), "public/red.png")
RESULTS_DIR = "results"


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
    """Load TensorFlow model once."""
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.error("Please set the OAK_WILT_MODEL_PATH environment variable or ensure the model file exists.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model = load_model()


# ===========================
# HELPER FUNCTIONS
# ===========================
def predict_img(img):
    """Use the centralized preprocessing function."""
    img_processed = preprocess_image(img)
    prediction = model.predict(img_processed)
    return prediction[0][0]

def preprocess_image(img_array):
    """Preprocess image for TensorFlow model."""
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded


def classify_prediction(confidence):
    """Classify prediction into 4 categories."""
    confidence = confidence * 100  # Convert to percentage
    
    if confidence > 99.5:
        return "THIS PICTURE HAS OAK WILT"
    elif 90 < confidence <= 99.5:
        return "HIGH CHANCE OF OAK WILTS"
    elif 70 < confidence <= 90:
        return "CHANGES OF COLORS ON TREE LEAVES"
    else:
        return "Not an Oak Wilt"


def get_gps_data(img_bytes):
    """Extract GPS data from image EXIF."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        exif_data = img._getexif()
        
        if not exif_data:
            return None
            
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == 'GPSInfo':
                gps_data = {}
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                    lat = convert_to_degrees(gps_data['GPSLatitude'])
                    lon = convert_to_degrees(gps_data['GPSLongitude'])
                    
                    if gps_data.get('GPSLatitudeRef') != "N":
                        lat = -lat
                    if gps_data.get('GPSLongitudeRef') != "E":
                        lon = -lon
                        
                    return (lat, lon)
    except Exception:
        pass
    
    return None


def convert_to_degrees(value):
    """Convert GPS coordinates to decimal degrees."""
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)


def process_image(img_bytes):
    """Process uploaded image bytes through the model."""
    
    # Option A: Save temporarily (like model.py)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    
    try:
        img = cv2.imread(tmp_path)
        prediction = predict_img(img)  # Returns 0-1 range
        
        # Use the classification function for consistency
        classification = classify_prediction(prediction)
        
        # Extract GPS
        gps = get_gps_data(img_bytes)
        
        return classification, prediction * 100, gps  # Convert to percentage for display
    finally:
        os.unlink(tmp_path)  # Clean up


def generate_csv(results):
    """Generate CSV file from results."""
    # Filter out "Not an Oak Wilt" for CSV
    positive_results = [r for r in results if r['classification'] != "Not an Oak Wilt"]
    
    if not positive_results:
        return None
    
    df = pd.DataFrame(positive_results)
    csv_path = os.path.join(RESULTS_DIR, f"oak_wilt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    return csv_path


def generate_geojson(results):
    """Generate GeoJSON file from results with GPS data."""
    # Filter positive results with GPS
    positive_results = [
        r for r in results 
        if r['classification'] != "Not an Oak Wilt" and r['gps']
    ]
    
    if not positive_results:
        return None
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for result in positive_results:
        lat, lon = result['gps']
        feature = {
            "type": "Feature",
            "properties": {
                "filename": result['filename'],
                "confidence": f"{result['confidence']:.2f}%",
                "classification": result['classification']
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        }
        geojson_data["features"].append(feature)
    
    geojson_path = os.path.join(RESULTS_DIR, f"oak_wilt_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    return geojson_path


# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.header("Classification Info:")
    
    for category, config in CLASSIFICATION_CATEGORIES.items():
        color = config['color']
        if 'min_conf' in config and 'max_conf' in config:
            conf_range = f"{config['min_conf']}-{config['max_conf']}%"
        elif 'min_conf' in config:
            conf_range = f">{config['min_conf']}%"
        else:
            conf_range = f"≤{config['max_conf']}%"
        
        st.markdown(f"{category}")
        st.caption(f"Confidence: {conf_range}")
        st.markdown("---")


# ===========================
# MAIN UI
# ===========================
tab_upload, tab_folder = st.tabs(["Upload Images", "Scan Folder"])

# Upload Tab 
with tab_upload:
    files = st.file_uploader(
        "Upload JPG/PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis"
    )

    if files:
        # Filter out duplicate filenames, keeping only the first occurrence
        seen_filenames = set()
        unique_files = []
        for file in files:
            if file.name not in seen_filenames:
                seen_filenames.add(file.name)
                unique_files.append(file)
        
        results = []
        
        with st.spinner("Analyzing images..."):
            for file in unique_files:
                img_bytes = file.read()
                classification, confidence, gps = process_image(img_bytes)
                
                results.append({
                    'filename': file.name,
                    'img_bytes': img_bytes,
                    'classification': classification,
                    'confidence': confidence,
                    'gps': gps
                })
        
        # Display results
        st.subheader(f"Results ({len(results)} images)")
        
        # Filter by classification
        all_categories = ["All"] + list(CLASSIFICATION_CATEGORIES.keys())
        selected_filter = st.selectbox("Filter by classification", all_categories)
        
        if selected_filter != "All":
            filtered_results = [r for r in results if r['classification'] == selected_filter]
        else:
            filtered_results = results
        
        # Display images with results
        for result in filtered_results:
            col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
            
            with col1:
                st.image(result['img_bytes'], caption=result['filename'], use_container_width=True)
            
            with col2:
                color = CLASSIFICATION_CATEGORIES[result['classification']]['color']
                st.markdown(f"**Classification:**")
                st.markdown(f'<span style="color:{color};font-weight:bold">{result["classification"]}</span>', 
                           unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**Confidence of OW:**")
                st.markdown(f"**{result['confidence']:.2f}%**")
            
            with col4:
                st.markdown(f"**GPS:**")
                if result['gps']:
                    st.markdown(f"{result['gps'][0]:.6f}, {result['gps'][1]:.6f}")
                else:
                    st.markdown("Not found")
            
            with col5:
                st.markdown("**Feedback:**")
                feedback_key_up = f"thumbs_up_{result['filename']}"
                feedback_key_down = f"thumbs_down_{result['filename']}"
                
                col_up, col_down = st.columns(2)
                with col_up:
                    if st.button("Good", key=feedback_key_up, help="Good prediction"):
                        st.toast("Feedback received! Analysis was correct.")
                with col_down:
                    if st.button("Bad", key=feedback_key_down, help="Bad prediction"):
                        st.toast("Feedback received! Analysis was incorrect.")
            
            st.markdown("---")
        
        csv_path = generate_csv(results)
        geojson_path = generate_geojson(results)

        csv_exists = csv_path and os.path.exists(csv_path)
        geojson_exists = geojson_path and os.path.exists(geojson_path)

        st.subheader("Export Results:")
        btn_csv, btn_geojson, _ = st.columns([1, 1.4, 4.6], gap=None)

        st.markdown("""
        <style>
        [data-testid="stDownloadButton"] button {
            height: 2.5rem;
            white-space: nowrap;
        }
        </style>
        """, unsafe_allow_html=True)
        with btn_csv:
            st.download_button(
                label="Download CSV",
                data=open(csv_path, "rb").read() if csv_exists else b"",
                file_name=os.path.basename(csv_path) if csv_exists else "no_data.csv",
                mime="text/csv",
                disabled=not csv_exists,
                key="download_csv_main"
            )

        with btn_geojson:
            st.download_button(
                label="Download GeoJSON",
                data=open(geojson_path, "rb").read() if geojson_exists else b"",
                file_name=os.path.basename(geojson_path) if geojson_exists else "no_data.geojson",
                mime="application/geo+json",
                disabled=not geojson_exists,
                key="download_geojson_main"
            )

        # Map positive detections with GPS
        gps_results = [r for r in results if r['gps'] and r['classification'] != "Not an Oak Wilt"]
        
        if gps_results:
            st.subheader("GPS Map of Positive Detections")
            
            avg_lat = sum(r['gps'][0] for r in gps_results) / len(gps_results)
            avg_lon = sum(r['gps'][1] for r in gps_results) / len(gps_results)
            
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
            cluster = MarkerCluster().add_to(m)
            
            icon = folium.CustomIcon(MARKER_ICON_PATH, icon_size=(30, 30)) if os.path.exists(MARKER_ICON_PATH) else None
            
            for result in gps_results:
                lat, lon = result['gps']
                popup_text = f"{result['filename']}<br>{result['classification']}<br>{result['confidence']:.2f}%"
                
                if icon:
                    folium.Marker([lat, lon], popup=popup_text, icon=icon).add_to(cluster)
                else:
                    folium.Marker([lat, lon], popup=popup_text).add_to(cluster)
            
            st_folium(m, width=900, height=500)
        else:
            st.info("No positive detections with GPS data found for mapping.")


# Folder Scan Tab
with tab_folder:
    img_dir = st.text_input(
        "Folder to scan",
        value="",
        help="Enter a local folder path containing JPG/PNG images"
    )
    
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
                            'filename': os.path.basename(path),
                            'img_bytes': img_bytes,
                            'classification': classification,
                            'confidence': confidence,
                            'gps': gps
                        })
                    except Exception as e:
                        st.error(f"Error processing {os.path.basename(path)}: {e}")
            
            if results:
                st.success(f"Successfully analyzed {len(results)} images!")
                
                # Show summary statistics
                st.subheader("Summary")
                category_counts = {}
                for result in results:
                    cat = result['classification']
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                for category, count in category_counts.items():
                    color = CLASSIFICATION_CATEGORIES[category]['color']
                    st.markdown(f"**{category}**: {count} images")
                
                # Display results (similar to upload tab)
                st.subheader("Detailed Results")
                
                for result in results:
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    with col1:
                        st.image(result['img_bytes'], caption=result['filename'], use_container_width=True)
                    
                    with col2:
                        color = CLASSIFICATION_CATEGORIES[result['classification']]['color']
                        st.markdown(f"**{result['classification']}**")
                        st.markdown(f'<span style="color:{color}">{result["confidence"]:.2f}%</span>', 
                                   unsafe_allow_html=True)
                    
                    with col3:
                        if result['gps']:
                            st.markdown(f"GPS: {result['gps'][0]:.6f}, {result['gps'][1]:.6f}")
                        else:
                            st.markdown("GPS: Not found")
                    
                    st.markdown("---")
    
    elif img_dir:
        st.error("Directory not found. Please check the path.")