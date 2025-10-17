# app2.py
# Fast Streamlit oak-wilt detector with uploads + folder scan + GPS mapping
# OW/NOW mapping: index 0 = NOW, index 1 = OW  (CLASS_NAMES is ordered accordingly)

import os
import io
import math

import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

import exifread
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# ===========================
# CONFIG
# ===========================
MODEL_PATH = r"D:\DNR\Streamlit App\swinv2_tiny_oakwilt25.pth"
MODEL_NAME = "swinv2_tiny_window8_256"

# IMPORTANT: Keep the order [NOW, OW] so logits[1] is OW
CLASS_NAMES = [
    "There's No Oak Wilt in this Image",  # 0 = NOW
    "There's Oak Wilt in this Image",     # 1 = OW
]

IMG_SIZE  = 256
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.75  # used as initial UI value; function uses UI-selected threshold
MARKER_ICON_PATH = "red.png"  # Put this file next to app2.py if you want a custom red pin


# ===========================
# PAGE / SIDEBAR
# ===========================
st.set_page_config(layout="wide", page_title="Oak-Wilt Detector & Geo-Mapper")
st.title("Oak-Wilt Detector & Geo-Mapper")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Confidence threshold for mapping OW detections",
        min_value=0.50, max_value=0.99, value=DEFAULT_THRESHOLD, step=0.01,
        help="Only OW predictions with confidence ≥ this value are mapped."
    )
    st.caption("Label mapping: **NOW = 0**, **OW = 1**")
st.caption(f"Only detections ≥ {int(threshold*100)}% confidence are mapped. (OW=1, NOW=0)")


# ===========================
# CACHE HELPERS
# ===========================
def _model_cache_key(path: str):
    """Key cache by absolute path, modified time, and size to avoid stale model loads."""
    try:
        return (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return (os.path.abspath(path), None, None)


@st.cache_resource(show_spinner=True)
def load_model(path: str, key: tuple):
    """Load model weights once per file content change."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
    # Open explicitly in rb to reduce odd permission issues
    with open(path, "rb") as f:
        state = torch.load(f, map_location=DEVICE)
    m.load_state_dict(state)
    return m.to(DEVICE).eval()


# Instantiate model (cached)
model = load_model(MODEL_PATH, _model_cache_key(MODEL_PATH))


# ===========================
# PREPROCESS
# ===========================
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# ===========================
# INFERENCE
# ===========================
@st.cache_data(show_spinner=False)
def infer_from_bytes(img_bytes: bytes, thresh: float):
    """
    Run inference on raw image bytes; return (label_str, conf_ow, gps_tuple_or_None).
    Cache includes the threshold so the label decision is consistent with UI.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    conf_ow = float(probs[1])  # index 1 = OW
    label = CLASS_NAMES[1] if conf_ow >= thresh else CLASS_NAMES[0]

    # Try EXIF GPS from original bytes (uploads may strip; still worth trying)
    gps = None
    try:
        tags = exifread.process_file(io.BytesIO(img_bytes), details=False)
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            def to_deg(vals):
                d = float(vals[0].num) / vals[0].den
                m = float(vals[1].num) / vals[1].den
                s = float(vals[2].num) / vals[2].den
                return d + m/60 + s/3600
            lat = to_deg(tags['GPS GPSLatitude'].values)
            if tags.get('GPS GPSLatitudeRef') and tags['GPS GPSLatitudeRef'].values != 'N':
                lat = -lat
            lon = to_deg(tags['GPS GPSLongitude'].values)
            if tags.get('GPS GPSLongitudeRef') and tags['GPS GPSLongitudeRef'].values != 'E':
                lon = -lon
            gps = (lat, lon)
    except Exception:
        gps = None

    return label, conf_ow, gps


# ===========================
# UI TABS
# ===========================
tab_upload, tab_folder = st.tabs(["Upload images", "Scan a folder"])


# ---------- Upload Tab ----------
with tab_upload:
    files = st.file_uploader(
        "Upload JPG/PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can select multiple images."
    )

    if files:
        rows = []
        for f in files:
            b = f.read()
            label, conf, gps = infer_from_bytes(b, threshold)
            rows.append((f.name, b, label, conf, gps))

        for name, img_bytes, label, conf, gps in rows:
            c0, c1, c2, c3 = st.columns([2, 3, 2, 2], vertical_alignment="center")
            with c0:
                st.image(img_bytes, caption=name, use_container_width=True)
            c1.markdown(f"**Prediction:** {label}")
            c2.markdown(f"**Confidence (OW):** {conf:.4f}")
            c3.markdown(f"**GPS:** {gps[0]:.6f}, {gps[1]:.6f}" if gps else "—")

        # Map only high-confidence OW with GPS
        positives = [gps for _, _, label, conf, gps in rows if (label == CLASS_NAMES[1]) and gps]
        if positives:
            avg_lat = sum(lat for lat, _ in positives) / len(positives)
            avg_lon = sum(lon for _, lon in positives) / len(positives)
            m = folium.Map((avg_lat, avg_lon), zoom_start=12)
            cluster = MarkerCluster().add_to(m)
            icon = folium.CustomIcon(MARKER_ICON_PATH, icon_size=(30, 30)) if os.path.exists(MARKER_ICON_PATH) else None
            for lat, lon in positives:
                if icon:
                    folium.Marker((lat, lon), icon=icon).add_to(cluster)
                else:
                    folium.Marker((lat, lon)).add_to(cluster)
            st.subheader("Mapped High-Confidence Oak-Wilt Detections")
            st_folium(m, width=900, height=520)
        else:
            st.info("No high-confidence Oak-Wilt detections (or no GPS found).")


# ---------- Folder Scan Tab ----------
with tab_folder:
    img_dir = st.text_input(
        "Folder to scan",
        value=r"D:\DNR\Model Comparison\Sample Images",
        help="Enter a local folder path containing JPG/PNG images."
    )

    if img_dir and os.path.isdir(img_dir):
        exts = {".jpg", ".jpeg", ".png"}
        paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]
        st.write(f"Found {len(paths)} images.")

        data = []
        with st.spinner("Running inference…"):
            for p in paths:
                try:
                    with open(p, "rb") as fh:
                        b = fh.read()
                    label, conf, gps = infer_from_bytes(b, threshold)
                    data.append((p, label, conf, gps))
                except Exception as e:
                    data.append((p, f"Error: {e}", 0.0, None))

        # Filter by prediction
        choice = st.selectbox("Filter by prediction", ["All"] + CLASS_NAMES, index=0)
        if choice != "All":
            data = [d for d in data if d[1] == choice]

        # Pagination
        per_page = st.number_input("Images per page", min_value=5, max_value=100, value=12)
        total = len(data)
        pages = max(1, math.ceil(total / per_page))
        page = st.slider("Page", 1, pages, 1)
        start, end = (page - 1) * per_page, page * per_page
        subset = data[start:end]

        st.subheader(f"Showing {len(subset)} of {total} (Page {page}/{pages})")
        for path, label, conf, gps in subset:
            c0, c1, c2, c3, c4 = st.columns([1, 3, 2, 1, 1], vertical_alignment="center")
            with c0:
                try:
                    im = Image.open(path)
                    im.thumbnail((80, 80))
                    st.image(im)
                except Exception:
                    st.write("—")
            c1.write(os.path.basename(path))
            c2.write(label)
            c3.write(f"{conf:.4f}" if isinstance(conf, (int, float)) else str(conf))
            c4.write(f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else "—")

        # Map only high-confidence OW with GPS
        positives = [gps for _, lab, conf, gps in data if (lab == CLASS_NAMES[1]) and gps]
        if positives:
            avg_lat = sum(lat for lat, _ in positives) / len(positives)
            avg_lon = sum(lon for _, lon in positives) / len(positives)
            m = folium.Map((avg_lat, avg_lon), zoom_start=12)
            cluster = MarkerCluster().add_to(m)
            icon = folium.CustomIcon(MARKER_ICON_PATH, icon_size=(30, 30)) if os.path.exists(MARKER_ICON_PATH) else None
            for lat, lon in positives:
                if icon:
                    folium.Marker((lat, lon), icon=icon).add_to(cluster)
                else:
                    folium.Marker((lat, lon)).add_to(cluster)
            st.subheader("Mapped High-Confidence Oak-Wilt Detections")
            st_folium(m, width=900, height=520)
        else:
            st.info("No high-confidence Oak-Wilt detections (or no GPS found).")

    elif img_dir:
        st.error("Directory not found. Please check the path.")
