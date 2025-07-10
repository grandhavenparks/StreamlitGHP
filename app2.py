import os
import io
import math
import exifread
import folium

import torch
import timm
import streamlit as st

from PIL import Image
from torchvision import transforms
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────
# 1) CONFIG
# ─────────────────────────────────────────────────
MODEL_PATH       = r"E:/Research/Model Comparison/TRAINED MODELS/swinv2_tiny_oakwilt.pth"
MODEL_NAME       = "swinv2_tiny_window8_256"
CLASS_NAMES      = [
    "There's No Oak Wilt in this Image",
    "There's Oak Wilt in this Image"
]
IMG_SIZE         = 256
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD        = 0.75
MARKER_ICON_PATH = "red.png"  # must live next to app.py

# ─────────────────────────────────────────────────
# 2) MODEL LOADING
# ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

model = load_model()

# ─────────────────────────────────────────────────
# 3) PREPROCESSOR
# ─────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

# ─────────────────────────────────────────────────
# 4) PER-IMAGE PROCESSING (CACHED)
# ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_image(path: str):
    # load image
    img = Image.open(path).convert("RGB")
    # inference
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    conf  = float(probs[1])
    label = CLASS_NAMES[1] if conf >= THRESHOLD else CLASS_NAMES[0]
    # extract GPS
    raw = open(path, "rb").read()
    tags = exifread.process_file(io.BytesIO(raw), details=False)
    gps = None
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        def to_deg(vals):
            d = float(vals[0].num)/vals[0].den
            m = float(vals[1].num)/vals[1].den
            s = float(vals[2].num)/vals[2].den
            return d + m/60 + s/3600
        lat = to_deg(tags['GPS GPSLatitude'].values)
        if tags['GPS GPSLatitudeRef'].values != 'N': lat = -lat
        lon = to_deg(tags['GPS GPSLongitude'].values)
        if tags['GPS GPSLongitudeRef'].values != 'E': lon = -lon
        gps = (lat, lon)
    return label, conf, gps

# ─────────────────────────────────────────────────
# 5) UI
# ─────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Oak-Wilt Detector & Geo-Mapper")
st.write(f"*Only detections ≥{int(THRESHOLD*100)}% confidence are mapped.*")

# directory chooser
img_dir = st.text_input(
    "Local directory containing JPG/PNG images",
    value="E:/Research/Model Comparison/Demo4_Dataset"
)

if img_dir and os.path.isdir(img_dir):
    # scan for image files
    exts = {".jpg",".jpeg",".png"}
    files = [
        os.path.join(img_dir,f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]
    st.write(f"Found {len(files)} images.")

    # process all (cached)
    with st.spinner("Running inference... this happens only once per image"):
        data = [ (fp,) + process_image(fp) for fp in files ]
        # data elements: (path, label, conf, gps)

    # filtering
    choice = st.selectbox(
        "Filter by prediction", ["All"] + CLASS_NAMES, index=0
    )
    if choice != "All":
        data = [d for d in data if d[1]==choice]

    # pagination
    per_page = st.number_input("Images per page", min_value=5, max_value=100, value=10)
    total    = len(data)
    pages    = math.ceil(total / per_page)
    page     = st.slider("Page", 1, pages, 1)
    start, end = (page-1)*per_page, page*per_page
    subset  = data[start:end]

    # display table-like layout
    st.subheader(f"Showing {len(subset)} of {total} results (Page {page}/{pages})")
    for path, label, conf, gps in subset:
        c0,c1,c2,c3,c4 = st.columns([1,3,2,1,1])
        with c0:
            thumb = Image.open(path)
            thumb.thumbnail((80,80))
            st.image(thumb)
            with st.expander("View larger"):
                st.image(path, use_column_width=True)
        c1.write(os.path.basename(path))
        c2.write(label)
        c3.write(f"{conf:.4f}")
        c4.write(f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else "—")

    # map high-confidence positives
    positives = [
        gps for _,lab,conf,gps in data
        if lab==CLASS_NAMES[1] and gps
    ]
    if positives:
        avg_lat = sum(lat for lat,lon in positives)/len(positives)
        avg_lon = sum(lon for lat,lon in positives)/len(positives)
        m = folium.Map((avg_lat,avg_lon), zoom_start=12)
        marker_cluster = MarkerCluster().add_to(m)
        icon = folium.CustomIcon(MARKER_ICON_PATH, icon_size=(30,30))
        for (lat,lon) in positives:
            folium.Marker((lat,lon), icon=icon).add_to(marker_cluster)
        st.subheader("Mapped High-Confidence Oak-Wilt Detections")
        st_folium(m, width=700, height=500)
    else:
        st.info("No high-confidence oak-wilt detections to map.")

else:
    if img_dir:
        st.error("Directory not found. Please check the path.")
