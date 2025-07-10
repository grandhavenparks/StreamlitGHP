import io
import base64

import streamlit as st
import torch
import timm
import exifread
import folium

from PIL import Image
from torchvision import transforms
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
MARKER_ICON_PATH = "red.png"  # put red.png next to this script

# ─────────────────────────────────────────────────
# 2) LOAD MODEL
# ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(CLASS_NAMES)
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return m.to(DEVICE).eval()

model = load_model()

# ─────────────────────────────────────────────────
# 3) PREPROCESS & PREDICT
# ─────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),
                         (0.229,0.224,0.225)),
])

def predict_ow_confidence(img: Image.Image) -> float:
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return float(probs[1])  # index 1 = “Oak Wilt”

# ─────────────────────────────────────────────────
# 4) EXIF GPS EXTRACTION
# ─────────────────────────────────────────────────
def get_gps(raw_bytes):
    tags = exifread.process_file(io.BytesIO(raw_bytes), details=False)
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
        return lat, lon
    return None

# ─────────────────────────────────────────────────
# 5) STREAMLIT UI
# ─────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Oak-Wilt Detector")
st.subheader("Detect Oak Wilt in Tree Images")
st.write("A product by EDGE FORESTRY AI")
st.write(f"Only images with Oak Wilt confidence ≥ {int(THRESHOLD*100)}% will be mapped.")

uploaded = st.file_uploader(
    "Choose JPG/PNG images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if uploaded:
    # 1) Run inference on each upload
    results = []
    for up in uploaded:
        raw = up.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        conf  = predict_ow_confidence(img)
        label = CLASS_NAMES[1] if conf >= THRESHOLD else CLASS_NAMES[0]
        gps   = get_gps(raw)

        # data URI
        ext  = up.name.split('.')[-1].lower()
        mime = "image/jpeg" if ext in ["jpg","jpeg"] else f"image/{ext}"
        uri  = f"data:{mime};base64," + base64.b64encode(raw).decode()

        results.append({
            "name":  up.name,
            "img":   img,
            "uri":   uri,
            "label": label,
            "conf":  conf,
            "gps":   gps
        })

    # 2) Filter selector
    filter_opt = st.selectbox(
        "Filter by prediction",
        options=["All"] + CLASS_NAMES,
        index=0
    )
    if filter_opt != "All":
        filtered = [r for r in results if r["label"] == filter_opt]
    else:
        filtered = results

    # 3) Display filtered results
    st.subheader("Results")
    cols = st.columns([1,2,2,1,1])
    cols[0].write("Preview")
    cols[1].write("Filename")
    cols[2].write("Prediction")
    cols[3].write("Confidence")
    cols[4].write("GPS")

    for r in filtered:
        c0, c1, c2, c3, c4 = st.columns([1,2,2,1,1])
        with c0:
            st.image(r["img"], width=80)
            with st.expander("Enlarge"):
                st.image(r["img"], use_column_width=True)
        c1.markdown(f"[{r['name']}]({r['uri']})")
        c2.write(r["label"])
        c3.write(f"{r['conf']:.4f}")
        c4.write(f"{r['gps'][0]:.6f}, {r['gps'][1]:.6f}" if r["gps"] else "—")

    # 4) Map only high-confidence Oak Wilt among filtered
    positives = [
        r for r in filtered
        if r["label"] == CLASS_NAMES[1] and r["conf"] >= THRESHOLD and r["gps"]
    ]
    if positives:
        avg_lat = sum(lat for lat,lon in (p["gps"] for p in positives)) / len(positives)
        avg_lon = sum(lon for lat,lon in (p["gps"] for p in positives)) / len(positives)
        m = folium.Map(location=(avg_lat,avg_lon),
                       zoom_start=12,
                       tiles="OpenStreetMap")

        icon = folium.CustomIcon(MARKER_ICON_PATH, icon_size=(30,30))
        for p in positives:
            folium.Marker(
                location=p["gps"],
                popup=f"{p['name']}: {p['label']} ({p['conf']:.2f})",
                icon=icon
            ).add_to(m)

        st.subheader("Mapped Affected Trees")
        st_folium(m, width=700, height=500)
    else:
        st.info("No high-confidence Oak Wilt detections with GPS to map.")
