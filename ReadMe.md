# Oak-Wilt Detector & Geo-Mapper (Streamlit)

Fast, lightweight Streamlit app to detect **Oak Wilt (OW)** in images using a fine-tuned **SwinV2-Tiny** model. The app supports **single/multi image uploads** and **local folder scanning**, extracts **GPS EXIF** when available, and renders detections on an interactive **Folium** map.

> **Label mapping is fixed and explicit:**  
> `NOW = 0` (index 0) → “There’s No Oak Wilt in this Image”  
> `OW  = 1` (index 1) → “There’s Oak Wilt in this Image”

---

## ✨ Features

- **Two modes**: Upload images (multi-file) or scan a local folder.
- **Deterministic label mapping** (NOW=0, OW=1) baked into the UI and logic.
- **Confidence threshold slider** (default 0.75) controls what gets mapped.
- **GPS EXIF extraction** (if present) to pin detections on a **Folium** map.
- **Fast & responsive**: Model + data caching; GPU used if available.

---

## 🧠 Model & Data

- **Trained model** (PyTorch `state_dict`):
  - `D:\DNR\Streamlit App\swinv2_tiny_oakwilt25.pth` (saved in my local environment)
  - Cloud copy: **Google Drive**  
    👉 [Download swin_tiny_model.pth](https://drive.google.com/drive/folders/1Ga4BZxb08oR-Fg8uTnfxtpx3SUNauCMq?usp=drive_link)

- **Sample images** (optional):  
  👉 [Download the sample data](https://drive.google.com/drive/folders/11WMiFoaSgoMpJaXgrz7N1qRDuYiQ53t8?usp=sharing)

- **Backbone**: `swinv2_tiny_window8_256` (from `timm`)  
- **Num classes**: 2 (`NOW`, `OW`)

---

## 📦 Repository Structure (suggested)

```

oak-wilt-streamlit/
├─ app2.py                      # Streamlit app (your provided code)
├─ requirements.txt
├─ README.md
├─ assets/
│   └─ red.png                  # Optional custom map marker (placed next to app2.py is also fine)
└─ models/
└─ swinv2_tiny_oakwilt25.pth  # (Optional: if you prefer to keep the model inside repo folder)

```

> If you keep the model in `models/`, update `MODEL_PATH` in `app2.py`.

---

## 🔧 Setup

### 1) Create & activate a virtual environment (Windows PowerShell)
```ps
python -m venv venv
.\venv\Scripts\Activate.ps1
````

### 2) Install dependencies

> Works with **Python 3.11.9**. The `requirements.txt` installs CPU PyTorch by default.

```ps
pip install -r requirements.txt
```

**GPU (optional):** If you have a CUDA setup, install the matching `torch/torchvision` from the official PyTorch instructions for your CUDA version, then install the rest of the requirements:

```ps
# Example (adjust to your CUDA version per pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps
```

---

## ⬇️ Get the Model & Samples

1. **Download** `swinv2_tiny_oakwilt25.pth` from Google Drive (link above).

2. Place it where `MODEL_PATH` in `app2.py` points to:

   ```py
   MODEL_PATH = r"D:\DNR\Streamlit App\swinv2_tiny_oakwilt25.pth"
   ```

   Or change `MODEL_PATH` to wherever you store it (e.g., `models/swinv2_tiny_oakwilt25.pth`).

3. (Optional) Download **sample images** and use them to test the app.

---

## 🚀 Run

```ps
streamlit run app2.py
```

* Open the “**Upload images**” tab to analyze JPG/PNG files directly.
* Or use the “**Scan a folder**” tab to point to a local directory.
* Use the **sidebar slider** to adjust the mapping **confidence threshold**.
* **High-confidence OW** detections with GPS metadata appear on the **Folium** map.

---

## 🧩 How It Works

* **Model loading (cached):**
  `@st.cache_resource` loads the model once per unique file (path, mtime, size) to avoid stale caches.

* **Preprocess:**
  Resize → CenterCrop (256) → ToTensor → Normalize (ImageNet stats).

* **Inference:**
  Softmax over logits → `probs[1]` is **OW** confidence (index 1 maps to OW by design).

* **Thresholding & Mapping:**
  If **OW confidence ≥ threshold**, label as OW; if GPS in EXIF is present, it appears on the map.

* **EXIF GPS Parsing:**
  Uses `exifread`. Supports typical `GPSLatitude/Longitude (+ Ref)` tags.

---

## 🏷️ Label Mapping (Important)

* **Order is enforced** in `CLASS_NAMES`:

  ```py
  CLASS_NAMES = [
      "There's No Oak Wilt in this Image",  # 0 = NOW
      "There's Oak Wilt in this Image",     # 1 = OW
  ]
  ```
* The app always interprets `probs[1]` as **OW confidence**, ensuring consistency with training.

---

## ⚡ Performance Tips

* Prefer **GPU** if available (`torch.cuda.is_available()`).
* Use **larger batch processing** only when scanning folders (current app processes per-image for UI responsiveness).
* Keep the **threshold** in a reasonable range (0.70–0.85) depending on recall/precision needs.
* If classes are visually similar:

  * Use **more diverse training data** and **hard augmentations**.
  * Consider **class-weighted loss** (during training) or **test-time augmentation** (TTA) for marginal gains.

---

## 🧰 Troubleshooting

* **`ModuleNotFoundError: No module named 'streamlit_folium'`**
  `pip install streamlit-folium`

* **`PermissionError` when loading model**

  * Ensure the path is correct and the file isn’t locked by another process.
  * Try opening the terminal **as Administrator**.
  * In `app2.py`, the model is opened with `"rb"` to reduce odd permission issues.

* **“Directory not found”** in the **Scan a folder** tab

  * Double-check the path exists and you have permissions.

* **Black map / no pins**

  * Ensure images actually contain **GPS EXIF**.
  * Only **OW** predictions above the threshold are mapped.

---

## 📝 Notes

* `red.png` (custom map marker) is optional. If not present, default markers are used.
* The app works without internet; Google Drive links are only for downloading the model/samples.
* The model detects everything right from the sample image folder except one: "IMG_0304.jpg"

---

## 🙏 Acknowledgments

* [PyTorch](https://pytorch.org/)
* [timm](https://github.com/huggingface/pytorch-image-models)
* [Streamlit](https://streamlit.io/)
* [Folium](https://python-visualization.github.io/folium/)
* [exifread](https://github.com/ianare/exif-py)

> **GPU users:** replace the `torch`/`torchvision` lines with the CUDA-specific wheels recommended by PyTorch for your CUDA version, then install the rest with `--no-deps`.

