# Oak-Wilt Detector & Geo-Mapper

A Streamlit application for detecting oak wilt disease in tree images using a pretrained Swin Transformer model, extracting GPS metadata, and visualizing high-confidence detections on an interactive map.

## 🚀 Features

- **Image Classification**  
  Uses a fine-tuned `swinv2_tiny_window8_256` model to classify uploaded JPG/PNG images as “Oak Wilt” vs. “No Oak Wilt” with a configurable confidence threshold.

- **EXIF GPS Extraction**  
  Parses EXIF metadata to extract latitude/longitude from images taken with GPS-enabled cameras.

- **Interactive Map**  
  Plots high-confidence “Oak Wilt” detections on a Folium map with clustering support.

- **Pagination & Filtering**  
  Browse large image sets page by page; filter results by predicted class.

- **Two App Variants**  
  - **`app.py`**: Single-session file uploader.  
  - **`app2.py`**: Directory-based batch processing UI.

## 📁 File Structure

.
├── app.py # Streamlit app: upload & classify individual images
├── app2.py # Streamlit app: process a local directory of images
├── red.png # Marker icon for Folium map
├── requirements.txt # Python dependencies
└── README.md # This file

markdown
Copy
Edit

## 🔧 Prerequisites

- Python 3.8+  
- GPU recommended (CUDA) but not required  
- Git LFS (optional, for large model files)

## ⚙️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/oak-wilt-detector.git
   cd oak-wilt-detector
Create & activate virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download pretrained model
The SwinV2 tiny model weights (swinv2_tiny_oakwilt.pth) can be found here:
https://drive.google.com/drive/folders/1Ga4BZxb08oR-Fg8uTnfxtpx3SUNauCMq?usp=sharing
Place the .pth file in models/ (or update MODEL_PATH in the apps accordingly).

Ensure red.png is alongside app.py / app2.py
This icon is used for map markers.

▶️ Usage
1. Run the single‐upload app (app.py)
bash
Copy
Edit
streamlit run app.py
Use the file uploader to select one or more images.

View predictions, confidence scores, and GPS coordinates in a table.

High-confidence positives are plotted on an interactive Folium map.

2. Run the batch‐processing app (app2.py)
bash
Copy
Edit
streamlit run app2.py
Enter the path to a local directory containing JPG/PNG images.

The app will scan, classify, and page through results.

Filter by class, adjust “images per page”, and view map of high-confidence detections.

🔧 Configuration
Variable	Description	Default
MODEL_NAME	Timm model architecture name	swinv2_tiny_window8_256
MODEL_PATH	Path to the .pth weights file	Update to your local/model folder
IMG_SIZE	Input size for resizing/cropping	256
THRESHOLD	Confidence threshold for “Oak Wilt” label (0.0–1.0)	0.75
MARKER_ICON_PATH	File path for the Folium marker icon	red.png

Edit these constants at the top of app.py/app2.py as needed.

📦 Dependencies
streamlit

torch

timm

Pillow

exifread

folium

streamlit-folium

🤝 Contributing
Fork this repository

Create a feature branch (git checkout -b feature/YourFeature)

Commit your changes (git commit -m "Add YourFeature")

Push to your branch (git push origin feature/YourFeature)

Open a Pull Request

Please follow PEP 8 and write meaningful commit messages.

📜 License
This project is released under the MIT License. See LICENSE for details.