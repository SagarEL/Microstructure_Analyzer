# 🧊 Microstructure Image Analyzer

A **Streamlit-based web application** for analyzing SEM/TEM microstructure images.  
It performs **preprocessing, grain segmentation, feature extraction, and visualization** of microstructural properties.  
Includes a machine learning model (`model.pkl`) for predicting material properties from extracted features.

---

## 🚀 Features

- **Upload Images** (PNG, JPG, TIF) or select sample images.
- **Image Preprocessing** for noise removal and enhancement.
- **Grain Segmentation** using advanced image processing techniques.
- **Statistical Analysis** of grain properties:
  - Equivalent diameter
  - Aspect ratio
  - Area distribution
  - Shape and size classification
- **Interactive Plots**:
  - Grain size distribution histograms
  - Aspect ratio vs. equivalent diameter scatter plot
  - Shape and size distribution pie charts
- **Comparison Mode**:
  - Compare two images side-by-side
  - See differences in grain count and properties
- **Export Results** as CSV
- **ML Prediction** (example) from extracted features

---

---

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/microstructure-analyzer.git
cd microstructure-analyzer
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶ Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Open the app**  
Go to `http://localhost:8501` in your browser.

3. **Upload an image** or select a sample image to start the analysis.

---
 
## 📊 Machine Learning Model

- `train_model.py` contains example code to train a regression model predicting material properties based on grain features.
- The default `model.pkl` is trained on dummy data and should be replaced with a real trained model for actual use.

**To retrain the model:**
```
python train_model.py
```

---




