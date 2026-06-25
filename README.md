# Vistone: AI-Powered Personal Color Palette Matcher

<div align="center">
  <em>Discover the colors that love you back.</em>
</div>

---

## 🎯 Project Objective
**Vistone** is an advanced AI-powered web application designed to democratize personal color analysis. Traditionally, finding the "right" colors to wear requires expensive, manual consultations with professional stylists. Vistone automates this process by using state-of-the-art facial recognition, computer vision, and color science to analyze a user's skin tone and undertone, instantly providing a highly accurate, personalized color palette.

## ❗ Problem Statement
Determining a person's exact skin undertone from a digital photograph is notoriously difficult. Lighting conditions (warm indoor bulbs, cool overcast days, harsh flash) severely distort the true color of the skin captured by the camera. Furthermore, simple color averaging often includes shadows, highlights, facial hair, or the background, leading to inaccurate results. 

Vistone solves these issues through aggressive, algorithmic lighting correction and extremely precise facial landmark sampling.

---

## ⚙️ How It Works (The User Flow)
1. **Upload**: The user uploads a standard selfie to the Vistone web interface.
2. **Analysis**: Behind the scenes, the AI strips away bad lighting, finds the face, and samples pure skin pixels.
3. **Classification**: The skin is classified into one of the 10 shades of the **Google Monk Skin Tone Scale**, and the underlying temperature (Cool, Warm, Neutral) is calculated.
4. **Recommendation**: Based on the tone and undertone, Vistone references a strictly curated database built on **Seasonal Color Analysis (SCA)** principles to recommend 3 "Best" colors to wear and 3 colors to "Avoid".

---

## 🧠 Technical Architecture & Algorithms

Vistone is built on a highly sophisticated computer vision pipeline.

### 1. Face Detection & Landmark Extraction
- **MediaPipe Face Mesh**: Used to extract 468 dense 3D facial landmarks. Vistone maps specific polygon coordinates to isolate the cheeks, forehead, and nose.
- **Exclusion Masking**: The algorithm explicitly masks out the eyes, eyebrows, mouth, and beard zones to ensure only pure skin is sampled.
- **Fallback DNN**: An OpenCV SSD (Single Shot MultiBox Detector) Caffe model (`res10_300x300`) is used as a fast, robust fallback for initial face verification.

### 2. Dynamic Color Correction
To combat the problem of bad lighting, Vistone applies two major correction steps before analyzing the skin:
- **Shades of Gray White Balance**: Uses the Minkowski p-norm algorithm to estimate and remove the color cast of the illuminant (e.g., yellow indoor lighting).
- **Sclera-Based Exposure Normalization**: The algorithm locates the sclera (white of the eye) using HSV masking. By measuring the brightness of the sclera, Vistone calculates a dynamic exposure factor to artificially "relight" underexposed or overexposed skin to a scientifically standard brightness.

### 3. Precision Skin Sampling
- **Morphological Erosion**: To avoid edge blending (where skin meets hair or the background), the cheek masks undergo an inward morphological erosion.
- **Gaussian Mixture Model (GMM)**: Instead of simple averaging, a GMM (or a robust percentile median) is used on the pooled skin pixels to find the true dominant skin color, ignoring specular highlights (sweat/oil) and deep shadows.

### 4. Color Classification (CIE L*a*b*)
RGB is terrible for measuring human perception of color. Vistone converts all extracted pixels to the **CIE L*a*b* (D65)** color space.
- **Tone Classification (Monk Scale)**: The L* (Lightness) channel is compared against photo-calibrated anchor points for the 10 Google Monk Skin Tones.
- **Undertone Classification**: The a* (green-red) and b* (blue-yellow) channels are compared against curated Cool, Warm, and Neutral reference points using the **CIE Delta E 2000 (ΔE₀₀)** formula, which accurately mimics human vision color differences.

### 5. Professional Seasonal Color Analysis (SCA)
The final recommendation engine doesn't just guess colors. It maps the 30 possible combinations (10 Tones × 3 Undertones) to the 12 Seasonal Color Palettes (e.g., Light Spring, Deep Winter, Soft Autumn). The generated recommendations (`monk_skin_tone_color_recommendations.json`) use 100% professional fashion-industry hex codes (e.g., *Powder Blue*, *Rich Terracotta*, *Royal Purple*).

---

## 🛠️ Tech Stack

- **Backend / Server**: Python, Flask, Werkzeug
- **Computer Vision**: OpenCV (`cv2`), Google MediaPipe
- **Color Science**: `colormath` (Delta E CIE 2000), NumPy, SciPy
- **Frontend**: HTML5, TailwindCSS (via CDN), Vanilla JavaScript
- **Design Aesthetic**: Glassmorphism, CSS micro-animations, responsive grid layouts

---

## 🚀 Installation & Setup

1. **Clone the repository**
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   *(Requires OpenCV, MediaPipe, NumPy, SciPy, Flask, Colormath)*
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application**:
   ```bash
   python app.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000`

---
*Built with precision color science and AI.*
