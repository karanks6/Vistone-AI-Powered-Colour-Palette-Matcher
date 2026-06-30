<div align="center">

# Vistone

### AI-Powered Personal Color Palette Matcher

*Discover the colors that love you back.*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/Google_MediaPipe-blue?style=flat-square)](https://mediapipe.dev)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [How It Works — User Flow](#how-it-works--user-flow)
- [Technical Pipeline — Deep Dive](#technical-pipeline--deep-dive)
  - [Stage 1 — Face Detection & Landmark Extraction](#stage-1--face-detection--landmark-extraction)
  - [Stage 2 — Lighting & Color Correction](#stage-2--lighting--color-correction)
  - [Stage 3 — Precision Skin Sampling](#stage-3--precision-skin-sampling)
  - [Stage 4 — Skin Tone Classification (Monk Scale)](#stage-4--skin-tone-classification-monk-scale)
  - [Stage 5 — Undertone Analysis](#stage-5--undertone-analysis)
  - [Stage 6 — Color Recommendation (Seasonal Color Analysis)](#stage-6--color-recommendation-seasonal-color-analysis)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [API Reference](#api-reference)
- [Accuracy & Design Decisions](#accuracy--design-decisions)

---

## Overview

**Vistone** is a full-stack AI web application that automates professional personal color analysis — a service that has historically required an expensive visit to a trained stylist. 

By uploading a single selfie, a user receives:
- Their precise **Monk Skin Tone** (1–10 on the Google MST Scale)
- Their **undertone** classification (Cool, Warm, or Neutral)
- **3 recommended clothing colors** that scientifically harmonize with their complexion
- **3 colors to avoid** that clash with their skin tone and undertone

Vistone v12.0 is built on a multi-stage computer vision pipeline that aggressively corrects for real-world photography conditions — lighting variations, camera gamma, and specular skin reflections — to deliver dermatologically accurate skin tone measurements from ordinary selfies.

---

## Problem Statement

Determining a person's true skin tone and undertone from a digital photograph is deceptively difficult. Several compounding factors conspire to make this inaccurate with naive approaches:

| Problem | Consequence |
|---|---|
| **Warm indoor lighting** | Shifts skin pixels yellow, inflating the "warm" channel and pushing classification to a darker, warmer Monk tone |
| **Camera gamma correction** | Digital cameras apply non-linear gamma curves that make dark skin appear significantly lighter than it truly is |
| **Specular highlights** | Skin oils produce bright hotspots on cheeks that massively inflate the measured L* (lightness) value |
| **Shadows & facial hair** | Shadows around the jaw, stubble, and eyebrows darken the pixel pool, dragging the classification toward a darker Monk tone |
| **Simple averaging** | Averaging all face pixels ignores region suitability — the forehead has different texture/reflectance than the cheeks |

Vistone addresses every one of these problems through a dedicated algorithmic stage in its pipeline.

---

## How It Works — User Flow

```
[User Uploads Selfie]
        │
        ▼
[1. Face & Landmark Detection]  ←  MediaPipe Face Mesh (468 landmarks)
        │
        ▼
[2. Lighting Correction]        ←  Shades-of-Gray WB + Sclera Normalization
        │
        ▼
[3. Precision Skin Sampling]    ←  Morphological Erosion + GMM Dominant Color
        │
        ▼
[4. Monk Tone Classification]   ←  CIE L*a*b* + L*-Anchored 2-Stage Classifier
        │
        ▼
[5. Undertone Analysis]         ←  CIE ΔE 2000 vs. Tone-Stratified References
        │
        ▼
[6. Color Recommendations]      ←  Seasonal Color Analysis (SCA) Lookup
        │
        ▼
[Render Result Page with Monk Scale Visualization + Color Swatches]
```

---

## Technical Pipeline — Deep Dive

### Stage 1 — Face Detection & Landmark Extraction

**Goal:** Locate the face and extract exactly the skin regions we want to sample, while explicitly excluding everything that isn't skin.

**Tools used:**
- **Google MediaPipe Face Mesh** — Detects 468 3D facial landmarks in a single pass. Vistone uses a specific subset of these landmarks to draw precise polygons over the upper cheeks, forehead, and nose bridge.
- **OpenCV SSD DNN** — A Caffe-based Single Shot MultiBox Detector (`res10_300x300_ssd_iter_140000`) acts as an initial face verification gate. If no face is detected with sufficient confidence, the pipeline aborts early.

**Exclusion masking:**

Before sampling any pixels, Vistone creates exclusion masks for every non-skin region and subtracts them from the sampling zone:

```
Excluded Regions
├── Eyes (left + right ring, dilated by 16px)
├── Eyebrows (left + right, dilated by 14px)
├── Mouth (outer polygon, dilated by 18px)
└── Beard/jaw zone (removed from cheek polygon entirely)
```

The result is a binary mask covering only the pure, unobstructed skin of the upper cheeks — the most reliable region for undertone measurement.

---

### Stage 2 — Lighting & Color Correction

**Goal:** Remove the color cast introduced by the illuminant (light source) and normalize exposure, so the same person's skin reads the same L* whether photographed indoors or outdoors.

#### 2a. Shades of Gray White Balance

This algorithm estimates the color of the light source from the image itself and then mathematically removes it, leaving neutral-white illuminated skin.

**How it works:**
1. Compute the p-norm (Minkowski norm, p=6) of each color channel (R, G, B) across the entire image. This estimates the "dominant" color of the light.
2. Divide each channel by its norm, scaled to the white point. This shifts the illuminant toward neutral D65 (standard daylight).
3. A `max_shift=0.15` cap prevents over-correction on already correctly-lit photos.

**Why not simple gray-world?** Simple gray-world averaging assumes the average color of the image is gray, which fails badly on images with one dominant color (e.g., a photo in a green field). The Minkowski norm is more robust because it weights bright pixels more heavily, making it a better estimator of the illuminant.

#### 2b. Sclera-Based Exposure Normalization

Even after white balancing, the overall brightness of the photo can vary dramatically. A photo taken in dim lighting will read a person's Monk 1 skin as if it were Monk 4.

Vistone solves this by using the **sclera (white of the eye)** as an internal calibration target, since the sclera has a known, relatively stable luminance:

1. The sclera region is isolated by finding white pixels (`V > 120` in HSV space) within the eye exclusion mask.
2. The mean RGB brightness of the sclera is measured.
3. An **exposure factor** is calculated: `exposure_factor = 210.0 / mean_sclera_brightness` (targeting L* ≈ 92–95 for the sclera).
4. This factor is clamped to `[0.85, 1.50]` to avoid extreme corrections and only applied to photos where `skin_L_mean >= 55.0` (fair/medium skin), since dark skin does not benefit from this type of correction.

---

### Stage 3 — Precision Skin Sampling

**Goal:** Extract only the most representative skin pixels from the corrected image, aggressively filtering out highlights, shadows, and facial hair.

#### Multi-Region Pixel Pool

Vistone samples from four distinct facial regions, prioritized in order of reliability:

1. **Inner cheeks** (highest priority) — The cheek polygon is morphologically eroded inward by 10px to avoid edge pixels where skin transitions to hair or background.
2. **Full cheeks** — The wider cheek region, filtered by luminance percentiles.
3. **Forehead** — A central strip, cropped to exclude the top 12% (hairline zone).
4. **Nose bridge** — A tight polygon, useful for medium/dark skin tones where cheek sampling may be limited.

#### YCrCb Skin Pixel Filter

Each pixel sampled is passed through a chrominance gate in YCrCb color space. Human skin has a well-defined range of Cr (red-difference) and Cb (blue-difference) chrominance values regardless of lightness. Pixels outside this range (hair, background intrusions, eyebrow stubs) are discarded before sampling.

#### Percentile Luminance Band

To further reject highlights and shadows:
- **Fair skin (L* > 75):** Sample the **50th–90th** brightness percentile (discard dark shadows at the bottom, keep the bright representational band)
- **Medium skin (L* 45–75):** Sample the **25th–75th** percentile
- **Dark skin (L* < 45):** Sample the **5th–35th** percentile (avoid specular highlights that can be dramatically brighter than true skin)

#### Gaussian Mixture Model (GMM) Dominant Color

Rather than averaging the pooled pixels (which would be skewed by any remaining noise), Vistone fits a **3-component Gaussian Mixture Model** to the pixel pool in CIE L*a*b* space. The component with the highest mixing weight (the dominant cluster) is selected as the representative skin color. This is equivalent to finding the single most common "type" of skin pixel, making it robust to small contaminations.

---

### Stage 4 — Skin Tone Classification (Monk Scale)

**Goal:** Map the extracted skin color to one of the 10 shades of the **Google Monk Skin Tone (MST) Scale** (Ellis et al., 2022).

The Monk Scale is a perceptually designed 10-shade scale created by Google to provide broader, more inclusive skin tone representation than older 6-shade scales like Fitzpatrick. Each shade has a defined hex color and a corresponding L* (lightness) value in CIE L*a*b* space.

#### Two-Stage L*-Anchored Classification

Vistone v12 uses a two-stage classification approach designed to be robust to the white balance errors that plagued earlier versions:

**Why not just measure ΔE (full color distance)?**
A pure ΔE₂₀₀₀ match against all 10 Monk swatches is very sensitive to the a* and b* channels (chroma). Warm indoor lighting shifts a photo's a*/b* values by 3–8 units, which can push a Monk 1 (very fair, cool) reading all the way to Monk 3 (light medium, warm). The L* channel, by contrast, is relatively immune to illuminant color shifts — only the brightness of the light changes L*, not its color.

**Stage 1 — L* Anchor**

The measured skin L* is compared against `MONK_L_PHOTO` — a set of **photo-calibrated L* anchors** that account for the fact that real photographs of dark skin appear measurably lighter than the reference swatches (due to camera gamma):

```python
MONK_L_PHOTO = [94.0, 92.0, 92.0, 87.0, 77.0, 52.0, 40.0, 33.0, 25.0, 19.0]
```

The closest Monk tone by L* distance alone is selected as the anchor. A window of ±2 tones around this anchor is opened for Stage 2.

**Stage 2 — ΔE₂₀₀₀ Fine-Tuning**

Within the ±2 tone window, Vistone computes the full **CIE Delta E 2000 (ΔE₀₀)** color distance between the sampled skin color and each candidate Monk swatch. ΔE₂₀₀₀ is the gold standard for perceptual color difference, designed to match how the human visual system perceives color changes (it is non-linear and accounts for the Euclidean shortcomings of raw L*a*b* distances).

The Monk tone with the minimum ΔE₂₀₀₀ within the window is the final classification. A **confidence score** is derived from the ratio of distances to the top two candidates.

---

### Stage 5 — Undertone Analysis

**Goal:** Classify the skin's underlying hue bias as **Cool** (pink/rosy/blue-shifted), **Warm** (golden/peachy/yellow-shifted), or **Neutral**.

Undertone analysis uses the **a*** (green↔red axis) and **b*** (blue↔yellow axis) channels from the skin's CIE L*a*b* representation — precisely the channels that encode the warmth or coolness of the skin.

Vistone maintains a tone-stratified reference table (`UNDERTONE_REFS`) with four groups (Monks 1–3, 4–5, 6–7, 8–10). Each group has three LabColor reference points representing the expected a*/b* values for Warm, Cool, and Neutral within that lightness range.

The undertone is determined by computing **ΔE₂₀₀₀** from the skin sample to each reference and selecting the minimum distance winner. The confidence score reflects how clearly the winner separates from the alternatives.

**Why stratify by tone group?**
Warm undertones on deep skin (Monk 9) have a completely different a*/b* signature than warm undertones on fair skin (Monk 1). A single global reference would be wildly inaccurate — a warm Monk 1 has b* ≈ 16, while a warm Monk 9 has b* ≈ 14 with a very different L*. Stratification ensures the reference is always calibrated to the correct lightness band.

---

### Stage 6 — Color Recommendation (Seasonal Color Analysis)

**Goal:** Map the (Monk tone, undertone) combination to a set of clothing colors that harmonize with the user's complexion.

Vistone uses **Seasonal Color Analysis (SCA)** — the same framework used by professional image consultants. SCA divides the human population into 12 seasonal archetypes based on value (light/dark) and hue (warm/cool):

| Monk Tones | Cool | Warm | Neutral |
|---|---|---|---|
| 1–3 (Light) | True Summer / Clear Winter | Light Spring | Soft Summer |
| 4–7 (Medium) | True Winter / Cool Summer | True Autumn | Soft Autumn |
| 8–10 (Deep) | Deep Winter | Deep Autumn | Warm Deep |

For each of the 30 combinations, the `monk_skin_tone_color_recommendations.json` database contains:
- **3 "Best" colors**: Fabric shades (real-world hex, not digital primaries) that bring out the skin's natural luminosity. E.g., *Powder Blue (#9AB2CD)*, *Rich Terracotta (#C65D47)*, *Royal Purple (#4B2D73)*.
- **3 "Avoid" colors**: Shades that clash with the undertone or wash out the complexion. E.g., *Warm Mustard* (clashes with cool undertones), *Frosty Pink* (clashes with warm autumn skin).

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | Python · Flask · Werkzeug |
| **Computer Vision** | OpenCV 4.x · Google MediaPipe |
| **Machine Learning** | scikit-learn (Gaussian Mixture Models) |
| **Color Science** | `colormath` (CIE L*a*b*, ΔE 2000) · NumPy · SciPy |
| **DNN Model** | Caffe SSD (res10_300x300) for face verification |
| **Frontend** | HTML5 · TailwindCSS · Vanilla JavaScript |
| **UI Design** | Glassmorphism · CSS animations · Responsive grid |

---

## Project Structure

```
Vistone/
├── app.py                                  # Flask application, routes (/,/analyze,/about)
├── skin_tone.py                            # Core v12.0 classification pipeline
├── image_quality.py                        # Photo quality pre-check (blur, face angle)
├── monk_skin_tone_color_recommendations.json  # SCA color database (30 combinations)
├── generate_colors_pro.py                  # Script to regenerate the SCA database
├── deploy.prototxt                         # Caffe SSD model architecture
├── res10_300x300_ssd_iter_140000.caffemodel   # Caffe SSD pretrained weights
├── templates/
│   ├── base.html                           # Shared layout, navigation
│   ├── index.html                          # Home / upload page
│   ├── result.html                         # Results page with Monk scale slider
│   └── about.html                          # About page
├── static/
│   ├── js/main.js                          # Frontend upload interaction
│   └── uploads/                           # Temporary uploaded images
├── tests/
│   ├── run_accuracy.py                     # Batch accuracy test across tone images
│   └── diag_undertone.py                   # Undertone ΔE diagnostics
└── images/                                 # Test reference images (tone1.jpg–tone10.jpg)
```

---

## Installation & Setup

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/your-username/vistone.git
cd vistone

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install flask opencv-python mediapipe scikit-learn colormath numpy scipy werkzeug

# 4. Run the application
python app.py
```

Open your browser and navigate to **http://127.0.0.1:5000**

> **Photo Tips for Best Accuracy:**
> - Use a front-facing photo in **natural daylight** or soft, even indoor light
> - Avoid heavy make-up, filters, or strong side-lighting
> - The face should be clearly visible and reasonably centered

---

## API Reference

### `classify_monk_v10(image_path, debug=False) → dict`

The core classification function. Accepts a file path to any JPG, PNG, or WebP image.

**Returns:**
```json
{
  "tone": 3,
  "undertone": "Warm",
  "tone_confidence": 0.912,
  "ut_confidence": 0.847,
  "best_colors": [
    { "name": "Warm Peach", "hex": "#F4B084" },
    { "name": "Soft Turquoise", "hex": "#40C0CB" },
    { "name": "Buttercup Yellow", "hex": "#F2D388" }
  ],
  "avoid_colors": [
    { "name": "Icy Blue", "hex": "#D0E4F2" },
    { "name": "Stark Black", "hex": "#1A1A1A" },
    { "name": "Magenta", "hex": "#C21E56" }
  ],
  "quality": { ... }
}
```

**Confidence interpretation:**
- `> 0.80` — High confidence, reliable result
- `0.45 – 0.80` — Moderate confidence; photo quality may be limiting
- `< 0.45` — Low confidence; front-facing photo in better lighting recommended

---

## Accuracy & Design Decisions

### Why CIE L*a*b* instead of RGB or HSV?

RGB is a device-dependent encoding. HSV is a mathematical rearrangement of RGB. Neither space reflects how the human eye perceives color differences. The **CIE L*a*b*** color space is designed so that equal numerical distances between two colors correspond to equal perceived differences by the human visual system. This is why both tone classification (L*) and undertone analysis (ΔE₂₀₀₀) are done in L*a*b*.

### Why photo-calibrated L* anchors (MONK_L_PHOTO)?

The Google Monk scale hex values are measured under controlled D65 illumination. Real photographs, especially of dark skin, consistently read 7–15 L* units *higher* than the reference swatches. This is because camera sensors apply gamma correction that compresses the shadow end of the tonal scale. If we compared against the raw swatch L* values, all Monk 8–10 classifications would be systematically off by 1–2 tones. MONK_L_PHOTO corrects for this camera-induced bias.

### Why GMM instead of mean/median for the dominant color?

A simple mean is easily corrupted by even a small percentage of stray pixels (e.g., a few dark eyebrow hairs inside the cheek mask). A median is better but still treats all pixels as equally valid. The GMM finds the *mode* of the pixel distribution — the single most dense cluster — making it fundamentally resistant to multi-modal contamination from highlights or shadows.

---

<div align="center">
*Built with precision color science, computer vision, and a genuine care for inclusive design.*
</div>