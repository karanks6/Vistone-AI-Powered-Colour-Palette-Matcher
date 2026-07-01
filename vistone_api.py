"""
vistone_api.py — Vistone Mobile API  (port 8000)
=================================================
A SEPARATE, dedicated JSON REST API for the Vistone Flutter mobile app.
This file is completely independent from app.py (which serves the website on port 5000).

DO NOT MODIFY app.py — this file handles mobile clients only.

Endpoint:
  POST /api/analyze
    Body: multipart/form-data  { image: <file> }
    Returns: application/json  { tone, undertone, tone_confidence, ut_confidence,
                                  best_colors, avoid_colors, monk_colors }
"""

import os
import uuid
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from skin_tone import classify_monk_v10

# ── Flask setup ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins for mobile dev; restrict in production

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

# The 10 Google Monk Skin Tone hex colors (for the scale visualization)
MONK_COLORS = [
    "#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
    "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"
]


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check for the Flutter app to test connectivity."""
    return jsonify({"status": "ok", "service": "vistone-mobile-api", "version": "1.0.0"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze an uploaded face photo and return Monk tone + color recommendations.

    Form field: image  (JPG / PNG / WebP file)
    Returns:    JSON with tone, undertone, confidence scores, and color palettes.
    """
    file = request.files.get("image")

    if not file or file.filename == "":
        return jsonify({"error": "No image provided. Send an 'image' field in the form data."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, or WebP."}), 415

    # Save to a temporary file (auto-deleted after analysis)
    ext = file.filename.rsplit(".", 1)[1].lower()
    tmp_path = os.path.join(tempfile.gettempdir(), f"vistone_{uuid.uuid4().hex}.{ext}")

    try:
        file.save(tmp_path)
        result = classify_monk_v10(tmp_path, debug=False)
    except Exception as e:
        print(f"[mobile-api] Error during analysis: {e}")
        return jsonify({"error": "Analysis failed. Please use a clear, front-facing photo in good lighting."}), 422
    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    tone = result.get("tone", 5)
    undertone = result.get("undertone", "Neutral")
    tone_conf = result.get("tone_confidence", 0.0)
    ut_conf = result.get("ut_confidence", 0.0)
    best_colors = result.get("best_colors", []) or []
    avoid_colors = result.get("avoid_colors", []) or []

    # Validate face was detected (low confidence = likely no face found)
    if tone_conf == 0.0:
        return jsonify({"error": "Could not detect a face. Use a clear, front-facing photo."}), 422

    return jsonify({
        "tone": tone,
        "undertone": undertone,
        "tone_confidence": round(tone_conf, 3),
        "ut_confidence": round(ut_conf, 3),
        "best_colors": best_colors,
        "avoid_colors": avoid_colors,
        "monk_colors": MONK_COLORS,
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  Vistone Mobile API  —  http://0.0.0.0:8000")
    print("  Web app still runs separately on port 5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8000, debug=True)
