import os, uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from skin_tone import classify_monk_v9_5

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "static", "uploads")
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please upload an image.")
        return redirect(url_for("index"))

    if not allowed(file.filename):
        flash("Unsupported file type. Use JPG/PNG/WebP.")
        return redirect(url_for("index"))

    # save upload
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # run your classifier
    try:
        result = classify_monk_v9_5(save_path, debug=False)
    except Exception as e:
        # if anything crashes, show a friendly message
        print("[error] analyze:", e)
        flash("Could not analyze the photo. Try a clearer, front-facing image.")
        return redirect(url_for("index"))

    # ensure keys exist
    tone        = result.get("tone", 5)
    undertone   = result.get("undertone", "Neutral")
    best_colors = result.get("best_colors", []) or []
    avoid_colors= result.get("avoid_colors", []) or []

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{filename}"),
        tone=tone,
        undertone=undertone,
        best_colors=best_colors,
        avoid_colors=avoid_colors,
    )

if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
