import os, uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from skin_tone import classify_monk_v9_5   # v9_5 alias → calls v10 internally

APP_ROOT   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "static", "uploads")
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["SECRET_KEY"]    = "change-me"
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

    # Save upload
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Run classifier (v10 via alias)
    try:
        result = classify_monk_v9_5(save_path, debug=False)
    except Exception as e:
        print("[error] analyze:", e)
        flash("Could not analyze the photo. Try a clearer, front-facing image.")
        return redirect(url_for("index"))

    tone          = result.get("tone", 5)
    undertone     = result.get("undertone", "Neutral")
    tone_conf     = result.get("tone_confidence", 0.0)
    ut_conf       = result.get("ut_confidence", 0.0)
    best_colors   = result.get("best_colors",  []) or []
    avoid_colors  = result.get("avoid_colors", []) or []

    return render_template(
        "result.html",
        image_url    = url_for("static", filename=f"uploads/{filename}"),
        tone         = tone,
        undertone    = undertone,
        tone_conf    = tone_conf,
        ut_conf      = ut_conf,
        best_colors  = best_colors,
        avoid_colors = avoid_colors,
        soft_warning = False,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
