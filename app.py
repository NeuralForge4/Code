import os
from flask import Flask, jsonify, render_template, request
from analyzer import analyze_source, get_ml_artifact

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    artifact = get_ml_artifact()
    return jsonify({
        "status": "ok",
        "model_loaded": artifact is not None,
        "metrics": artifact.get("metrics", {}) if artifact else {},
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    code = payload.get("code", "")
    language = payload.get("language", "auto")
    filename = payload.get("filename", "")

    if not isinstance(code, str) or not code.strip():
        return jsonify({"error": "No source code provided."}), 400
    if len(code) > 200_000:
        return jsonify({"error": "Source code exceeds 200 KB limit."}), 400

    try:
        report = analyze_source(code=code, language=language, filename=filename)
        return jsonify(report)
    except Exception as exc:
        return jsonify({"error": f"Analysis engine error: {str(exc)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
