import io, sqlite3
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
DB = "predictions.db"
_model = None

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model("model/fruit_model.keras")
    return _model

def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT, confidence REAL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

@app.route("/")
def index():
    with sqlite3.connect(DB) as c:
        rows = c.execute("SELECT filename,label,confidence,created FROM predictions ORDER BY id DESC LIMIT 10").fetchall()
    return render_template("index.html", history=rows)

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded"), 400
    img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((224, 224))
    p = float(get_model().predict(np.expand_dims(np.array(img), 0), verbose=0)[0][0])
    label = "rotten" if p >= 0.5 else "fresh"
    conf = p if p >= 0.5 else 1 - p
    with sqlite3.connect(DB) as c:
        c.execute("INSERT INTO predictions(filename,label,confidence) VALUES(?,?,?)", (f.filename, label, conf))
    return jsonify(label=label, confidence=round(conf * 100, 2))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
