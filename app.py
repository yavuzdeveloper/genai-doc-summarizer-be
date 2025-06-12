# backend/app.py
from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import os



app = Flask(__name__)
CORS(app)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")



@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text", "")
    summary = summarizer(text, max_length=100, min_length=25, do_sample=False)
    return jsonify(summary[0])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
