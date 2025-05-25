from flask import Flask, request, jsonify
from tokenizer import Tokenizer

app = Flask(__name__)

tokenizer = Tokenizer(None, False, 'bpe', merges=None)

@app.route("/tokenize", methods=["POST"])
def tokenize():
    data = request.get_json()
    text = data.get("text", "")
    tokens = tokenizer.tokenize(text)
    return jsonify({"tokens": tokens})

@app.route("/", methods=["GET"])
def home():
    return "Tokenizer API is running!", 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
