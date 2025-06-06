from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenizer import Tokenizer
import traceback

app = Flask(__name__)
CORS(app)

tokenizer = Tokenizer(None, False, 'bpe', merges=None)

@app.route("/tokenize", methods=["POST"])
def tokenize():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        vocab = data.get("vocab")
        lowercase = data.get("lowercase", False)
        tokenizer_type = data.get("type", "bpe")
        merges = data.get("merges")

        tokenizer = Tokenizer(vocab, lowercase, tokenizer_type, merges)

        tokens = tokenizer.tokenize(text)
        return jsonify({"tokens": tokens})
    
    except Exception as e:
        print(f"Error in tokenize: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Tokenizer API is running!", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
