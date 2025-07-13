import sys
import os

# Menambahkan path ke sys.path untuk mencari modul di lokasi tertentu
path = '/Users/moham/OneDrive/Documents/CODE/FlaskProj'
sys.path.append(path)

# Impor Flask setelah menambahkan path
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask! Aplikasi berjalan dengan path khusus."

# Endpoint untuk menerima permintaan POST dengan data JSON
@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json  # Mendapatkan data JSON dari request
        # Menampilkan data yang diterima
        return jsonify({"message": "Data received successfully", "data": content}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 400

if __name__ == '__main__':
    # Jalankan server Flask pada port 5000
    app.run(debug=True)
