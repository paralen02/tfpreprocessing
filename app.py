from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from PIL import Image, ImageOps
import json

app = Flask(__name__)
CORS(app)

# URLs
TENSORFLOW_SERVE_URL = "https://beefsensemodels-863994867283.southamerica-west1.run.app/v1/models/saved3_model:predict"
SPRING_BOOT_API_URL = "https://beefsenseapiv2-863994867283.southamerica-west1.run.app:8080/clasificaciones"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Missing JWT token'}), 401

        image = Image.open(file.stream).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = json.dumps({"instances": [normalized_image_array.tolist()]})

        response = requests.post(
            TENSORFLOW_SERVE_URL, data=data, headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            print(f"Error from TensorFlow Serve: {response.text}")
            return jsonify({'error': 'Failed to get prediction from TensorFlow Serve', 'details': response.text}), 500

        prediction = response.json()['predictions'][0]
        class_index = np.argmax(prediction)
        confidence_score = prediction[class_index]

        category_mapping = {0: "primera", 1: "segunda", 2: "tercera", 4: "industrial"}
        category = category_mapping.get(class_index, "Unknown")

        spring_boot_response = requests.post(
            SPRING_BOOT_API_URL,
            json={'categoria': category, 'precision': confidence_score},
            headers={"Authorization": token, "Content-Type": "application/json"}
        )

        if spring_boot_response.status_code != 200:
            print(f"Error from Spring Boot API: {spring_boot_response.text}")
            return jsonify({'error': 'Failed to forward data to Spring Boot API', 'details': spring_boot_response.text}), 500

        return spring_boot_response.json()

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
