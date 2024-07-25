from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def decode_predictions(predictions: np.ndarray, top=1):
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top)
    results = [{'label': d[1], 'score': float(d[2])} for d in decoded[0]]
    return results

@app.route('/', methods=['POST'])
def predict():
    logger.info("Received request")
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.error('No file provided')
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file:
            try:
                logger.info("Processing file")
                image = Image.open(io.BytesIO(file.read()))
                preprocessed_image = preprocess_image(image)

                predictions = model.predict(preprocessed_image)
                decoded_predictions = decode_predictions(predictions)

                logger.info("Prediction successful")
                return jsonify(decoded_predictions[0])
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                return jsonify({'error': 'An error occurred while processing the file.'}), 500
        else:
            logger.error('Invalid file')
            return jsonify({'error': 'Invalid file'}), 400
    else:
        logger.error('Method not allowed')
        return jsonify({'error': 'Method not allowed'}), 405

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the image classification API. Please use POST to upload an image.'})

if __name__ == '__main__':
    app.run(debug=True)
