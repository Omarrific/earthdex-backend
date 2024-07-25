from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import logging
import time
from threading import Thread

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

def log_processing_time(image_name):
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        logger.info(f"Image '{image_name}' has been processed for {elapsed_time:.2f} seconds")
        time.sleep(60)  # Log every minute

@app.route('/', methods=['POST'])
def predict():
    logger.info("Received request")
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.error('No file provided')
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        image_name = file.filename

        if file:
            try:
                logger.info(f"Processing file: {image_name}")
                processing_thread = Thread(target=log_processing_time, args=(image_name,))
                processing_thread.daemon = True
                processing_thread.start()

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

@app.route('/status')
def status():
    return render_template_string('''
        <html>
        <body>
            <h1>Backend Status</h1>
            <p>Currently processing image: {{ image_name }}</p>
        </body>
        </html>
    ''', image_name='No image being processed')

if __name__ == '__main__':
    app.run(debug=True)
