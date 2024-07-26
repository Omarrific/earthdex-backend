import os
import openai
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

api_key = xxx
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable must be set")

openai.api_key = api_key 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = tf.keras.applications.MobileNetV2(weights='imagenet')

current_image_name = None
processing_start_time = None
species_description = None

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

def generate_species_description(species_name):
    try:
        response = openai.ChatCompletion.create(
            model="text-davinci-003",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Provide a brief description for the species: {species_name}."}
            ],
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"Error generating species description: {e}")
        return "Description not available."

def log_processing_time():
    global processing_start_time
    while True:
        if current_image_name and processing_start_time:
            elapsed_time = time.time() - processing_start_time
            logger.info(f"Image '{current_image_name}' has been processed for {elapsed_time:.2f} seconds")
        time.sleep(60)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        logger.info("GET request to /")
        return render_template_string('''
            <html>
            <body>
                <h1>Welcome to the Earthdex API</h1>
                <p>Please use the POST method to upload an image for the Earthdex, or you can go to this link: https://earthdex.vercel.app/ </p>
            </body>
            </html>
        ''')

    if request.method == 'POST':
        logger.info("Received POST request to /")
        if 'file' not in request.files:
            logger.error('No file provided')
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        global current_image_name
        current_image_name = file.filename

        if file:
            try:
                logger.info(f"Processing file: {current_image_name}")

                global processing_start_time
                processing_start_time = time.time()
                image = Image.open(io.BytesIO(file.read()))
                preprocessed_image = preprocess_image(image)

                predictions = model.predict(preprocessed_image)
                decoded_predictions = decode_predictions(predictions)

                # Get the top prediction
                top_prediction = decoded_predictions[0]
                species_name = top_prediction['label']
                
                global species_description
                species_description = generate_species_description(species_name)

                logger.info("Prediction successful")
                return jsonify({
                    'label': species_name,
                    'description': species_description,
                    'score': top_prediction['score']
                })
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                return jsonify({'error': 'An error occurred while processing the file.'}), 500
        else:
            logger.error('Invalid file')
            return jsonify({'error': 'Invalid file'}), 400

@app.route('/description', methods=['GET'])
def description():
    species = request.args.get('species')
    if not species:
        return jsonify({'error': 'No species provided'}), 400

    description_text = generate_species_description(species)
    return jsonify({'description': description_text})

@app.route('/status', methods=['GET'])
def status():
    elapsed_time = 0
    if current_image_name and processing_start_time:
        elapsed_time = time.time() - processing_start_time
    return render_template_string('''
        <html>
        <body>
            <h1>Backend Status</h1>
            <p>Currently processing image: {{ image_name }}</p>
            <p>Processing time: {{ elapsed_time }} seconds</p>
        </body>
        </html>
    ''', image_name=current_image_name or "No image being processed", elapsed_time=round(elapsed_time, 2))

if __name__ == '__main__':
    log_thread = Thread(target=log_processing_time)
    log_thread.daemon = True
    log_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)
