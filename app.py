from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app) 

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
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        
        if file:
            try:
                image = Image.open(io.BytesIO(file.read()))
                preprocessed_image = preprocess_image(image)
                
                predictions = model.predict(preprocessed_image)
                decoded_predictions = decode_predictions(predictions)

                return jsonify(decoded_predictions[0])
            
            except Exception as e:
                print(f"Error processing file: {e}")
                return jsonify({'error': 'An error occurred while processing the file.'}), 500
        else:
            return jsonify({'error': 'Invalid file'}), 400
    else:
        return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True)
