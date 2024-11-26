from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model, Model
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model
model = load_model('solar_module_model_best.keras')

# Function to preprocess image
# Function to read and preprocess a single image
def read_and_preprocess_image(image_path, target_size=(227, 227)):  # Resize to 227x227
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    image_resized = image_rgb.resize(target_size, Image.Resampling.LANCZOS)
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize pixel values
    return image_array

# Map predicted class index to class name
class_names = ['mono', 'poly']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    # Save the file temporarily and preprocess
    file_path = f"temp_{file.filename}"
    file.save(file_path)
    test_image = read_and_preprocess_image(file_path)

    # Make prediction
    pred_probabilities = model.predict(test_image)
    predicted_class_index = np.argmax(pred_probabilities)
    predicted_class = class_names[predicted_class_index]
    predicted_prob = pred_probabilities[0][predicted_class_index]

    return render_template(
        'result.html',
        image_path=file_path,
        predicted_class=predicted_class,
        probability=f"{predicted_prob:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)

