from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
MODEL_PATH = 'asl_mobilenetv2_best.keras'
model = load_model(MODEL_PATH)
print("✅ Model loaded.")
print("✅ Model expects input shape:", model.input_shape)

# Upload folder
UPLOAD_FOLDER = os.path.join('static', 'output')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
classes = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction.html')
def predict():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # ✅ Resize to 128x128 — this is the most important line!
        img = image.load_img(save_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        print("📸 Image shape passed to model:", img_array.shape)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]

        return render_template('logout.html', prediction=predicted_class, filename=filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
