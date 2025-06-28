from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.utils import get_custom_objects
from keras.mixed_precision import Policy
from keras.preprocessing import image
import os
import logging

# Register the DTypePolicy to fix the loading issue
get_custom_objects()['DTypePolicy'] = Policy('float32')

# List of butterfly class names (replace with your actual class names if needed
class_names = [
    "CLOUDYWING","AMERICAN SNOOT", "AN 88", "APPOLLO",
    "ARCIGERA FLOWER MOTH", "ATALA", "ATLAS MOTH", "BANDED ORANGE HELICONIAN", "BANDED PEACOCK",
    "BECKERS WHITE", "BIRD CHERRY ERMINE MOTH", "BLACK HAIRSTREAK", "BLUE MORPHO", "BLUE SPOTTED CROW",
    "BROWN SIPROETA", "CABBAGE WHITE", "CAIRNS BIRDWING", "CHALKHILL BLUE", "CLEOPATRA",
    "CLODIUS PARNASSIAN", "COMMON BANDED AWL", "COMMON WOOD-NYMPH", "COPPER TAIL",
    "CRECENT", "CRIMSON PATCH", "DANAID EGGFLY", "EASTERN COMA", "EASTERN DAPPLE WHITE",
    "EASTERN PINE ELFIN", "ELBOWED PIERROT", "GARDEN TIGER MOTH", "GIANT LEOPARD MOTH", "GLASSWINGED BUTTERFLY",
    "GREAT EGGFLY", "GREAT JAY", "GREEN CELLED CATTLEHEART", "GREEN HAIRSTREAK", "GREY HAIRSTREAK",
    "HAMMOCK SKIPPER", "HELICONIUS BUTTERFLY", "HIBISCUS HARLEQUIN BUG", "HORACE DUSKYWING", "INDRA SWALLOW",
    "IO MOTH", "JULIA BUTTERFLY", "LARGE MARBLE", "LUNA MOTH", "MALACHITE",
    "MANGROVE SKIPPER", "MESTRA", "METALMARK", "MILBERTS TORTOISESHELL", "MONARCH",
    "MOURNING CLOAK", "ORANGE OAKLEAF", "ORCHARD SWALLOWTAIL", "PAINTED LADY", "PAPER KITE",
    "PEACOCK", "PINE WHITE", "PIPEVINE SWALLOW", "POPINJAY", "PURPLISH COPPER",
    "QUESTION MARK", "RED ADMIRAL", "RED CRACKER", "RED POSTMAN", "RED SPOTTED PURPLE",
    "ROSY MAPLE MOTH", "SCARCE SWALLOW", "SILVER SPOT SKIPPER", "SIXSPOT BURNET MOTH", "SLEEPY ORANGE",
    "SOOTYWING", "SOUTHERN DOGFACE", "STRAITED QUEEN", "TROPICAL LEAFWING", "TWO BARRED FLASHER",
    "ULYSES", "VICEROY", "WHITE LINED SPHINX MOTH", "WOOD SATYR", "YELLOW SWALLOW TAIL",
    "ZEBRA LONG WING"
]

# Initialize Flask app
app = Flask(__name__)

# Load model
try:
    model = load_model('vgg16_model.keras')
    print("✅ Model loaded successfully.")
except Exception as e:
    logging.error("❌ Error loading model", exc_info=e)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file!', 400

    # Save uploaded file
    img_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(img_path)

    # Process image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0


    prediction = model.predict(x)
    class_idx = int(np.argmax(prediction[0]))
    class_name = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

    return render_template('output.html', prediction=class_name, class_idx=class_idx)


if __name__ == '__main__':
    app.run(debug=True)


