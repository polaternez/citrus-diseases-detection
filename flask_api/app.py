from flask import Flask, request, render_template
import tensorflow as tf
from keras.models import load_model
from PIL import Image


class_names = ["Black Spot", "Canker", "Greening", "Healthy"]

# Load model
model = load_model("models/citrus_model.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Function to pre-process the image
def preprocess_image(img):
    img = Image.open(img)
    img = img.resize((256, 256))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = img_arr.reshape((1, 256, 256, 3))
    img_arr = tf.cast(img_arr, tf.float32) / 255

    return img_arr


# WSGI
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# API endpoint to predict the class of an image
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['imageInput']

        # Pre-process the image
        prep_image = preprocess_image(image_file)

        # Make predictions using the model
        prediction = model.predict(prep_image)

        # Get the predicted class and confidence
        predicted_class = class_names[prediction.argmax()]
        confidence = "{:.2f} %".format(prediction.max() * 100)
        return render_template("index.html",
                                predicted_class=predicted_class,
                                confidence=confidence)
        

if __name__ == "__main__":
    app.run(debug=True)


