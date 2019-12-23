# USAGE
# Start the server:
# 	python imageqcserver.py
# Submit a request via cURL:
# 	curl -X POST -F image=@image.png 'http://localhost:5000/predict'
# Based on https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import flask
import io
import time
from tensorflow import logging

# Initialize the Flask application and the model
app = flask.Flask(__name__)
model = None
GOOD_IMAGE_THRESHOLD = 0.95
logging.set_verbosity(logging.ERROR)

def load_model():
	# Load the pre-trained model
    global model
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    model._make_predict_function()	# initialize before threading

def prepare_image(image, target):
    image = np.array(image[:,:,:3]).reshape(1, target[0], target[1], 3)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image
            image = flask.request.files["image"].read()
            image = plt.imread(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(600, 780))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            start_time = time.time()
            preds = model.predict(image)
            data["time"] = time.time() - start_time
            data["predictions"] = np.array2string(preds)
            good_value = preds.item(0)
            data["category"] = "BAD"
            if (good_value >= GOOD_IMAGE_THRESHOLD):
                data["category"] = "GOOD"

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0',port=5000,debug=False)
