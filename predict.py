import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json

if (len(sys.argv) != 2):
    print("Usage: predict image_file_name")
    sys.exit()
image_file = sys.argv[1]

# Load JSON and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Use the model to make a prediction
x = np.array(plt.imread(image_file)[:,:,:3]).reshape(1, 600, 780, 3)
start_time = time.time()
prediction = loaded_model.predict(x)
print('Prediction time: %s sec' % (time.time() - start_time))
print(prediction)
