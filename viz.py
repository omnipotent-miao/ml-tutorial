import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json, Model
from keras.preprocessing import image
from keras import activations
from keras.applications import VGG16
from vis.utils import utils
from vis.visualization import visualize_activation
from tensorflow import logging

# Disable TensorFlow warnings
logging.set_verbosity(logging.ERROR)

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

# Load test image
img = image.load_img(image_file)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
plt.imshow(img) #img_tensor[0]
plt.axis('off')
plt.show()

# Visualize class outputs. Shows a visual representation of the final softmax 
# layer. For a given output category ("good image" or "bad image") the CNN 
# will generate an image that maximally represents the category.
print()
answer = input('Enter "y" to visualize the class outputs: ');
if answer == 'y':
    layer_idx = utils.find_layer_idx(loaded_model, 'dense_2')
    loaded_model.layers[layer_idx].activation = activations.linear
    loaded_model = utils.apply_modifications(loaded_model)
    img = visualize_activation(loaded_model, layer_idx, filter_indices=1) # 0 = good, 1 = bad
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

# Generate intermediate activations for the model layers
layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']
layer_outputs = [layer.output for layer in loaded_model.layers if layer.name in layer_names]
activation_model = Model(inputs=loaded_model.input, outputs=layer_outputs)
intermediate_activations = activation_model.predict(img_tensor)
images_per_row = 8
max_images = 8

# Display the feature maps
for layer_name, layer_activation in zip(layer_names, intermediate_activations):
    n_features = layer_activation.shape[-1]
    n_features = min(n_features, max_images)
    xsize = layer_activation.shape[1]
    ysize = layer_activation.shape[2]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((xsize * n_cols, images_per_row * ysize))

    # Tile each filter into a horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * xsize : (col + 1) * xsize,
                         row * ysize : (row + 1) * ysize] = channel_image

    xscale = 2. / xsize
    yscale = 2. / ysize
    plt.figure(figsize=(yscale * display_grid.shape[1],
                        xscale * display_grid.shape[0]))
    plt.axis('off')
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()
