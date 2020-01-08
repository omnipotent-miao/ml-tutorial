import os
from pathlib import Path


def read_image(dataset_tag, extension=".png", onehot=False, reload=False, cache=True):
    from matplotlib import pyplot as plt
    import tensorflow.keras as keras
    import numpy as np

    # For sake of simplicity, switch dir and do things inside
    pwd = Path(".").absolute()
    image_dir = f"{dataset_tag}_images"
    label_file = f"{dataset_tag}_labels.csv"
    cache_npz = f"{dataset_tag}_cache.npz"

    if not reload and Path(cache_npz).exists():
        print(f"Reuse cached array {cache_npz}")
        cache = np.load(cache_npz)
        return cache['x'], cache['y']

    # pushd
    os.chdir(image_dir)
    im_files = sorted(os.listdir("."))
    images = [plt.imread(f)[:, :, :3] for f in im_files if f.endswith(extension)]
    print(f'Number of {dataset_tag} images:', len(images))
    x = np.array(images)
    print(f'x_{dataset_tag} shape:', x.shape)
    # popd
    os.chdir(pwd)

    # Read training labels
    y_labels = np.genfromtxt(label_file, delimiter=',')
    print(f'Number of training labels equals number of {dataset_tag} images:',
          len(y_labels) == x.shape[0])

    # Do we want to use one-hot encoding for labels
    if onehot:
        y = keras.utils.to_categorical(y_labels, num_classes=2)
        if cache:
            np.savez_compressed(cache_npz, x=x, y=y)
            print(f"Cached saved as {cache_npz}")
        return x, y
    else:
        if cache:
            np.savez_compressed(cache_npz, x=x, y=y_labels)
            print(f"Cached saved as {cache_npz}")
        return x, y_labels


def inject_config():
    import inspect
    stack = inspect.stack()
    caller_globals = stack[1][0].f_globals

    # Means we need switch to project directory
    if '_src_root' in caller_globals:
        os.chdir(caller_globals['_src_root'])

    # Set variables
    HOME_DIR = Path(".").absolute()
    caller_globals['HOME_DIR'] = HOME_DIR
    caller_globals['TRAIN_DIR'] = str(HOME_DIR / 'train_images')
    caller_globals['VALIDATION_DIR'] = str(HOME_DIR / 'validation_images')
    caller_globals['TEST_DIR'] = str(HOME_DIR / 'test_images')
    caller_globals['TRAIN_LABELS'] = str(HOME_DIR / 'train_labels.csv')
    caller_globals['VALIDATION_LABELS'] = str(HOME_DIR / 'validation_labels.csv')
    caller_globals['TEST_LABELS'] = str(HOME_DIR / 'test_labels.csv')
    caller_globals['EXTENSION'] = '.png'