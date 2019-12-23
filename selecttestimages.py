import os
import random
import shutil

HOME_DIR = 'C:\\Users\\JoeMeli\\Documents\\git\\imageqc\\'
INPUT_DIR = HOME_DIR + 'train_images\\'
OUTPUT_DIR = HOME_DIR + 'test_images\\'

test_image_percent = 0.2

os.chdir(INPUT_DIR)
im_files = os.listdir()
image_count = len(im_files)

for i in range(int(image_count * test_image_percent)):
    index = random.randint(1,image_count+1)
    image_name = INPUT_DIR + ("%03d" % index) + '.png'
    try:
        shutil.move(image_name, OUTPUT_DIR)
    except:
        print('Already moved:', image_name)
