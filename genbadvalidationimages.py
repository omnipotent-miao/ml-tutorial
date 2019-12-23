import os
import matplotlib.pyplot as plt

HOME_DIR = 'C:\\Users\\JoeMeli\\Documents\\git\\imageqc\\'
INPUT_DIR = HOME_DIR + 'validation_images\\'
OUTPUT_DIR = HOME_DIR + 'bad_validation_images\\'
row1 = 55
row2 = 562

# These image indices represent unique image headers as determined by visual inspection of the full set of training images
indices = [1001, 1014, 1025, 1053, 1080, 1107, 1118]

# Black out the content area of an Energy image
# Note: pyplot.imsave converts the image from RGB to RGBA. Ignore the alpha channel using image[:,:,:3].
os.chdir(INPUT_DIR)
im_files = os.listdir()
count = 1000 # validation image names start at 1001
for f in im_files:
    fnum = int(f[0:4])
    if fnum in indices:
        count = count + 1
        output_name = OUTPUT_DIR + str(len(im_files) + count) + '.png'
        image = plt.imread(f)
        image[row1:row2, :, :] = 0
        plt.imsave(output_name, image)
