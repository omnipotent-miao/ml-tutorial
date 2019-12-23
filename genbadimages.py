import os
import matplotlib.pyplot as plt

HOME_DIR = 'C:\\Users\\JoeMeli\\Documents\\git\\imageqc\\'
INPUT_DIR = HOME_DIR + 'train_images\\'
OUTPUT_DIR = HOME_DIR + 'bad_images\\'
row1 = 55
row2 = 562

# These image indices represent unique image headers as determined by visual inspection of the full set of training images
indices = [1,2,14,41,68,79,90,107,139,167,194,225,253,280,293,306,319,332,359,386,397,408,422,434,461,474,483,510,537,548,559,563,567,580,593,621,648,661]

# Black out the content area of an Energy image
# Note: pyplot.imsave converts the image from RGB to RGBA. Ignore the alpha channel using image[:,:,:3].
os.chdir(INPUT_DIR)
im_files = os.listdir()
count = 0
for f in im_files:
    fnum = int(f[0:3])
    if fnum in indices:
        count = count + 1
        output_name = OUTPUT_DIR + str(len(im_files) + count) + '.png'
        image = plt.imread(f)
        image[row1:row2, :, :] = 0
        plt.imsave(output_name, image)

# Create an image with only the white header line
image = plt.imread(im_files[1])
image[0:50, :, :] = 0
image[55:600, :, :] = 0
count = count + 1
output_name = OUTPUT_DIR + str(len(im_files) + count) + '.png'
plt.imsave(output_name, image)

# Create an all black image
image = plt.imread(im_files[1])
image[0:600, :, :] = 0
count = count + 1
output_name = OUTPUT_DIR + str(len(im_files) + count) + '.png'
plt.imsave(output_name, image)
