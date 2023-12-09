import stltovoxel
import numpy as np
from PIL import Image
import sys
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filename = sys.argv[1]
    path = "C:/Users/slin1/Documents/Fall 2023/AA 228/final_project/Training_Samples"
    output_folder = path + filename + '/output'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    stltovoxel.convert_file(path + filename + filename + '.stl', output_folder + '/output.png', resolution = 50) 

    count = 0
    for path in os.listdir(output_folder):
        if os.path.isfile(os.path.join(output_folder, path)):
            count += 1

    shaper = Image.open(output_folder + '/output_' + str(count - 1) +  '.png')
    shaper_np = np.asarray(shaper)
    y, x = shaper_np.shape
    voxel = np.zeros((x, y, count)) 

    for i in np.arange(count):
        img = Image.open(output_folder + '/output_' + str(i).zfill(len(str(count))) + '.png')
        voxel[:,:,i] = np.asarray(img).T

    np.save(filename[1:] + ".npy", voxel)