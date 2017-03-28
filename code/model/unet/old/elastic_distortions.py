import mahotas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage

def deform_image(image):
    displacement_x = np.random.normal(size=image.shape, scale=10)
    displacement_y = np.random.normal(size=image.shape, scale=10)
    
    # smooth over image
    coords_x, coords_y = np.meshgrid(np.arange(0,image.shape[0]), np.arange(0,image.shape[1]), indexing='ij')

    displacement_x = coords_x.flatten() + scipy.ndimage.gaussian_filter(displacement_x, sigma=5).flatten()
    displacement_y = coords_y.flatten() + scipy.ndimage.gaussian_filter(displacement_y, sigma=5).flatten()
    
    coordinates = np.vstack([displacement_x, displacement_y])
    
    deformed = scipy.ndimage.map_coordinates(image, coordinates, mode='reflect')
    return np.reshape(deformed, image.shape)
    

#image = mahotas.imread('ac3_input_0141.tif')
image = mahotas.imread('ac3_labels_0000.tif')
test = deform_image(image)

plt.imshow(image); plt.figure()
plt.imshow(test); plt.show()
