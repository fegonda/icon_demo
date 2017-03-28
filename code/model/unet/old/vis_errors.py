import mahotas
import matplotlib.pyplot as plt
import numpy as np
import fast64counter
from evaluation import thin_boundaries

def relabel(image):
    id_list = np.unique(image)
    for index, id in enumerate(id_list):
        image[image==id] = index
    return image


labelPath = '/media/vkaynig/Data1/all_data/testing/left_cylinder_test/daniel_dongil_vast_mojo_merge_export_cc/z=00000183.png'

boundaryPath = '/media/vkaynig/Data1/all_data/testing/left_cylinder_test/boundaryProbabilities/unet_sampling_best_fineTuned_best/0183.tif'

labelImg = mahotas.imread(labelPath)
tmp = np.zeros((labelImg.shape[0], labelImg.shape[1]))
tmp = labelImg[:,:,0] + 255*labelImg[:,:,1] + 255**2 * labelImg[:,:,2]
labelImg = relabel(tmp).astype(np.int32)

boundaries = labelImg == -1
boundaries[0:-1,:] = np.logical_or(boundaries[0:-1,:], np.diff(labelImg, axis=0)!=0)
boundaries[:,0:-1] = np.logical_or(boundaries[:,0:-1], np.diff(labelImg, axis=1)!=0)

membranes = np.logical_or(boundaries, labelImg[:,:]==0) 
labelImg, _ = mahotas.label(membranes==0)

boundaryImg = mahotas.imread(boundaryPath)

mask = (membranes == 0)

error_pixel_count = np.inf

for thresh in xrange(0,255,50):
    print "#################################"
    print thresh
    pred, _ = mahotas.label(boundaryImg>thresh)
    pred = thin_boundaries(pred, mask)
    # transfer background region from gt
    pred[labelImg==0] = 0
    pred = pred.astype(np.int32)
    
    errors_r = np.zeros(pred.shape)
    errors_g = np.zeros(pred.shape)
    errors_b = np.zeros(pred.shape)
    
    counter_pairwise = fast64counter.ValueCountInt64()
    counter_pairwise.add_values_pair32(labelImg[mask], pred[mask])
    
    label_id, pred_id, count = counter_pairwise.get_counts_pair32()
    count_copy = count.copy()
    
    for i in xrange(count_copy.shape[0]):
        if i%100==0:
            print count_copy.shape[0]-i
            
        ind = np.argmax(count_copy)
        
        pixels_label = labelImg==label_id[ind]
        pixels_pred = pred==pred_id[ind]
        
        #split error
        #same groundtruth region overlapped by different prediction regions
        if np.sum(errors_g[pixels_label]) > 0:
            errors_r = np.logical_or(errors_r, np.logical_and(pixels_label, pixels_pred))
    
        else:
            errors_g = np.logical_or(errors_g, pixels_label)
            #merge error

        count_copy[ind] = 0


    print np.sum(errors_r)
    if np.sum(errors_r) < error_pixel_count:
        print "UPDATE UPDATE UPDATE"
        color_rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        #r
        color_rgb[:,:,0] = errors_r
        #g
        color_rgb[:,:,1] = boundaryImg
        #b
        color_rgb[:,:,2] = errors_g

        mahotas.imsave('error_vis.png', np.uint8(color_rgb))
        error_pixel_count = np.sum(errors_r)
    
