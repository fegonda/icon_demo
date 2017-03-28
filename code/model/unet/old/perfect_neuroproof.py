import mahotas
import matplotlib.pyplot as plt
import numpy as np
import fast64counter
from evaluation import thin_boundaries, segmentation_metrics
import glob

def relabel(image):
    id_list = np.unique(image)
    for index, id in enumerate(id_list):
        image[image==id] = index
    return image


labelPath = '/media/vkaynig/Data1/Cmor_paper_data/ac3/daniel_vast_export/'
boundaryPath = '/media/vkaynig/Data1/all_data/testing/AC3/boundaryProbabilities/unet_sampling_best_fineTuned_best/'

label_search_string = labelPath + '*.tif'
boundary_search_string = boundaryPath + '*.tif'

label_files = sorted( glob.glob( label_search_string ) )
boundary_files = sorted( glob.glob( boundary_search_string ) )

tmp = mahotas.imread(label_files[0])
matched_volume = np.zeros((tmp.shape[0], tmp.shape[1], len(label_files)))
label_volume = np.zeros((tmp.shape[0], tmp.shape[1], len(label_files)))
image_counter = -1

# WARNING this code is super conservative. It will regard small imprecisions in the boundary as errors.
# E.g. it does only threshold and do connected components. No filtering for regions too small to be neurons, or
# similar. 

for label_f, boundary_f in zip(label_files, boundary_files):
    image_counter +=1
    print image_counter
    # if image_counter < 100:
    #     continue

    #print label_f
    labelImg = mahotas.imread(label_f)
    # convert color ids to unique ids
    # tmp = np.zeros((labelImg.shape[0], labelImg.shape[1]))
    # tmp = labelImg[:,:,0] + 255*labelImg[:,:,1] + 255**2 * labelImg[:,:,2]
    # labelImg = tmp.astype(np.int32)
    labelImg = labelImg.astype(np.int32)

    # some gt images in the left cylinder data are actually blank. skip those
    if np.max(labelImg) == 0:
        continue
    
    # recomputing boundaries to split up 3D connectivity. 
    boundaries = labelImg == -1
    boundaries[0:-1,:] = np.logical_or(boundaries[0:-1,:], np.diff(labelImg, axis=0)!=0)
    boundaries[:,0:-1] = np.logical_or(boundaries[:,0:-1], np.diff(labelImg, axis=1)!=0)
    membranes = np.logical_or(boundaries, labelImg[:,:]==0) 
    
    labelImg_cc, _ = mahotas.label(membranes==0)
    
    boundaryImg = mahotas.imread(boundary_f)
    # sanity check
    #boundaryImg = (1-membranes)*255

    mask = (membranes == 0)
    
    for thresh in [150]:
        #"#################################"
        #print thresh
        
        pred, _ = mahotas.label(boundaryImg>thresh)
        pred = thin_boundaries(pred, mask)
        # transfer background region from gt
        pred[labelImg==0] = 0
        pred = pred.astype(np.int32)
        
        matched_image = np.zeros(labelImg.shape)
        splitIDs_counter = np.max(labelImg) + 1
        
        counter_pairwise = fast64counter.ValueCountInt64()
        counter_pairwise.add_values_pair32(labelImg_cc[mask], pred[mask])
        
        label_id, pred_id, count = counter_pairwise.get_counts_pair32()
        count_copy = count.copy()
        
        # create count matrix
        # most probably there is a faster more elegant way to do this

        count_matrix = np.zeros((np.max(label_id)+1, np.max(pred_id)+1))
        for l, p, c in zip(label_id, pred_id, count_copy):
            count_matrix[l,p] = c

        while np.max(count_matrix)>0:
            #print np.nonzero(count_matrix.flatten())[0].shape
            # find largest overlap
            ind = np.argmax(count_matrix)
            label_id, pred_id = np.unravel_index(ind, count_matrix.shape)
            
            # assign largest overlap to ground truth annotation
            # first get correct 3D connected label id
            gt_label_id = np.median(labelImg[labelImg_cc==label_id])
            # draw it into matched image
            matched_image[pred==pred_id] = gt_label_id
            # plt.subplot(1,2,1)
            # plt.imshow(boundaryImg>thresh); 
            # plt.subplot(1,2,2)
            # plt.title(str(gt_label_id))
            # plt.imshow(matched_image); plt.show()
            
            #plt.imshow(matched_image); plt.show()
            
            # this one is done now so set count matrix to zero
            count_matrix[label_id, pred_id] = 0
            
            # split errors are other pred ids overlapping with this gt id
            # and they do not overlap more with another segment
            split_ids = np.nonzero(count_matrix[label_id,:])
            for sid in split_ids[0]:
                # save the count
                split_id_count = count_matrix[label_id, sid]
                # set it to zero in the matrix
                count_matrix[label_id, sid] = 0
                # test if this has a larger overlap with a different gt label
                if split_id_count > np.max(count_matrix[:,sid]):
                    # this is true split error
                    matched_image[pred==sid] = splitIDs_counter
                    # plt.subplot(1,2,1)
                    # plt.imshow(boundaryImg>thresh); 
                    # plt.subplot(1,2,2)
                    # plt.title(str(splitIDs_counter))
                    # plt.imshow(matched_image); plt.show()

                    splitIDs_counter +=1
                    # this label is done so set remaining overlaps to 0
                    count_matrix[:,sid] = 0
                    
            # merge ids is basically the same thing, just looking at the column of the count matrix
            # instead of the row
            merge_ids = np.nonzero(count_matrix[:,pred_id])
            for mid in merge_ids[0]:
                # save the count
                merge_id_count = count_matrix[mid, pred_id]
                # set it to zero in the matrix
                count_matrix[mid, pred_id] = 0
                # test if this has a larger overlap with a different gt label
                if merge_id_count > np.max(count_matrix[mid,:]):
                    # this is true merge error
                    # it doesn't need to be painted into the image, the gt label is just not going to have a match
                    count_matrix[mid,:] = 0
                    
    # test by computing VI_info
    print segmentation_metrics(labelImg, matched_image)
    matched_volume[:,:,image_counter] = matched_image
    label_volume[:,:,image_counter] = labelImg
    #image_counter += 1

print "########### ALL DONE ###################"
print segmentation_metrics(label_volume, matched_volume)
    
    
    
