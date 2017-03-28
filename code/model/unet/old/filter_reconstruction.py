import glob
import os
import mahotas
import numpy as np
import matplotlib.pyplot as plt
import fast64counter
import skimage.color
import time

stackSize = 300
nr_largest_objects = 500
nr_reruns = 1

file_search_string = '/media/vkaynig/Data1/Cmor_paper_data/left_cylinder_test/pipeline_output/output_ids_small/*.tif'
files = sorted( glob.glob( file_search_string ) )
files = files[1:stackSize]

tmp = mahotas.imread(files[0])

stack = np.zeros((tmp.shape[0], tmp.shape[1], len(files)))
newStack = np.zeros((tmp.shape[0], tmp.shape[1], len(files)))

print "reading images"
for i,f in enumerate(files):
    print len(files)-i
    img = mahotas.imread(f)
    stack[:,:,i] = img


print "counting"
counter = fast64counter.ValueCountInt64()
counter.add_values_32(np.int32(stack.flatten()))

ids, counts = counter.get_counts()

newIDs = 1

print "filtering"
start_time = time.clock()
stack = stack.flatten()

for j in range(nr_reruns):
    newStack = newStack.flatten()
    newStack[:] = 0
    for i in range(nr_largest_objects):
        max_loc = np.argmax(counts)
        maxID = ids[max_loc]
        counts[max_loc] = 0
        newStack[stack==maxID] = newIDs
        newIDs +=1
        
    newStack = np.reshape(newStack,(tmp.shape[0], tmp.shape[1], len(files)))
    
    print "so far this took in seconds: ", time.clock() - start_time

    print "saving"
    file_output_string = '/media/vkaynig/Data1/Cmor_paper_data/left_cylinder_test/pipeline_output/output_color/'
    for i,f in enumerate(files):
        label_image = np.int16(newStack[:,:,i])
        #color_image = skimage.color.label2rgb(label_image, bg_label=0)
        
        path = file_output_string + os.path.basename(f)
        
        #mahotas.imsave(path[:-4]+'_'+str(j)+'_filtered.tif',np.uint8(color_image*255))
        mahotas.imsave(path[:-4]+'_'+str(j)+'_filtered.tif',np.int16(label_image))
