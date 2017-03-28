#---------------------------------------------------------------------------
# datasets.py
#
# Author  : Felix Gonda
# Date    : July 12, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains the implementation of a dataset class
#           that houses all data used by the learning model.  The data
#           is loaded through the load_* methods.
#---------------------------------------------------------------------------


import os
import sys
import math
import theano
import theano.tensor as T
import numpy as np
import mahotas
import scipy.ndimage
import scipy.misc
import skimage.transform
import glob
import random
import time
import sys
import shutil
import json
import PIL.Image

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from utility import Utility
from settings import Paths
from project import Project
from db import DB

class Entry:
    def __init__(self, name, offset, length):
        self.name   = name
        self.offset = offset
        self.length = length


class Data:

    Split                  = [0.8, 0.10, 0.05]
    MinSamples             = 1024  
    MaxSamples             = 1024*1024 # about 13 GB of ram needed
    TrainSuperBatchSize    = 1024*4
    ValidSuperBatchSize    = 1024*1    

    #-------------------------------------------------------------------
    # Main constructor of the datasets. Initializes the associated
    # image id and settings, and invalidates the datasets until
    # later loaded via the load_* methods
    #-------------------------------------------------------------------
    def __init__(self, project):
        self.project      = project
        self.reset()

    def reset(self):
        self.entries      = []
        self.x            = []
        self.y            = []
        self.p            = []
        self.i            = []
        self.i_train      = []

        self.entries_valid=[]
        self.x_valid      = []
        self.y_valid      = []
        self.i_valid      = []

        self.i_randomize  = 0
        self.n_superbatch = 0
        self.data_changed = 0
        self.avg_losses   = []
        self.last_avg_loss= 0
        self.best_validation_loss = np.inf

    def add_validation_loss(self, loss):
        self.avg_losses.append( loss )

    def save_stats(self, project):
        Utility.report_status('.', '.')
        for entry in self.entries:
            i = np.arange( entry.offset, entry.offset+entry.length )
            #y = self.y[ i ]
            p = self.p[ i ]
            n_good  = len( np.where( p == 0 )[0] )

            print 'data.save_stats'
            print np.bincount( self.p ), np.bincount( p ), n_good
            print len(p), '/', len(self.p)

            score = 0.0 if n_good == 0 else float(n_good)/len(p)
            DB.storeTrainingScore( project.id, entry.name, score )
            Utility.report_status('%s'%(entry.name), '%.2f'%(score))
            #print 'image (%s)(%.2f)'%(entry.name, score)
        Utility.report_status('.', '.')

    def valid(self): 
        print 'data.valid...'
        print '#samples:', len(self.y)
        print 'Data.ValidSuperBatchSize:', Data.ValidSuperBatchSize
        print 'Data.TrainSuperBatchSize:', Data.TrainSuperBatchSize
        return len(self.y) > (Data.TrainSuperBatchSize + Data.ValidSuperBatchSize) 

    def get_pixel_count(self, project):

        counts = [ 0 for i in project.labels]
        images = DB.getTrainingImages( project.id, new=False)

        # Load training samples for each image.
        for image in images:

            annPath = '%s/%s.%s.json'%(Paths.Labels, image.id, project.id)
            with open(annPath) as json_file:
                annotations = json.load( json_file )

            if len(annotations) == 0:
                continue

            for i, coordinates in enumerate(annotations):
                counts[i] += len(coordinates)/2
            
        return counts

    def load(self, project):

        self.projecft = project

        train_new = len(self.entries) > 0
        valid_new = len(self.entries_valid) > 0
        train_images = DB.getImages( project.id, purpose=0, new=train_new)
        valid_images = DB.getImages( project.id, purpose=1, new=valid_new )
        self.load_data(Paths.TrainGrayscale, train_images, project, self.x, self.y, self.entries)
        self.load_data(Paths.ValidGrayscale, valid_images, project, self.x_valid, self.y_valid, self.entries_valid)
        exit(1)


    def load_data(self, path, images, project, x_out, y_out, entries_out):
        
        print 'data.load_data'
        print 'path:', path
        print 'images:'
        print [img.id for img in images]
              
        if len(images) == 0:
            return

        # determine the maximum number of samples to draw
        # from each image
        n_samples_per_image = Data.MaxSamples/len(images)


        entries = []

        for image in images:
            Utility.report_status( 'loading', image.id)
            print 'ttime:', image.trainingTime
            print 'atime:', image.annotationTime
            print 'tstat:', image.trainingStatus

            offset = len( entries )


    def oldload(self, project):

        self.projecft = project

        # retrieve the list of training images 
        # (annotated images)
        first_time = (len(self.entries) == 0)
        images     = DB.getTrainingImages( project.id, new=(not first_time) )
        imgs = DB.getImages( project.id )

        '''
        print 'data.load'
        print 'imgs:'
        for img in imgs:
            print img.id, img.annotationFile, img.annotationTime, img.annotationStatus
        print 'images:'
        print images
        exit(1)
        '''

        # bailout if there's no images to train.
        if len(images) == 0:
            return

        # determine the maximum number of samples to draw
        # from each image
        n_samples_per_image = Data.MaxSamples/len(images)

        print '#n_samples_per_image:', n_samples_per_image
        print '#images:', len(images)

        entries = []

        # Load training samples for each image.
        for image in images:

            Utility.report_status( 'loading', image.id)
            print 'ttime:', image.trainingTime
            print 'atime:', image.annotationTime
            print 'tstat:', image.trainingStatus

            offset = len( entries )

            # generate samples for the image
            data   = self.gen_samples( project, image.id, n_samples_per_image )
            x_data = data[0]
            y_data = data[1]
            n_data = len( y_data )

            # skip if no annotations found
            if n_data == 0:
                continue

            # add sample to the training set
            if offset == 0:
                x = x_data
                y = y_data
                p = np.ones( n_data, dtype=np.int )
            else:
                x = np.vstack( (x, x_data) )
                y = np.hstack( (y, y_data) )
                p = np.hstack( (p, np.ones( n_data, dtype=np.int )) ) 

            # keep track of each image's data in an entry for 
            # easier replacement.
            entries.append( Entry( image.id, offset, n_data ) )

            #Utility.report_memused()
            Utility.report_status('x', '(%d bytes)'%(x.nbytes))
            Utility.report_status('y', '(%d bytes)'%(y.nbytes))
            Utility.report_status('.','.')


        # bailout if no entries found
        if len(entries) == 0:
            Utility.report_status('Fetching new data', 'None Found')
            return

        Utility.report_status( 'Loading new data', 'please wait')

        # bailout if no current entries
        if len(self.entries) > 0:
            #append old entries after the new entries
            offset = len(y)

            print entries[-1].name, entries[-1].offset, entries[-1].length
            mask = np.ones( len(self.y), dtype=bool)
            names = [ e.name for e in entries ]

            for entry in self.entries:
                if entry.name in names:
                    mask[ entry.offset : entry.offset+entry.length ] = False
                else:
                    entry.offset = offset
                    offset += entry.length
                    entries.append( entry )
                    print entry.name, entry.offset, entry.length

            x_keep = self.x[ mask ]
            y_keep = self.y[ mask ]
            p_keep = self.p[ mask ]
            x = np.vstack( (x, x_keep) )
            y = np.hstack( (y, y_keep) )
            p = np.hstack( (p, p_keep) )

        
        if len( np.unique( y ) ) <= 1:
            print 'not enough labels specified...'
            return

        n_data = len(y)
        DB.finishLoadingTrainingset( project.id )

        # save the data
        self.x = x
        self.y = y
        self.p = p
        self.i = np.arange( n_data )
        self.i = np.random.choice(self.i, n_data, replace=False)
        self.entries = entries
        self.n_superbatch = int(n_data/(Data.TrainSuperBatchSize + Data.ValidSuperBatchSize))
        self.i_randomize  = 0
        self.data_changed = True
        self.i_train = []
        self.avg_losses = []
        self.last_avg_loss = 0

        Utility.report_status('.','.')
        Utility.report_status('loading complete','.')
        Utility.report_status('#samples','(%d)'%(n_data))
        Utility.report_status('x shape','(%d,%d)'%(self.x.shape[0], self.x.shape[1]))
        Utility.report_status('y shape','(%d)'%(self.x.shape[0]))
        Utility.report_status('x memory', '(%d bytes)'%(self.x.nbytes))
        Utility.report_status('y memory', '(%d bytes)'%(self.y.nbytes))
        Utility.report_memused()

    def randomize(self):
        print 'data.randomize...'
        self.i_randomize = 0
        self.i = np.concatenate( (self.i, self.i_train) )
        self.i = np.random.choice(self.i, len(self.i), replace=False)
        self.i_train = self.i[:Data.TrainSuperBatchSize]
        #self.i_valid = self.i[Data.TrainSuperBatchSize:Data.TrainSuperBatchSize+Data.ValidSuperBatchSize]
        self.i = self.i[Data.TrainSuperBatchSize:]

    def sample(self):

        avg_loss = 0
        if len(self.avg_losses)>0:
            avg_loss = np.mean(self.avg_losses)

        diff = abs(avg_loss-self.last_avg_loss)
        self.last_avg_loss = avg_loss
        print '----lsss:', len(self.avg_losses)
        print ':::::dif:', diff
        print ':::::avg:', self.last_avg_loss
        if diff < 0.00005 and len(self.avg_losses)>0:
            print '--->--->>>----'
            print 'loading.....'
            print 'last_avg_loss:',self.last_avg_loss 
            self.reset()
            self.load( self.project )
            
        if len(self.avg_losses) > 0:
            avg = np.mean(self.avg_losses)

        n_data = len(self.y)

        resetValidationError = False

        if n_data < (Data.TrainSuperBatchSize + Data.ValidSuperBatchSize):
            print 'Data.sample - not enough data...'
            return

        resetValidationError = False

        if len(self.i_train) > 0:
            print 'stratified sampling....', self.i_randomize, self.n_superbatch
            self.i_randomize += 1

            #self.i = np.concatenate( (self.i, self.i_valid) )

            #self.i = np.concatenate( (self.i, self.i_train) )
            #self.i = np.concatenate( (self.i, self.i_valid) )

            #if self.data_changed and self.i_randomize >0 and (self.i_randomize%2 == 0):
            if self.data_changed and len(self.i_train) > 0:
                resetValidationError = True
                self.data_changed = False

            '''
            if self.i_randomize == self.n_superbatch:
                self.i_randomize = 0
                self.i = np.concatenate( (self.i, self.i_train) )
                self.i = np.random.choice(self.i, len(self.i), replace=False)
                self.i_train = self.i[:Data.ValidSuperBatchSize]
                self.i = self.i[Data.ValidSuperBatchSize:]
            else:
                self.i = np.concatenate( (self.i, self.i_train) )
                self.i_train = self.i[:Data.ValidSuperBatchSize]
                self.i = self.i[Data.ValidSuperBatchSize:]

            if True:
                #self.i_randomize = 0
                #self.i_train = np.random.choice(self.i, Data.TrainSuperBatchSize, replace=False)
            '''

            #if self.i_randomize == 5: #self.n_superbatch:
            #    print 'randomizing...'
            #    self.randomize()
            #else:
            
            n_train      = len(self.i_train)
            p_train      = self.p[ self.i_train ]
            #i_sorted     = np.argsort( p_train, axis = 0)
            print 'p_train', len(p_train)
            print np.unique( p_train )
            print p_train

            n_good       = len( np.where( p_train == 0 )[0] )
            #n_good       = len( np.where( p_train[ i_sorted ] == 0 )[0] )
            print 'n_train:', n_train
            print 'n_good:', n_good
            print ' n_bad:', n_train - n_good

            #if n_good == n_train:
            if True:
                #self.randomize()
                #resetValidationError = True
                #self.i = np.concatenate( (self.i, self.i_train) )
                #self.i = np.random.choice(self.i, len(self.i), replace=False)
                i_good = np.where( self.p == 0 )[0]
                i_bad  = np.where( self.p == 1 )[0]
                n_good = len(i_good)
                n_bad  = len(i_bad)
                print '===>----<===='
                print 'ngood:', n_good, 'nbad:', n_bad
                #i_good = np.random.choice( i_good, n_good, replace=False)
                if n_bad > 0:
                    i_bad  = np.random.choice( i_bad, n_bad, replace=False )
                self.i = np.hstack( (i_good, i_bad)  )
                
                n_good = min( len(i_good), n_train/2)
                n_bad  = min( len(i_bad), n_train/2)

                if n_good < (n_train/2):
                    n_bad = n_train - n_good
                elif n_bad < (n_train/2):
                    n_good = n_train - n_bad

                #i_good = i_good[:n_good]
                #i_bad  = i_bad[:n_bad]
                self.i_train = np.hstack( (i_good[:n_good], i_bad[:n_bad])  )
                self.i_train = np.random.choice(self.i_train, n_train, replace=False)
                self.i = np.concatenate( (self.i, i_good[n_good:]) )
                self.i = np.concatenate( (self.i, i_bad[n_bad:]) )

                print '===>***<===='
                print 'ngood:', n_good, 'nbad:', n_bad
                print 'igood:', i_good[:5]
                print 'ibad:', i_bad[:5]

                '''
                print 'ngood:', n_good, 'nbad:', n_bad
                print 'igood'
                print i_good
                print 'ibad'
                print i_bad
                '''
            else:
                i_sorted     = np.argsort( p_train, axis = 0)
                n_good       = max( n_good, n_train/2)
                n_bad        = n_train - n_good

                i_good       = i_sorted[ : n_good ]
                i_bad        = i_sorted[ n_good: ]

                self.i       = np.concatenate( (self.i, i_good ))
                i_new        = self.i[:n_good]
                self.i       = self.i[n_good:]
                self.i_train = np.hstack( (i_new, i_bad) ) 

                print 'i_good:', i_good[:5]
                print ' i_bad:', i_bad[:5]
                print 'bad:', len(i_bad)
                print 'good:', len(i_good)


        else:
            print 'initial sampling...'
            #self.i_train = self.i[:Data.TrainSuperBatchSize]
            #self.i_valid = self.i[Data.TrainSuperBatchSize:Data.TrainSuperBatchSize+Data.ValidSuperBatchSize]   
            #self.i = self.i[Data.TrainSuperBatchSize+Data.ValidSuperBatchSize:]
            self.i_valid = self.i[:Data.ValidSuperBatchSize]
            self.i       = self.i[Data.ValidSuperBatchSize:]
            self.i_train = np.random.choice(self.i, Data.TrainSuperBatchSize, replace=False)
            

        #self.i = self.i[Data.TrainSuperBatchSize+Data.ValidSuperBatchSize:]
        #indices  = np.random.choice( self.i, len(self.i), replace=False)
        #indices = indices.astype(dtype=int)
    
        #self.i_train = indices[:Data.TrainSuperBatchSize]
        #self.i_valid = indices[Data.TrainSuperBatchSize:Data.TrainSuperBatchSize+Data.ValidSuperBatchSize]

        self.avg_losses = []
        x_train = self.x[ self.i_train ]
        y_train = self.y[ self.i_train ]
        x_valid = self.x[ self.i_valid ]
        y_valid = self.y[ self.i_valid ]

        print 'n_superbatch:', self.n_superbatch
        print 'i_randomize:', self.i_randomize
        '''
        print 'x_train:', x_train
        print 'y_train:', y_train
        print 'x_valid:', x_valid
        print 'y_valid:', y_valid
        '''
        print 'i:', self.i.shape
        print 'it:', self.i_train.shape
        print 'iv:', self.i_valid.shape
        print 'xt:', x_train.shape
        print 'yt:', y_train.shape
        print 'xv:', x_valid.shape
        print 'yv:', y_valid.shape

        return x_train, y_train, x_valid, y_valid, resetValidationError
        
    def gen_samples(self, project, imageId, nsamples):

        data_mean=project.mean
        data_std=project.std

        annPath = '%s/%s.%s.json'%(Paths.Labels, imageId, project.id)
        imgPath = '%s/%s.tif'%(Paths.TrainGrayscale, imageId)

        if not os.path.exists(annPath) or not os.path.exists( imgPath):
            return [], []

        with open(annPath) as json_file:
            annotations = json.load( json_file )

        if len(annotations) == 0:
            return [], []

        n_labels  = len(annotations)

        # compute the sample sizes for each label in the annotations
        n_samples_size  = nsamples/n_labels
        samples_sizes = []
        for coordinates in annotations:
            n_label_samples_size = len(coordinates)/2
            n_label_samples_size = min( n_label_samples_size, n_samples_size )
            samples_sizes.append( n_label_samples_size )

        # bailout if not enough samples in the annotations
        n_total = np.sum(samples_sizes)
        if n_total < Data.MinSamples:
            print 'Data.gen_samples'
            print 'Not enough samples in image: %s'%(imageId)
            return [], []

        # recompute the label sample sizes to ensure the min required samples
        # is fullfiled
        n_diff = nsamples - n_total
        i = 0
        while n_diff > 0 and i < n_labels:
            n_label_samples_size = len(annotations[i])/2
            n_add_samples_size   = n_label_samples_size - samples_sizes[i]
            n_add_samples_size   = min( n_add_samples_size, n_diff )
            n_add_samples_size   = max( n_add_samples_size, 0)
            samples_sizes[i]  += n_add_samples_size
            n_diff              -= n_add_samples_size
            i                   += 1

        '''
        print '---Data.gen_samples---'
        print 'nsamples:', nsamples
        print 'nsamples actual:', np.sum( samples_sizes )
        print 'n_samples_size:', n_samples_size
        print 'sample sizes:', samples_sizes
        print 'len samples:', len(samples_sizes)
        print '#samples: ', np.sum(samples_sizes)
        print '#actual:', np.sum( [ len(c)/2 for c in annotations ] )
        '''

        mode   = 'symmetric'
        patchSize = project.patchSize
        pad = patchSize
        img = mahotas.imread( imgPath )
        img = np.pad(img, ((pad, pad), (pad, pad)), mode)
        img = Utility.normalizeImage(img)

        whole_set_patches = np.zeros((n_total, patchSize*patchSize), dtype=np.float)
        whole_set_labels = np.zeros(n_total, dtype=np.int32)

        border_patch = np.ceil(patchSize/2.0)

        counter = 0
        for label, coordinates in enumerate( annotations ):

            if counter >= n_total:
                break

            ncoordinates = len(coordinates)
        
            if ncoordinates == 0:
                continue

            # randomly sample from the label
            indices = np.random.choice( ncoordinates, samples_sizes[label], replace=False)
 
            for i in indices:
                if i%2 == 1:
                    i = i-1

                if counter >= n_total:
                    break

                col = coordinates[i]
                row = coordinates[i+1]
                r1  = row+patchSize-border_patch
                r2  = row+patchSize+border_patch+1
                c1  = col+patchSize-border_patch
                c2  = col+patchSize+border_patch+1

                imgPatch = img[r1:r2,c1:c2]
                imgPatch = skimage.transform.rotate(imgPatch, random.choice(xrange(360)))
                imgPatch = imgPatch[0:patchSize,0:patchSize]

                if random.random() < 0.5:
                    imgPatch = np.fliplr(imgPatch)

                whole_set_patches[counter,:] = imgPatch.flatten()
                whole_set_labels[counter] = label
                counter += 1

        whole_data = np.float32(whole_set_patches)
        if data_mean == None:
            whole_data_mean = np.mean(whole_data,axis=0)
        else:
            whole_data_mean = data_mean

        whole_data = whole_data - np.tile(whole_data_mean,(np.shape(whole_data)[0],1))
        if data_std == None:
            whole_data_std = np.std(whole_data,axis=0)
        else:
            whole_data_std = data_std

        whole_data_std = np.clip(whole_data_std, 0.00001, np.max(whole_data_std))
        whole_data = whole_data / np.tile(whole_data_std,(np.shape(whole_data)[0],1))

        return whole_data, whole_set_labels

    def pad_image(self, img, patchSize):
        mode   = 'symmetric'
        pad = patchSize
        img = np.pad(img, ((pad, pad), (pad, pad)), mode)
        return img

    def gen_samples_offline(self, nsamples, purpose, patchSize, mean=None, std=None):

        data_mean=mean
        data_std=std

        pathPrefix = '%s/'%Paths.Reference
        img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'
        img_search_string_membraneImages = pathPrefix + 'labels/membranes/' + purpose + '/*.tif'
        img_search_string_backgroundMaskImages = pathPrefix + 'labels/background/' + purpose + '/*.tif'

        img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )
        img_files_label = sorted( glob.glob( img_search_string_membraneImages ) )
        img_files_backgroundMask = sorted( glob.glob( img_search_string_backgroundMaskImages ) )

        whole_set_patches = np.zeros((nsamples, patchSize*patchSize), dtype=np.float)
        whole_set_labels = np.zeros(nsamples, dtype=np.int32)

        print 'src:', img_search_string_grayImages
        print 'mem:', img_search_string_membraneImages
        print 'label:', img_search_string_membraneImages
        print 'msk:', img_search_string_backgroundMaskImages

    #how many samples per image?
        nsamples_perImage = np.uint(np.ceil(
                (nsamples) / np.float(np.shape(img_files_gray)[0])
                ))
        print 'using ' + np.str(nsamples_perImage) + ' samples per image.'
        counter = 0
    #    bar = progressbar.ProgressBar(maxval=nsamples, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


        img = mahotas.imread(img_files_gray[0])
        img = self.pad_image(img, patchSize)
        grayImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
        labelImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
        maskImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))

        for img_index in xrange(np.shape(img_files_gray)[0]):
            img = mahotas.imread(img_files_gray[img_index])
            img = self.pad_image(img, patchSize)
            img = Utility.normalizeImage(img)
            grayImages[:,:,img_index] = img

            label_img = mahotas.imread(img_files_label[img_index])
            label_img = self.pad_image(label_img, patchSize)
            labelImages[:,:,img_index] = label_img

            mask_img = mahotas.imread(img_files_backgroundMask[img_index])
            mask_img = self.pad_image(mask_img, patchSize)
            maskImages[:,:,img_index] = mask_img

        for img_index in xrange(np.shape(img_files_gray)[0]):
            img = grayImages[:,:,img_index]
            label_img = labelImages[:,:,img_index]
            mask_img = maskImages[:,:,img_index]

            #get rid of invalid image borders
            border_patch = np.ceil(patchSize/2.0)
            #border = np.ceil(patchSize/2.0)
            border = np.ceil(np.sqrt(2*(border_patch**2)))
            label_img[:border,:] = 0 #top
            label_img[-border:,:] = 0 #bottom
            label_img[:,:border] = 0 #left
            label_img[:,-border:] = 0 #right

            mask_img[:border,:] = 0
            mask_img[-border:,:] = 0
            mask_img[:,:border] = 0
            mask_img[:,-border:] = 0

            membrane_indices = np.nonzero(label_img)
            non_membrane_indices = np.nonzero(mask_img)

            n_mem_indices = len(membrane_indices[0])
            n_non_mem_indices = len(non_membrane_indices[0])

            rand_mem_indices = np.random.choice(n_mem_indices,n_mem_indices,replace=False)
            rand_non_mem_indices = np.random.choice(n_non_mem_indices, n_non_mem_indices,replace=False)

            i_mem=0
            i_non_mem=0

            positiveSample = True
            for i in xrange(nsamples_perImage):
                if counter >= nsamples:
                    break
    #            positiveSample = rnd.random() < balanceRate

                if i_mem >= n_mem_indices and i_non_men  >= n_non_mem_indices:
                    break
                elif not positiveSample and i_non_mem >= n_non_mem_indices:
                    positiveSample = True
                elif positiveSample and i_mem >= n_mem_indices:
                    positiveSample = False
                    

                if positiveSample:
                    randmem = rand_mem_indices[ i_mem ]
                    i_mem += 1
                    #randmem = random.choice(xrange(len(membrane_indices[0])))
                    (row,col) = (membrane_indices[0][randmem],
                                 membrane_indices[1][randmem])
                    label = 1.0
                    positiveSample = False
                else:
                    randmem = rand_mem_indices[ i_non_mem ]
                    i_non_mem += 1
                    #randmem = random.choice(xrange(len(non_membrane_indices[0])))
                    (row,col) = (non_membrane_indices[0][randmem],
                                 non_membrane_indices[1][randmem])
                    label = 0.0
                    positiveSample = True

                imgPatch = img[row-border+1:row+border, col-border+1:col+border]
    #            print "patch: ", np.max(imgPatch)
                #imgPatch = scipy.misc.imrotate(imgPatch,random.choice(xrange(360)))
                imgPatch = skimage.transform.rotate(imgPatch, random.choice(xrange(360)))
    #            print "after rotation: ", np.max(imgPatch)
                imgPatch = imgPatch[border-border_patch+1:border+border_patch,border-border_patch+1:border+border_patch]
    #            print "small patch: ", np.max(imgPatch)
    #            middle = np.floor(patchSize/2.0)
    #            blindSpotSize = 6
    #            imgPatch[middle-blindSpotSize:middle+blindSpotSize,middle-blindSpotSize:middle+blindSpotSize] = 0.0

                if random.random() < 0.5:
                        imgPatch = np.fliplr(imgPatch)
    #                    print "after flip: ", np.max(imgPatch)

                #Force network to learn shapes instead of gray values by inverting randomly
                if True: #random.random() < 0.5:
                        whole_set_patches[counter,:] = imgPatch.flatten()
                else:
                        whole_set_patches[counter,:] = 1-imgPatch.flatten()

                whole_set_labels[counter] = label
                counter += 1
    #            bar.update(counter)



        #    bar.finish()     
        #normalize data
        whole_data = np.float32(whole_set_patches)
        #    print "whole_data: ", np.max(whole_data)    
        if data_mean == None:
            whole_data_mean = np.mean(whole_data,axis=0)
        else:
            whole_data_mean = data_mean

        whole_data = whole_data - np.tile(whole_data_mean,(np.shape(whole_data)[0],1))
        #    print "whole_data normalized mean: ", np.max(whole_data)        

        #    whole_data_mean = np.mean(whole_data, axis=1)
        #    whole_data = whole_data - np.transpose(np.tile(whole_data_mean, (np.shape(whole_data)[1],1)))

        if data_std == None:
            whole_data_std = np.std(whole_data,axis=0)
        else:
            whole_data_std = data_std

        whole_data_std = np.clip(whole_data_std, 0.00001, np.max(whole_data_std))
        whole_data = whole_data / np.tile(whole_data_std,(np.shape(whole_data)[0],1))

        #data = np.uint8(whole_data*255).copy()
        #labels = np.uint8(whole_set_labels).copy()

        data = whole_data.copy()
        labels = whole_set_labels.copy()

        #remove the sorting in image order
        shuffleIndex = np.random.permutation(np.shape(labels)[0])
        for i in xrange(np.shape(labels)[0]):
            whole_data[i,:] = data[shuffleIndex[i],:]
            whole_set_labels[i] = labels[shuffleIndex[i]]

        return whole_data, whole_set_labels, whole_data_mean, whole_data_std
