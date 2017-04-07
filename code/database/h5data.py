import os
import sys
import numpy as np
import h5py
#import tifffile as tiff
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from skimage import color
#from skimage import io
from PIL import Image
import json


base_path = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(base_path, '../common'))
sys.path.insert(0,os.path.join(base_path, '../database'))
sys.path.insert(0,os.path.join(base_path, '../../'))

from paths import Paths
from db import DB
from config import *

class H5Data:
    def __init__(self, path=Paths.Data, name='main'):
        self.index = 0
        self.path='%s/%s'%(path,data_stack_file)
        print 'path:', self.path
        print 'isfile:', os.path.isfile(path)
        self.volume = np.array(h5py.File( path ,'r')[name],dtype=np.float32)/(2.**8)

    @staticmethod
    def getSliceCount(path, name='main'):
        volume = np.array(h5py.File( '%s/%s'%(path,data_stack_file) ,'r')[name],dtype=np.float32)/(2.**8)
        return volume.shape[0];

    @staticmethod
    def extract_to(p_h5data, name, p_image, projectId, imageIndex, purpose):
        # print 'p_h5data:', p_h5data
        # print 'name:',name
        # print 'to:', p_image
        # print 'index:', imageIndex
        path = '%s/%s.jpg'%(p_image, imageIndex)
        if os.path.isfile(path):
            return

        project = DB.getProject(projectId)
        project.addImage(imageIndex, annFile=None, segFile=None, score=0.0, purpose=DB.purpose_str_to_int(purpose))
        DB.storeProject( project )
        volume = np.array(h5py.File( p_h5data ,'r')[name],dtype=np.float32)/(2.**8)
        image = volume[int(imageIndex),:,:]
        # imsave('%s'%(path), image)

        image = color.gray2rgb( image )
        image = imresize(image, (525,525))
        #path = '%s/%s.png'%(p_image, imageIndex)
        imsave('%s'%(path), image)
 

    @staticmethod
    def alpha_composite(src, dst):
        '''
        Return the alpha composite of src and dst.

        Parameters:
        src -- PIL RGBA Image object
        dst -- PIL RGBA Image object

        The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
        '''
        # http://stackoverflow.com/a/3375291/190597
        # http://stackoverflow.com/a/9166671/190597
        src = np.asarray(src)
        dst = np.asarray(dst)
        out = np.empty(src.shape, dtype = 'float')
        alpha = np.index_exp[:, :, 3:]
        rgb = np.index_exp[:, :, :3]
        src_a = src[alpha]/255.0
        dst_a = dst[alpha]/255.0
        out[alpha] = src_a+dst_a*(1-src_a)
        old_setting = np.seterr(invalid = 'ignore')
        out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
        np.seterr(**old_setting)    
        out[alpha] *= 255
        np.clip(out,0,255)
        # astype('uint8') maps np.nan (and np.inf) to 0
        out = out.astype('uint8')
        out = Image.fromarray(out, 'RGBA')
        return out

    @staticmethod
    def generate_preview( pathH5data, name, pathLabels, pathSegmentation, pathImages, imageId, projectId ):

        # input image
        volume = np.array(h5py.File( '%s/%s'%(pathH5data,data_stack_file) ,'r')[name],dtype=np.float32)#/(2.**8)
        image = volume[int(imageId),:,:]
        background = color.gray2rgb(image)*255
        background = background.astype(dtype=np.int8)
        data = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.int8)
        data[:,:,:3] = background
        data[:,:,3] = 255
        background = Image.frombuffer('RGBA',image.shape,data,'raw','RGBA',0,1)


        # segmentations
        p_segmentations = '%s/%s.%s.json'%(pathSegmentation,imageId, projectId )
        data = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.int8)
        data[:,:,:3] = 0
        data[:,:500,:1] = 0
        data[:,:500,:2] = 0
        data[:,:,3] = 0
        if os.path.isfile(p_segmentations):
             with open(p_segmentations) as json_file:
                json_data = json.load( json_file )                
                labels = [(255,0,0,255), (0,255,0,255)]
                for label, coords in zip(labels, json_data):
                    for i in range(0, len(coords), 2):
                        col = coords[i]
                        row = coords[i+1]
                        data[row, col, 0] = label[0]
                        data[row, col, 1] = label[1]
                        data[row, col, 2] = label[2]
                        data[row, col, 3] = label[3]
        segmentation = Image.frombuffer('RGBA',image.shape,data,'raw','RGBA',0,1)


        # annotations
        data = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.int8)
        data[:,:,:3] = 0
        data[:,:500,:1] = 0
        data[:,:500,:2] = 0
        data[:,:,3] = 0
        p_annotations = '%s/%s.%s.json'%(pathLabels,imageId, projectId )
        if os.path.isfile(p_annotations):
            with open(p_annotations) as json_file:
                json_data = json.load( json_file )                
                labels = [(255,0,0,255), (0,255,0,255)]
                for label, coords in zip(labels, json_data):
                    for i in range(0, len(coords), 2):
                        col = coords[i]
                        row = coords[i+1]
                        data[row, col, 0] = label[0]
                        data[row, col, 1] = label[1]
                        data[row, col, 2] = label[2]
                        data[row, col, 3] = label[3]
        annotations = Image.frombuffer('RGBA',image.shape,data,'raw','RGBA',0,1)

        combined = H5Data.alpha_composite(segmentation, background)
        combined = H5Data.alpha_composite(annotations, combined)
        combined.thumbnail((525,525), Image.ANTIALIAS)
        path = '%s/%s.jpg'%(pathImages, imageId)
        combined.save( path )

        # path = '%s/%s.jpg'%(p_image, i)
        # imsave('%s'%(path), image)

        #dsafd
        # volume = np.array(h5py.File( p_h5data ,'r')[name],dtype=np.float32)/(2.**8)
        # image = volume[int(i),:,:]
        # background = color.gray2rgb( image )

        # np.array
        # foreground = Image.open("test2.png")

        # background.paste(foreground, (0, 0), foreground)
        # background.show()
        #pass

    @staticmethod
    def extract_all(p_h5data, name, p_image):
        # print 'p_h5data:', p_h5data
        # print 'name:',name
        # print 'to:', p_image
        # print 'index:', imageIndex

        volume = np.array(h5py.File( '%s/%s'%(p_h5data,data_stack_file) ,'r')[name],dtype=np.float32)#/(2.**8)
        for i in range(volume.shape[0]):
            image = volume[int(i),:,:]
            # image = color.gray2rgb( image )
            image = imresize(image, (525,525))
            path = '%s/%s.jpg'%(p_image, i)
            imsave('%s'%(path), image)


    @staticmethod
    def get_slice(p_h5data, name, index):
        print 'path: ', '%s/%s'%(p_h5data,data_stack_file)
        volume = np.array(h5py.File( '%s/%s'%(p_h5data,data_stack_file) ,'r')[name],dtype=np.float32)
        image = volume[int(index),:,:]
        print np.min(image), np.max(image)
        return image

    def get_pair(self, index1, index2):
        img1 = self.volume[self.index1,:,:]
        img2 = self.volume[self.index2,:,:]
        return img1, img2
    
    def get_triple(self, index1, index2, index3):
        img1 = self.volume[self.index1,:,:]
        img2 = self.volume[self.index2,:,:]
        img3 = self.volume[self.index3,:,:]
        return img1, img2, img3
    