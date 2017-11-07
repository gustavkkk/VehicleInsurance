import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from feature.bbox import drawBBox,analyzeBBox

import sys
sys.path.append('../')

from nets import ssd_vgg_300, np_methods#, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization

MINIMUM_VEHICLE_RECT = 30000

class VehicleDetector(object):
    def __init__(self, params=None):    
        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        self.gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
        # Input placeholder.
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)
        # Define the SSD model.
        self.reuse = True if 'ssd_net' in locals() else None
        self.ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
            self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=self.reuse)
        # Restore SSD model.
        self.ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
        # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.anchors(self.net_shape)

        self.isess = tf.InteractiveSession(config=self.config)     
        # Load Model     
        self.isess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.isess, self.ckpt_filename)
        
    def __enter__(self):
        return self
    
    # Main image processing routine.
    def process_image(self,
                      img,
                      select_threshold=0.5,
                      nms_threshold=.45,
                      net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d,self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes
    

    
    def detect(self,
               img,
               debug=False):
        
        rclasses, rscores, rbboxes =  self.process_image(img)
        #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        # Refine BBoxes
        bbox = VehicleDetector.pick_one_vehicle(img,rclasses, rscores, rbboxes)
        if debug:
            drawBBox(img,[bbox],drawmline=False)
        return bbox
                
    
    def detect_by_filename(self,
                           path,
                           debug=False):        
        # Load File
        img = mpimg.imread(path)#img = cv2.imread(path)
        # Detect
        bbox = self.detect(img)#mpimg.imread(path)
        # Check Result
        if bbox is not None and debug:
            drawBBox(img,[bbox])
        ##
        return bbox
        
    def detect_by_data(self,
                       img,
                       debug=False):         
        # Detect
        bbox = self.detect(img)#mpimg.imread(path)
        # Check Result
        if bbox is not None and debug:
            drawBBox(img,[bbox])
        ##
        return bbox

    @staticmethod    
    def rbbox2bbox(img,
                   rbbox):
        height,width,channels = img.shape
        ymin = int(rbbox[0] * height)
        xmin = int(rbbox[1] * width)
        ymax = int(rbbox[2] * height)
        xmax = int(rbbox[3] * width)
        return np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
    
    @staticmethod
    def pick_one_vehicle(img,rclasses, rscores, rbboxes):
        size = 0
        selected = None
        for i,rclass in enumerate(rclasses):
            if rclass == 7 and rscores[i]>0.5:
                bbox = VehicleDetector.rbbox2bbox(img,rbboxes[i])
                pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degrees = analyzeBBox(bbox)
                if w*h > size and w*h>30000:
                    selected = bbox
                    size = w*h
        return selected
    
    @staticmethod   
    def pick_vehicles(img,rclasses, rscores, rbboxes):
        bboxes = []
        sizes = []
        for i,rclass in enumerate(rclasses):
            if rclass == 7 and rscores[i]>0.5:
                bbox = VehicleDetector.rbbox2bbox(img,rbboxes[i])
                bboxes.append(rbboxes[i])
                pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
                sizes.append(w*h)
        return bboxes
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.isess.close()
        #self.ssd_net.cleanup()
        '''
        self.gpu_options.cleanup()
        self.config.cleanup()
        # Input placeholder.
        self.net_shape.cleanup()
        self.data_format.cleanup()
        self.img_input.cleanup()
        # Evaluation pre-processing: resize to SSD net shape.
        self.image_pre.cleanup()
        self.labels_pre.cleanup()
        self.bboxes_pre.cleanup()
        self.image_4d.cleanup()
        # Define the SSD model.
        self.reuse.cleanup()
        self.ssd_net.cleanup()
        
        self.predictions.cleanup()
        self.localisations.cleanup()
        # Restore SSD model.
        self.ckpt_filename.cleanup()
        # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        # SSD default anchor boxes.
        self.ssd_anchors.cleanup()

        self.isess.cleanup()
        # Load Model     
        self.saver.cleanup()
        '''
        
###
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "12.jpg"
fullpath = dataset_path + filename
###
def main():
    detector = VehicleDetector()
    detector.detect_by_filename(fullpath,True)   
if __name__ == '__main__':
    print("Under Test")
    #main()
    