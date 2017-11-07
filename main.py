# Main.py

import tensorflow as tf
#import sys
import cv2
#import numpy as np

from nets import ssd_vgg_300, np_methods#, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

slim = tf.contrib.slim

from feature.colorspace import opencv2skimage#,skimage2opencv,rgb2hsv,
from feature.bbox import cropImg_by_BBox,drawBBox,shiftBBoxes#,showResult,masked
from detect_vehicle import VehicleDetector

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)
# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
# Restore SSD model.
ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

isess = tf.InteractiveSession(config=config)     
# Load Model     
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
    
# Main image processing routine.
def process_image(img,
                  select_threshold=0.5,
                  nms_threshold=.45,
                  net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def detect(img,
           debug=False):
    
    rclasses, rscores, rbboxes =  process_image(img)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    # Refine BBoxes
    bbox = VehicleDetector.pick_one_vehicle(img,rclasses, rscores, rbboxes)
    if debug:
        drawBBox(img,[bbox],drawmline=False)
    return bbox

###               
sample_db_path = "./sample/"
test_db_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "11.jpg"#74,87,29,37
fullpath = test_db_path + filename
###
# MSER
#https://docs.opencv.org/master/d3/d28/classcv_1_1MSER.html
#ref:http://answers.opencv.org/question/19015/how-to-use-mser-in-python/
#https://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html#mser
#https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x
#https://fossies.org/dox/opencv-3.3.0/mser_8py_source.html
# Text Detection
#http://blog.csdn.net/windtalkersm/article/details/53027685
#http://digital.cs.usu.edu/~vkulyukin/vkweb/teaching/cs7900/Paper1.pdf
# Fill in contour
#https://stackoverflow.com/questions/44185854/extract-text-from-image-using-mser-in-opencv-python
# opencv Python tutorial
#https://github.com/opencv/opencv/tree/master/samples/python
from licenseplate import LicensePlate
from detector import LicensePlateDetector
###
import time

lp = LicensePlate()
lpdetector = LicensePlateDetector()

def main():
    start=time.time()
    # Load Image
    image = cv2.imread(fullpath)  
    # Load Model
    #detector = VehicleDetector()
    # Detect Vehicle           
    bbox_car = detect(opencv2skimage(image))#mpimg.imread(path)
    if bbox_car is not None:
        img_car = cropImg_by_BBox(image,bbox_car)
        # Detect License Plate
        start_detect_lp = time.time()
        confidence,bboxes_lp,rois = lpdetector.process(img_car)
        print("detecting LP elapsed time: "+str(int((time.time() - start_detect_lp)*1000)/1000.0)+"s")
        # Check Result
        if bboxes_lp is not None:
            bboxes_lp_refined = shiftBBoxes(bbox_car,bboxes_lp)
            print "confidence:",confidence
            drawBBox(image,bboxes_lp_refined,bbox_car,debug=True)
            '''
            for roi in rois:
                start_refine_lp = time.time()
                lp.process(roi,isdebug=True)
                print("refining LP elapsed time: "+str(int((time.time() - start_refine_lp)*1000)/1000.0)+"s")
                #pipline.recognizeLP2(seg_blocks,mid)
                print("confidence:"+str(lp.confidence*100)+"%")
            '''

        else:
            drawBBox(image,None,bbox_car,debug=True)
    else:
        start_detect_lp = time.time()
        confidence,bboxes_lp,rois = lpdetector.process(image)
        print("detecting LP elapsed time: "+str(int((time.time() - start_detect_lp)*1000)/1000.0)+"s")
        if bboxes_lp is not None:
            print "confidence:",confidence
            drawBBox(image,bboxes_lp,None,debug=True)
        
        
    print("total elapsed time: "+str(int((time.time() - start)*1000)/1000.0)+"s")

##
if __name__ == "__main__":
    main()

































    
'''
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_yen,threshold_li,
                             threshold_sauvola)
from skimage import io, morphology, img_as_bool, segmentation
from skimage import img_as_ubyte
from feature.threshold import windowthreshold,compositeThreshold,kojy_gray

from misc.preprocess import maximizeContrast

from feature.detect import textdetector
from feature.shadowremoval import removeshadow,removeshadow2

def fillin(binary):
    #
    h,w = binary.shape
    filled = binary.copy()
    for j in range(h):
        if j > h*0.2 and j < h*0.8:
            continue
        tmp = binary[j,:]
        if (tmp == 0).sum() > w*0.8 or (j == 0 or j == (h-1)):
            filled[j,:] = 255
    for i in range(w):
        if i > w*0.1 and i < w*0.9 or (i == 0 or i == (w-1)):
            continue
        tmp = binary[:,i]
        if (tmp == 0).sum() == h:
            filled[:,i] = 255                #
    showResult("final",filled)
    return filled
    
roi = cv2.resize(roi, (160, 44),interpolation = cv2.INTER_CUBIC)
showResult("roi",roi)
#removeshadow(roi)
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
denoised = cv2.GaussianBlur(gray,(5,5),0)
#
sobel_horizontal = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = 3)
sobel_vertical = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
sobel = cv2.min(sobel_horizontal,sobel_vertical)
print np.amin(sobel),np.amax(sobel)
sobel -= np.amin(sobel)
sobel = sobel * 255 / (np.amax(sobel) - np.amin(sobel))
#sobel[sobel<=0] = 0
#sobel[sobel>0] = 255
sobel = sobel.astype('uint8')
showResult("sobel",sobel)
#showResult("sobel_horizontal",sobel_horizontal)
#showResult("sobel_vertical",sobel_vertical)
#ret,lthr1 = cv2.threshold(sobel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
lthr1 = cv2.adaptiveThreshold(sobel, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
showResult("lthr",lthr1)
#               
laplacian = cv2.Laplacian(denoised,cv2.CV_64F)
print np.amin(laplacian),np.amax(laplacian)
laplacian -= np.amin(laplacian)
laplacian = laplacian * 255 / (np.amax(laplacian) - np.amin(laplacian))
laplacian[laplacian>255]=255
laplacian = laplacian.astype('uint8')
#ret,lthr2 = cv2.threshold(laplacian,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
lthr2 = cv2.adaptiveThreshold(laplacian, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
showResult("laplacian",laplacian)
edges = cv2.Canny(denoised,50,150,apertureSize = 3)
showResult("canny",edges)  
showResult("lthr2",lthr2)
textdetector(laplacian)

#roi = roi1
gray = 255 - gray
h,s,v = cv2.split(hsv)
kojy,fu = kojy_gray(roi)
#gray = maximizeContrast(kojy)
showResult("comthr",fu)
showResult("comthr",compositeThreshold(lthr2,mode='niblack-multi'))
#          
showResult("kojy",kojy)
thr = windowthreshold(kojy)#,mode='li')
showResult("thr",thr)
thr = cv2.max(thr,edges)
showResult("thr",thr)
#
img_contour = np.zeros((thr.shape),dtype=np.uint8)
chars = find_possible_chars_in_LP(lthr2,True)
drawChars(img_contour,"first-filtering",chars,isdebug=True)
print len(chars)
'''

















