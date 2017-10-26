# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 03:52:24 2017

@author: Frank

ref: https://github.com/ArunMichaelDsouza/tensorflow-image-detection
     https://github.com/thanhson1085/Hello-AI
     https://github.com/pyinstaller/pyinstaller
     https://pyinstaller.readthedocs.io/en/latest/usage.html
     https://irwinkwan.com/2013/04/29/python-executables-pyinstaller-and-a-48-hour-game-design-compo/
     https://stackoverflow.com/questions/34453458/how-to-use-pyinstaller
usage: pyinstaller  --onefile --windowed --icon=app.ico classify.py
       pyinstaller classify.spec
       
"""
import tensorflow as tf
import sys
import os
import time
#import cv2

# Millisecond Time
def getCurrentTime():
    return int(round(time.time() * 1000))

# Get curdir
curdir = os.getcwd()

# Code for pyInstaller
def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)

# Disable tensorflow compilation warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Classifier():
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile(resource_path(os.path.join(curdir,"tf_files/retrained_labels.txt")))]
        self.create_graph()
        self.sess = tf.Session()

    def create_graph(self):
        with tf.gfile.FastGFile(resource_path(os.path.join(curdir,"tf_files/retrained_graph.pb")), 'rb') as f:
            print("Restoring parameters from ./tf_files/retrained_graph.pb")
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def run(self, image_path):
        image_data = tf.gfile.FastGFile(resource_path(os.path.join(curdir,image_path)), 'rb').read()
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = self.sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        print('Result for %s' % (image_path))
        ret = ''
        for node_id in top_k:
            human_string = self.label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            if (score > 0.5):
                ret += human_string

        return ret
    
if __name__=="__main__":
    classifier = Classifier()
    start = getCurrentTime()
    image_path = sys.argv[1]
    #cv2.imshow("test",image_data)
    #cv2.waitKey(1000)
    end = getCurrentTime()
    print("inference model consumes %d ms" % (end - start))