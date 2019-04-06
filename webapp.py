#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 06:05:39 2017

@author: ubuntu
"""

"""
Created on Tue Sep 26 22:21:28 2017
@author: Frank
ref: https://stackoverflow.com/questions/10434599/how-to-get-data-received-in-flask-request
     https://www.reddit.com/r/flask/comments/5fl0xn/multifile_api_using_requests/
     https://github.com/mastercoder82/flask-test
     
     http://maximebf.com/blog/2012/10/building-websites-in-python-with-flask/#.WdcK0nVSw8o
     
     https://stackoverflow.com/questions/29221045/handling-urls-in-css-files-with-flask
     https://stackoverflow.com/questions/6978603/how-to-load-a-javascript-or-css-file-into-a-bottlepy-template
     
     #https://stackoverflow.com/questions/20900281/how-to-disable-flask-cache-caching
     #https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
     
     http://flask.pocoo.org/docs/0.10/api/#flask.Flask.after_request
"""

import os
from flask import Flask, request, redirect, url_for,jsonify
from werkzeug import secure_filename
#import json
import time
import cv2
import numpy as np
#from os.path import basename

from feature.colorspace import opencv2skimage#,rgb2hsv,skimage2opencv
from feature.bbox import cropImg_by_BBox,drawBBox,shiftBBoxes#,showResult
#from misc import pick_one_vehicle
from quality import AnalyzeImageQuality
#from classify import Classifier
from detector import LicensePlateDetector,VehicleDetector
from vin import VIN
from licenseplate import LicensePlate
from detect_angle import detect_angle
import html

from misc.switch import switch
#from os import listdir
from os.path import exists#,isfile, join

app = Flask(__name__)#, static_folder='static', static_url_path='')

UPLOAD_FOLDER = 'uploads'
WEBFILE_FOLDER = 'webfiles'
app.qualityanalyzer = AnalyzeImageQuality()
#app.classifier = Classifier()
app.detector = VehicleDetector()
app.licenseplatedetector = LicensePlateDetector()
app.licenseplate = LicensePlate()
app.vehicleidentifier = VIN()
app.results = []
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['WEBFILE_FOLDER'] = WEBFILE_FOLDER
app.config["CACHE_TYPE"] = "null"
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024    # 1 Mb limit
app.image_fn = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
app.result_fn = os.path.join(app.config['UPLOAD_FOLDER'], "result.txt")
app.filename = ""
#cache = Cache(config={'CACHE_TYPE': 'redis'})
        
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def processImg(file_path,filename):
    #thisiswhat = app.classifier.run(file_path)
    thisiswhat = 'lp' #, 'vin']
    image = cv2.imread(file_path)
    for case in switch(thisiswhat):
        app.results.append(filename + " :")
        if case('lp'):
            app.results.append("   Live photo")
            # Detect Vehicle           
            bbox_car = app.detector.detect(opencv2skimage(image))#mpimg.imread(path)
            if bbox_car is not None:
                img_car = cropImg_by_BBox(image,bbox_car)
                app.results.append(r"Car : Yes ")
                # Detect License Plate
                confidence,bboxes_lp,rois = app.licenseplatedetector.process(img_car)
                markImg = processLP(image,bbox_car,bboxes_lp,confidence)
                cv2.imwrite(file_path,markImg)
            else:
                app.results.append(r"Car : No Front")
                confidence,bboxes_lp,rois = app.licenseplatedetector.process(image)
                markImg = processLP(image,None,bboxes_lp,confidence)
                if markImg is not None:
                    cv2.imwrite(file_path,markImg)
            break
        if case('vin'):
            app.results.append(r"   VIN")
            app.vehicleidentifier.initialize()
            isFound,confidence,markImg = app.vehicleidentifier.process(image)
            if isFound:
                app.results.append(r"VIN : HAS")
                cv2.imwrite(file_path,markImg)
            else:
                app.results.append(r"VIN : HASn't")
            break
        if case():
            app.results.append(r"   No Idea")
            break
        
def processLP(image,bbox_car,bboxes_lp,confidence):
    markImg = None
    if bboxes_lp is not None: 
        # Select LP with highest confidence
        bbox_lp = bboxes_lp[0]
        ###
        if confidence > 0.5:
            app.results.append(r"License plate : Yes")
            # Mark Image
            bbox_lp_refined = shiftBBoxes(bbox_car,[bbox_lp]) if bbox_car is not None else bbox_lp
            markImg = drawBBox(image,[bbox_lp_refined],bbox_car)  
            #
            if confidence > 0.85 and bbox_car is not None:
                app.results.append(r"License plate : FRONT")                                    
                res,reps = detect_angle(image,bbox_lp_refined,bbox_car)
                app.results.append(r"Analysis result : " + res)
                for rep in reps:
                    app.results.append(rep)
            else:
                app.results.append(r"Analysis result : No Pass")
                app.results.append(r"车牌 : 不全面")

        else:
            app.results.append(r"License plate : NO")
            markImg = drawBBox(image,None,bbox_car)
    else:
        app.results.append(r"License plate : No Front")
        markImg = drawBBox(image,None,bbox_car)
    return markImg
            
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':        
   
        file = request.files['file']
        

        if file and allowed_file(file.filename):
            # Delete tmp files
            # if exists(app.image_fn):
            #    os.remove(app.image_fn)
            if exists(app.result_fn):
                os.remove(app.result_fn)
            app.results[:] = []
            # save an uploaded file to "uploads" folder
            filename = secure_filename(file.filename)
            app.filename = filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.filename)
            file.save(file_path)            
            #####################
            # Analyze the Image #
            #####################
            start=time.time()
            # Load Image
            image = cv2.imread(file_path)
            # Quality Analyzing
            app.qualityanalyzer.initialize()
            fm,result,explanation = app.qualityanalyzer.process(image)
            for case in switch(explanation):
                if case('Normal'):
                    # Determine what this is
                    processImg(file_path,filename)
                    break
                if case('Blurry'):
                    app.results.append(r"   blurry")
                    break
                if case('Reflective'):
                    app.results.append(r"   Not clear")
                    processImg(file_path,filename)
                    break
                if case('Too Reflective'):
                    app.results.append(r"   Overexposed")
                    break
            end=time.time()
            elapsedtime = end-start
            app.results.append(r"Elapsed time : " + str(int(elapsedtime*1000)/1000.0) + "s ")
            #######################
            #######################
            # Save Image and Result
            #os.rename(file_path, app.image_fn)
            with open(app.result_fn, 'w') as file:
                for result in app.results:
                    file.write(result)
                    file.write('\n')
                    
            return redirect("/")
            #return redirect(url_for('uploaded_file',
            #                        filename="facedetect-"+filename))

    results = ""
    for result in app.results:
        results += html.result_value_header + result + html.result_value_tail
    image = ""
    if exists(os.path.join(app.config['UPLOAD_FOLDER'], app.filename)):
        image = html.image_header + app.filename + '?' + str(time.time()) + html.image_tail 
    return  html.header + \
            html.result_header + \
            results + \
            html.result_tail +\
            html.image_container_header +\
            image + \
            html.image_container_tail + \
            html.tail

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
            
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
    
from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})
    
if __name__ == "__main__":
    app.run(host= '0.0.0.0', debug=True, port=4000)