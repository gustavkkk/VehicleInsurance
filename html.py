#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 22:14:22 2017

@author: junying
"""

header = '''
            <!DOCTYPE html>
            <html lang="en"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
                <title>Vehicle Insurance</title>
                <!-- Meta -->
                
                <meta http-equiv="X-UA-Compatible" content="IE=edge">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta name="description" content="Automatically Recognize License Plates from Videos and Images">
                <meta name="keywords" content="ALPR,ANPR,LPR,License Plate Recognition,Open Source,Free,License Plate, Number Plate">
                <link rel="shortcut icon" href="http://www.openalpr.com/favicon.ico">
                <link href="/static/css" rel="stylesheet" type="text/css">
                <link href="/static/css(1)" rel="stylesheet" type="text/css">
                <!-- Global CSS -->
                <link rel="stylesheet" href="/static/bootstrap.min.css">
                <!-- Plugins CSS -->
                <link rel="stylesheet" href="/static/font-awesome.min.css">
            		<link rel="stylesheet" type="text/css" href="/static/jquery.fancybox.css" media="screen">
                <link id="theme-style" rel="stylesheet" href="/static/style.css">
            
            
            	<link rel="stylesheet" type="text/css" href="/static/cloudapidemo.css">
            
            <style type="text/css"></style><style id="fit-vids-style">.fluid-width-video-wrapper{width:100%;position:relative;padding:0;}.fluid-width-video-wrapper iframe,.fluid-width-video-wrapper object,.fluid-width-video-wrapper embed {position:absolute;top:0;left:0;width:100%;height:100%;}</style><script type="text/javascript" charset="utf-8" async="" src="/static/button.b5276da659efda6dff11c91b8160a531.js"></script></head> 
            
            <body class="home-page">   
               
            		<!--//headline-bg-->
            			<div class="page-top-bg page-top-bg--cloud-api"></div>
            		<!--//headline-bg-->
            		<!-- ****** Headline ****** -->
            			<section class="page-head">
            				<div class="page-head__content">
            					<h2 class="page-head__title">Image Analyzer</h2>
            					<p class="page-head__subtitle"></p>
            				</div>
            			</section>
            		<!--// ****** Headline ****** -->
            
            <div class="container">
            	<div class="row">
            		<div class="col-md-12 col-sm-12 col-xs-12">
            			<div class="photos__inner">
            				<div class="photos__sample">
            					<div id="cloud_api_demo" class=""><div class="loading_overlay">
                  <div class="sk-cube-grid">
                    <div class="sk-cube sk-cube1"></div>
                    <div class="sk-cube sk-cube2"></div>
                    <div class="sk-cube sk-cube3"></div>
                    <div class="sk-cube sk-cube4"></div>
                    <div class="sk-cube sk-cube5"></div>
                    <div class="sk-cube sk-cube6"></div>
                    <div class="sk-cube sk-cube7"></div>
                    <div class="sk-cube sk-cube8"></div>
                    <div class="sk-cube sk-cube9"></div>
                  </div>
                   <!-- <svg width="110px" height="110px" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid" class="uil-gears"><rect x="0" y="0" width="100" height="100" fill="none" class="bk"></rect><g transform="translate(-20,-20)"><path d="M79.9,52.6C80,51.8,80,50.9,80,50s0-1.8-0.1-2.6l-5.1-0.4c-0.3-2.4-0.9-4.6-1.8-6.7l4.2-2.9c-0.7-1.6-1.6-3.1-2.6-4.5 L70,35c-1.4-1.9-3.1-3.5-4.9-4.9l2.2-4.6c-1.4-1-2.9-1.9-4.5-2.6L59.8,27c-2.1-0.9-4.4-1.5-6.7-1.8l-0.4-5.1C51.8,20,50.9,20,50,20 s-1.8,0-2.6,0.1l-0.4,5.1c-2.4,0.3-4.6,0.9-6.7,1.8l-2.9-4.1c-1.6,0.7-3.1,1.6-4.5,2.6l2.1,4.6c-1.9,1.4-3.5,3.1-5,4.9l-4.5-2.1 c-1,1.4-1.9,2.9-2.6,4.5l4.1,2.9c-0.9,2.1-1.5,4.4-1.8,6.8l-5,0.4C20,48.2,20,49.1,20,50s0,1.8,0.1,2.6l5,0.4 c0.3,2.4,0.9,4.7,1.8,6.8l-4.1,2.9c0.7,1.6,1.6,3.1,2.6,4.5l4.5-2.1c1.4,1.9,3.1,3.5,5,4.9l-2.1,4.6c1.4,1,2.9,1.9,4.5,2.6l2.9-4.1 c2.1,0.9,4.4,1.5,6.7,1.8l0.4,5.1C48.2,80,49.1,80,50,80s1.8,0,2.6-0.1l0.4-5.1c2.3-0.3,4.6-0.9,6.7-1.8l2.9,4.2 c1.6-0.7,3.1-1.6,4.5-2.6L65,69.9c1.9-1.4,3.5-3,4.9-4.9l4.6,2.2c1-1.4,1.9-2.9,2.6-4.5L73,59.8c0.9-2.1,1.5-4.4,1.8-6.7L79.9,52.6 z M50,65c-8.3,0-15-6.7-15-15c0-8.3,6.7-15,15-15s15,6.7,15,15C65,58.3,58.3,65,50,65z" fill="#8f7f59" transform="rotate(15 50 50)"><animateTransform attributeName="transform" type="rotate" from="90 50 50" to="0 50 50" dur="1s" repeatCount="indefinite"></animateTransform></path></g><g transform="translate(20,20) rotate(15 50 50)"><path d="M79.9,52.6C80,51.8,80,50.9,80,50s0-1.8-0.1-2.6l-5.1-0.4c-0.3-2.4-0.9-4.6-1.8-6.7l4.2-2.9c-0.7-1.6-1.6-3.1-2.6-4.5 L70,35c-1.4-1.9-3.1-3.5-4.9-4.9l2.2-4.6c-1.4-1-2.9-1.9-4.5-2.6L59.8,27c-2.1-0.9-4.4-1.5-6.7-1.8l-0.4-5.1C51.8,20,50.9,20,50,20 s-1.8,0-2.6,0.1l-0.4,5.1c-2.4,0.3-4.6,0.9-6.7,1.8l-2.9-4.1c-1.6,0.7-3.1,1.6-4.5,2.6l2.1,4.6c-1.9,1.4-3.5,3.1-5,4.9l-4.5-2.1 c-1,1.4-1.9,2.9-2.6,4.5l4.1,2.9c-0.9,2.1-1.5,4.4-1.8,6.8l-5,0.4C20,48.2,20,49.1,20,50s0,1.8,0.1,2.6l5,0.4 c0.3,2.4,0.9,4.7,1.8,6.8l-4.1,2.9c0.7,1.6,1.6,3.1,2.6,4.5l4.5-2.1c1.4,1.9,3.1,3.5,5,4.9l-2.1,4.6c1.4,1,2.9,1.9,4.5,2.6l2.9-4.1 c2.1,0.9,4.4,1.5,6.7,1.8l0.4,5.1C48.2,80,49.1,80,50,80s1.8,0,2.6-0.1l0.4-5.1c2.3-0.3,4.6-0.9,6.7-1.8l2.9,4.2 c1.6-0.7,3.1-1.6,4.5-2.6L65,69.9c1.9-1.4,3.5-3,4.9-4.9l4.6,2.2c1-1.4,1.9-2.9,2.6-4.5L73,59.8c0.9-2.1,1.5-4.4,1.8-6.7L79.9,52.6 z M50,65c-8.3,0-15-6.7-15-15c0-8.3,6.7-15,15-15s15,6.7,15,15C65,58.3,58.3,65,50,65z" fill="#9f9fab" transform="rotate(75 50 50)"><animateTransform attributeName="transform" type="rotate" from="0 50 50" to="90 50 50" dur="1s" repeatCount="indefinite"></animateTransform></path></g></svg> -->
                </div>
            
                <div class="demo_preview_img_container img-rounded" style="z-index: 16777271; height: 436.25px;">
            
                    <div class="demo_initial_overlay img-rounded" style="display: none;">
                        Drag and Drop an Image Here<br>
                        Or use the form below to upload.
                    </div>
    '''
result_header = '''
            
                        <div class="demo_preview_area demo_preview_info">
                '''
result_category_header = '''     
                                <div class="demo_info_header">'''
result_category_name = '''License Plate'''
result_category_tail = '''</div>'''
result_value_header = '''       
                                    <div class="demo_info_value">'''
result_value = '''CQY984 - 90.11%'''
result_value_tail = '''</div>'''
result_tail = '''
                        </div>
              '''
image_container_header = '''
                        <div class="demo_preview_area demo_preview_img">
                            <div id="demo_preview_canvas">
                        '''
image_header = '''
                                <img x="0" y="0" width="500" height="416.25" preserveAspectRatio="none" src="uploads/'''
image_fn = "image.jpg"
image_tail = '''" style="-webkit-tap-highlight-color: rgba(0, 0, 0, 0);"></img>'''
image_container_tail = '''
                            </div>
                        </div>
                        '''
tail = '''
                </div>
            
            
            
                <div class="demo_form_controls">
            
                    <div class="form-inline upload_section byfile">
                        <!--<span class="upload_form_label">Upload: </span>-->
            
                        <!--<select class="form-control upload_type">-->
                            <!--<option value="byfile">File</option>-->
                            <!--<option value="byurl">Url</option>-->
                        <!--</select>-->
            
                        <!--<input type="text" class="form-control url_entry" placeholder="Image URL">-->
            
                        <div class="demo_submit_buttons">                          
                                <form action="" method=post enctype=multipart/form-data>
                                  <p><input type=file name=file>
                                     <input type=submit value=Upload>
                                </form>
                        </div>
                    </div>
                </div>
            
                <div class="error_text" style="display: none;"></div></div>
            					<hr>
            				</div>
            			</div>
            		</div>
            	</div>
            </div>
                 
            		<!-- Base template -->
            		<script src="/static/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
            		<script src="/static/jquery-migrate-3.0.0.min.js" integrity="sha256-JklDYODbg0X+8sPiKkcFURb5z7RvlNMIaE3RA2z97vw=" crossorigin="anonymous"></script>
            		<script type="text/javascript" src="/static/bootstrap.min.js"></script>
            		<script type="text/javascript" src="/static/jquery.fancybox.pack.js"></script>
            						<!-- Common js -->
            		<script type="text/javascript" src="/static/main-deps.js"></script>
            		<script type="text/javascript" src="/static/main.js"></script>
            
            		
            		
            <script>
            	var cloudapi_secret_key = 'sk_DEMODEMODEMODEMODEMODEMO';
            </script>
            <script type="text/javascript" src="/static/SimpleAjaxUploader.min.js"></script>
            <script type="text/javascript" src="/static/raphael-min.js"></script>
            <script type="text/javascript" src="/static/bootstrap-multiselect.js"></script>
            <!--<script type="text/javascript" src="/static/cloudapidemo.js"></script>-->
            <script type="text/javascript" src="/static/jquery.imagesGrid.js"></script>
            <!-- Back to top -->
            <script>
            $(document).ready(function(){
            	$("#nav-features").addClass("current");
            	//$("a.plate_sample_images").fancybox();
            	$("a.plate_sample_images").fancybox();
            });
            </script>
            
            		
            		<!-- Google Analytics -->
            		<script>
            			(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
            				(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)})(window,document,'script','//www.google-analytics.com/analytics.js','ga'); ga('create', 'UA-46999297-1', 'openalpr.com'); ga('send', 'pageview');</script>
            
             
            <div id="topcontrol" title="Scroll Back to Top" style="position: fixed; bottom: 5px; right: 5px; opacity: 1; cursor: pointer;"><i class="fa fa-angle-up"></i></div><div style="display: block; position: absolute; overflow: hidden; margin: 0px; padding: 0px; opacity: 0; direction: ltr; z-index: 16777270; visibility: hidden;"><input type="file" name="image" style="position: absolute; right: 0px; margin: 0px; padding: 0px; font-size: 480px; font-family: sans-serif; cursor: pointer; height: 100%; z-index: 16777270;"></div></body></html>
    '''
