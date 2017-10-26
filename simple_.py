from hyperlpr import pipline


import cv2

dir_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
file_name = "2.jpg"
fullpath =  dir_path +file_name
# image1 = cv2.imread("./dataset/0.jpg")
# image2 = cv2.imread("./dataset/1.jpg")
#image3 = cv2.imread("./dataset/5.jpg")
# image4 = cv2.imread("./dataset/6.jpg")
image = cv2.imread(fullpath)
#image5 = cv2.imread("./dataset/3144391.png")
#
# pipline.SimpleRecognizePlate(image4)
pipline.SimpleRecognizePlate(image)
#pipline.SimpleRecognizePlate(image5)