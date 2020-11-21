### VehicleInsurance
  This project is aiming at validating pictures of accident spot for vehicle insurances. The service only accepts meaningful
  pictures while rejects the meaningless. It let uploading one know rejection reason with explanation to help them correct it.

  1. classify image into three group(that's, VIN,Vehicle,Else)
  2. detect a VIN(vehicle identification number)
  ![known](https://github.com/gustavkkk/VehicleInsurance/blob/master/imgs/vin.png)
  3. detect a vehicle
  4. detect a license plate(including a sloped)
  ![known](https://github.com/gustavkkk/VehicleInsurance/blob/master/imgs/lp.png)
  
### Setup & Run
    Setup
    Install Tesseract
    $ sudo apt-get install tesseract-ocr libtesseract-dev libleptonica-dev tesseract-ocr-all
    $ sudo pip install tesserocr
    Install Opnecv3
    $ cd ~
    $ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.0.zip
    $ unzip opencv.zip
    $ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.0.zip
    $ unzip opencv_contrib.zip
    $ cd ~
    $ wget https://bootstrap.pypa.io/get-pip.py
    $ sudo python get-pip.py
    $ sudo apt-get update
    $ sudo apt-get upgrade
    $ sudo apt-get install build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python2.7-dev python3.5-dev
    $ cd ~/opencv-3.1.0/
    $ mkdir build
    $ cd build
    $ cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
        -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
        -D BUILD_EXAMPLES=ON ..
    $ make -j4
    $ make
    $ make clean
    $ sudo make install
    $ sudo ldconfig

    Install Python Packages
    $ git clone https://github.com/gustavkkk/VehicleInsurance
    $ cd VehicleInsurance
    $ sudo pip install -r requirements
    
    Run
    $ python webapp.py

### Download
  You can download [models & others](https://pan.baidu.com/s/1HRePvv0UVibMsXKxJOUs7g) from the following ulrs.
    
    url: https://pan.baidu.com/s/1HRePvv0UVibMsXKxJOUs7g
    password: jwyz
    
### Reference
  - [MicrocontrollersAndMore](https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Python)
  - [tesserocr](https://github.com/sirfz/tesserocr)
