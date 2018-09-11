### VehicleInsurance
  1. classify image into three group(that's, VIN,Vehicle,Else)
  2. detect a VIN(vehicle identification number)
  3. detect a vehicle
  4. detect a license plate(including a sloped)
### Setup & Run
    Setup
    $ sudo apt-get install tesseract-ocr libtesseract-dev libleptonica-dev
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
