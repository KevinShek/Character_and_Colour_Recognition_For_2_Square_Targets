# Description
The motavation for this project was aimed to tackle the competition that was set by IMechE UAS challenge which has a task to identify the colour and alphanumeric character with a square target.

This repository contains the scripts for the following tasks: character and colour recognition and also target detection. The target this script is tailored for is a square target that contains another square which contains a colour and an alphanumeric character that is white. as shown below:

![](Test_Images/L_georgia_fuchsia.png | width = 100)

Within this file majority of the controls could be changed within the file called config.py which contains operations such as: 
switching between which K-NN and Tesseract or using RGB or HSV. 
Visual of the important stages you that wish to enable and see. 
Testing the system either by video or an image.
Testing the system for specific operation such as character or colour recognition.
It also contains the settings for what device you are using the scripts for either for a raspberry pi or PC.

However, if you wish to fine tune the scripts, they are seperated to their task that they are focused on. You can find the target detection inside the Main.py.

To run the script type ```python -m Main.py```

# Issue
- It is only able to detect the first square target within the image.

# Dependency
-opencv-python
-numpy
-webcolors
-pytesseract

if using on the raspberry pi, you need:
-picamera




