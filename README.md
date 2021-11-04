# Description
The motavation for this project was aimed to tackle the competition that was set by IMechE UAS challenge which has a task to identify the colour and alphanumeric character with a square target.

This repository contains the scripts for the following tasks: character and colour recognition and also target detection. The target this script is tailored for is a square target that contains another square which contains a colour and an alphanumeric character that is white. as shown below:

<img src="test_images/L_georgia_fuchsia.png" alt="L" width="100">

It contained a few test images for you to understand how it works and it can work with your webcam as well as long you have adjusted the settings in the **config.py**

# How-to-use

Within this code, majority of the controls could be changed within the file called **config.py** which contains operations such as: 
  <li>Switching between K-NN and Tesseract or using RGB or HSV. </li>
  <li>Visual of the important stages you that wish to enable and see. </li>
  <li>Testing the system either by video or an image.</li>
  <li>It also contains the settings for what device you are using the scripts for either a raspberry pi or PC.</li>

However, if you wish to finetune the scripts, they are seperated to their task that they are focused on. 
You can find the target detection inside the **main.py**.

To run the code type into the command line / Terminal the following command: ```python -m main.py```

# Issue
  <li>It is only able to detect the a single square target per frame/image.</li>

# Dependency
You can use the requirement.txt for an ease of installing the following python packages.

  <li>opencv-python</li>
  <li>numpy</li>
  <li>webcolors</li>
  <li>pytesseract</li>

if using on the raspberry pi, you need:
  <li>picamera</li>




