
Shadow detection and removal on digital images

Based on publicly available scientific papers I implemented two detections and two removal methods which only uses the image itself
to detect and remove shadows with different rates of success.

The methods were implemented based on these papers:<br>

First shadow detection and removal method: <br>
https://www.researchgate.net/publication/299509760_Removal_of_Shadows_from_a_Single_Image <br>
https://www.researchgate.net/publication/274563892_Shadow_Detection_and_Removal_from_a_Single_Image_Using_LAB_Color_Space <br>

Second shadow detection method: <br>
https://ieeexplore.ieee.org/document/5156272

Second shadow removal method: <br>
https://link.springer.com/article/10.1007/s11042-023-16282-0


Run gui.py to start the program with GUI <br>
Run menu.py to start the program without gui, you can uncomment which methods you want to use

As metrics the mean square error is calculated for the shadow mask and image, peak signal-to-noise ratio is calculated as well for the image.<br>
To calculate those the ground truth shadow mask and image is needed, same size and name as your chosen image, put them in their respective folders.<br>
It can be either images with .jpg and .png extension.<br>
If the images size or channel count doesn't match, metrics won't be calculated.