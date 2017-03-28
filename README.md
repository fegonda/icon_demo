#INSTRUCTIONS

#Installation onto linux system

##Code and Data
Get the code from github using git command as follows:
> git https://github.com/fegonda/icon_jeff.git

##Install database
Run the following command inside the directory icon_jeff:
> sh  install.sh


##Running the web server:
You need to login to odyssey on a specific port in order to access the system from a browser.  
For example to login using port 8889 you need to login to a specific rclogin machine as follows:

> ssh -L 8889:localhost:8889 yourusername@rclogin09.rc.fas.harvard.edu

Then after login, you need to srun to a partition where you can run icon from:

> srun --pty -p cox --mem 8000 -t 300 --tunnel 8889:8888 --gres=gpu:1 -n 4 -N 1 bash


#Starting icon services

> Start the web server by running: 
sh web.sh

## Running the training thread
Start the training thread by running: 
> sh train.sh

## Running the segmentation thread
Start the segmentation thread by running: 
> sh segment.sh


#Web Access
## Running the front-end.
Access the UI by launching the following URL on a browser: 
http://localhost:8888/browse

## Usability
- Then select a project from the drop down list. Press the start button to activate a project or stop to deactivate. Only one project can be active at a time.
- Click on an image form the browser screen to launch the annotation screen
- The annotation screen has a set of tools on the left.  
-- To annotate, you select a label color, then select the brush icon and then draw on the image surface.  
-- To remove annotations, you select a label color, then the eraser icon then draw on the image surface.
-- The L icon toggles on/off annotations
-- The S icon toggles on/off segmentations

