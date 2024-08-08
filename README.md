This is a revised algorithm that detects objects from both pre-loaded images and webcam feeds. The algorithm uses mainly OpenCV, with YOLO v3, as well as COCO for training. The quality of the outputs depends on the quality of input images as well as the webcam. The COCO class labels, YOLO weights and its configurations can be found as follows: 

yolov3.weights file: https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights

coco.names file: https://github.com/pjreddie/darknet/blob/master/data/coco.names

yolov3.cfg file: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg 

The images loaded are trained with following sample results: 

![output1](https://github.com/user-attachments/assets/a3124309-447a-47b8-b821-9d23b0322094)

![output2](https://github.com/user-attachments/assets/18f5d649-e401-462a-85a4-fa63a89f1b22)

The live feed from laptop's webcam also shows as follows: 

![output3](https://github.com/user-attachments/assets/8781300c-a3a5-44c0-a29b-847aea4524dc)



Reference: 

[1] https://github.com/patrick013/Object-Detection---Yolov3

[2] https://machinelearningspace.com/coco-dataset-a-step-by-step-guide-to-loading-and-visualizing/

[3] https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

[4] https://medium.com/@mikolaj.buchwald/yolo-and-coco-object-recognition-basics-in-python-65d06f42a6f8

[5] https://github.com/pjreddie/darknet/tree/master

