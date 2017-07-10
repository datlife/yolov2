# YOLOv2
---------
This repo is the  implementation of YOLOv2, an object detector using Deep Learning, discussed in ["YOLO9000: Better, Faster, Stronger"](https://arxiv.org/abs/1612.08242)

Project Status: **Under Development!!**

Download: [YOLOv2 Keras Weights]()

## TODO List:
- [x] Generate anchors using K-mean clustering on training data
- [x] Convert DarkNet19 weights to Keras weights
- [ ] YOLOv2 Loss Function
- [ ] Multi-GPU(s) Training
- [ ] Use MobileNet as feature extractor (super fast, accuaracy ~ VGG-16)
- [ ] Train on any custom data set
- [ ] Convert to ROS package (use video stream as input)
- [ ] Hierarchical Tree - Combine multiple data sets like YOLO9000 using

## Examples

* Detect objects in a given image
```
./predict.py --weight_path yolov2.weights test_image.jpg 
```

## Train on custom data set

### 1. Pre-process training data
Input data: list of imageas in `.txt`  as format

        
        path/to/image1.jpg, x1, y1, x2, y2, class_name1
        path/to/image2.jpg, x1, y1, x2, y2, class_name3
        path/to/image3.jpg, x1, y1, x2, y2, class_name4
        ...
        ...
        path/to/imagen.jpg, x1, y2, x2, y2, class_name6
        
### 2. Train YOLOv2
```angular2html
./train.py --weight_path yolov2,weights --training_data training.txt
```

## Run YOLOv2 using a webcam as input
```angular2html
./webcam.py --weight_path yolov2.weights
```
## Acknowledgement
Thank you Dr. Christoph Merzt, Jina Wang, fellow scholars of RISS 2017 and Carnegie Mellon University for providing extraordinary support, resources and mentorship to help me complete this project.
