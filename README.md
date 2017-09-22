# YOLOv2 - Object Detection Model on Keras
-------------------------------------------

This repo is the  implementation of YOLOv2, an object detector using Deep Learning, discussed in ["YOLO9000: Better, Faster, Stronger"](https://arxiv.org/abs/1612.08242)

Project Status: **Under Development!!**

## Dependencies

* Set up environment
```
conda env create -f environment.yml
```
* Activate environment
```
source activate yolo
```
* Install OpenCV
```
conda install -c menpo opencv=2.4.11
```


## Usage

* Download weight files and place under `weights` directory

```
https://drive.google.com/open?id=0ByoFGh573uhzWGx0aXNNNkY5Qms
```

* Detect objects in a given image using original YOLOv2
```
python predict.py -p ./test_imgs/multiple_stop.jpg -w ./weights/yolo-coco.weights -o ./
```

## TODO List:
- [x] Generate anchors using K-mean clustering on training data
- [x] Convert DarkNet19 weights to Keras weights
- [x] YOLOv2 Loss Function
- [x] Hierarchical Tree
- [x] Train on any custom data set
- [x] Use MobileNet/DenseNet as feature extractor.


## Acknowledgement
Thank you Dr. Christoph Merzt, Jina Wang, fellow scholars of RISS 2017 and Carnegie Mellon University for providing extraordinary support, resources and mentorship to help me complete this project.
