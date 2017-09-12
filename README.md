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

* Download weight files. This will download neccessary weight files for this project, including 'Mobilenet', 'DenseNet', 'YOLov2'
```
python weights/download_weights.py
```

* Detect objects in a given image using original YOLOv2
```
python predict.py --weight_path yolov2.weights test_image.jpg 
```

## Train on custom data set
        
**Step 1: Prepares training data**

In this project, we only accept training input fil as text file as following format:
```
path/to/image1.jpg, x1, y1, x2, y2, class_name1
path/to/image2.jpg, x1, y1, x2, y2, class_name3
path/to/image3.jpg, x1, y1, x2, y2, class_name4
...
path/to/imagen.jpg, x1, y2, x2, y2, class_name6
```

Assumption:
* If one image contains more than one objects, it would be split into multiple lines.
* `x1, y1, x2, y2` are absolute cooridinates.
* `class_name` are a string, no space or special characters.
        
        
**Step 2: Generate dataset for training **

```
python create_dataset.py
   --path       /path/to/text_file.txt
   --output_dir ./dataset/my_new_dataset
   --n_anchors   5
   --split       false
```

It will create the following files:
```
yolov2
|
|- dataset
     | - my_new_data_set
         |
         | --  categories.txt
         | --  anchors.txt
         | --  training_data.csv
         | --  testing_data.csv   # if split is enabled
```


**Step 3: Update parameters in `cfg.py` file**
Example:
```
FEATURE_EXTRACTOR     = 'yolov2'
IMG_INPUT_SIZE = 480
N_CLASSES      = 61
N_ANCHORS      = 5

# Map indices to actual label names
CATEGORIES = "./dataset/my_new_dataset/categories_tree.txt"
ANCHORS    = "./dataset/my_new_dataset/anchors.txt"
```

**Step 4: Fine-tune for your own dataset**

* Training on your own datset
```angular2html
python train.py  \
   --epochs        100
   --batch         8
   --learning_rate 0.00001
   --training_data your_own_data.txt 
```

** Step 5: Evaluate your finetuned model
```
python evaluate.py 
---weights yolov2.weights
---csv_path ./dataset/my_new_dataset/testing_data.csv
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
