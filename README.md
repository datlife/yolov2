# YOLOv2 - Object Detection Model
---------------------------------
This repo is the  implementation of YOLOv2, an object detector using Deep Learning, discussed in ["YOLO9000: Better, Faster, Stronger"](https://arxiv.org/abs/1612.08242)

Project Status: **Under Development!!**

## Dependencies

* Set up environment
```
# at ./yolov2 
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
# At ./yolov2 and conda environment has been activated
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
   * `class_name` are a string, no space or special characters
        
**Step 2: Generate dataset for training

```
python create_dataset.py
   --path       /path/to/text_file.txt
   --output_dir ./dataset/my_new_dataset
   --n_anchors   5
   --split       false
```

**Step 3: Update parameters in `cfg.py` file**

```
N_CLASSES     = 31   # <---- Number of classes in your data set
AUGMENT_LEVEL = 5    # The higher, the more data is augmented
ANCHORS       = np.array(((0.023717899133663362, 0.035715759075907606),  # <--- Update from anchors.txt
                          (0.059577141608391594, 0.08738709207459215),
                          (0.08816276658767774, 0.1294924960505529),
                          (0.03283318210930825, 0.0483890193566751),
                          (0.04450034340659346, 0.064308608058608)))
SHRINK_FACTOR = 32   # **For DarkNet19 only (max-pool 5 times) ** 2^5 = 32. If using different feature extractor, update accordingly
```

**Step 4: Fine-tune for your own dataset**

* Training on your own datset
```angular2html
./train.py --weight_path yolov2.weights --training_data your_own_data.txt --epochs 1000
```


## Run YOLOv2 using a webcam as input
```angular2html
./webcam.py --weight_path yolov2.weights
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
