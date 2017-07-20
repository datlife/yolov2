# YOLOv2
---------
This repo is the  implementation of YOLOv2, an object detector using Deep Learning, discussed in ["YOLO9000: Better, Faster, Stronger"](https://arxiv.org/abs/1612.08242)

Project Status: **Under Development!!**

Download: [YOLOv2 Keras Weights]()

## TODO List:
- [x] Generate anchors using K-mean clustering on training data
- [x] Convert DarkNet19 weights to Keras weights
- [x] YOLOv2 Loss Function
- [x] Multi-scale  Training
- [ ] LSTM Tracker
- [ ] Hierarchical Tree - Combine multiple data sets like YOLO9000 using
- [ ] Train on any custom data set
- [ ] Use MobileNet as feature extractor (super fast, accuracy ~ VGG-16)

## Examples

* Detect objects in a given image
```
./predict.py --weight_path yolov2.weights test_image.jpg 
```
## Train on custom data set
        
**Step 1: Prepares training data**

In this project, we only accept training input file as text file as following format:
      
        path/to/image1.jpg, x1, y1, x2, y2, class_name1
        path/to/image2.jpg, x1, y1, x2, y2, class_name3
        path/to/image3.jpg, x1, y1, x2, y2, class_name4
        ...
        path/to/imagen.jpg, x1, y2, x2, y2, class_name6
        
        
**Step 2: Generate Anchors for your data to train faster**

It will generate anchors.txt in the same directory
```
python gen_anchors.py --num_anchors 5 --label_bath training.txt --img_width 1280 --img_height 960
```

**Step 3: Update parameters in cfg file**

```
N_CLASSES     = 31   # <---- Number of classes in your data set
N_ANCHORS     = 5    # < ---- Must the the same as number of generated anchors
SHRINK_FACTOR = 32   # **For DarkNet19 only (max-pool 5 times) ** 2^5 = 32. If using different feature extractor, update accordingly
AUGMENT_LEVEL = 5    # The higher, the more data is augmented
MULTI_SCALE   = [0.25, 0.5, 0.75, 1.0, 1.25, 1.75]
ANCHORS       = np.array(((0.023717899133663362, 0.035715759075907606),  # <--- Update from anchors.txt
                          (0.059577141608391594, 0.08738709207459215),
                          (0.08816276658767774, 0.1294924960505529),
                          (0.03283318210930825, 0.0483890193566751),
                          (0.04450034340659346, 0.064308608058608)))
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
## Acknowledgement
Thank you Dr. Christoph Merzt, Jina Wang, fellow scholars of RISS 2017 and Carnegie Mellon University for providing extraordinary support, resources and mentorship to help me complete this project.
