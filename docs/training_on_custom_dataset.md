# Train on custom data set
----------------------------

## Step 1: Data Preparation

In this project, we create a simple interface which user only needs to create a simple .txt file and the program would handle the rest (generate anchors, categories, dataset prep).

This text file need to following this format
```
/absolute/path/to/image1.jpg, x1, y1, x2, y2, class_name1
/absolute/path/to/image2.jpg, x1, y1, x2, y2, class_name3
/absolute/path/to/image3.jpg, x1, y1, x2, y2, class_name4
...
/absolute/path/to/imagen.jpg, x1, y2, x2, y2, class_name6
```

**Assumption**:
* Path to image is absolute path
* `x1, y1, x2, y2` are absolute cooridinates.
* `class_name` are a string, no space or special characters. (e.g. `TrafficSign`, `StopSign`)
* If an image contains more than one object, each object would be split into multiple lines.
        
        
## Step 2: Generate dataset for training

```
python create_dataset.py
   --path             /path/to/text_file.txt
   --output_dir       ./dataset/my_new_dataset
   --number_anchors   5
   --split            True
```

It will create the following files:
```
yolov2
|- dataset
     | - my_new_data_set
         | --  categories.txt
         | --  anchors.txt
         | --  training_data.csv
         | --  validation_data.csv   
```


## Step 3: Update parameters in `cfg.py`

Example:
```
FEATURE_EXTRACTOR  = 'yolov2'
IMG_INPUT_SIZE     = 608
N_CLASSES          = 31
N_ANCHORS          = 5

# Map indices to actual label names
CATEGORIES = "./dataset/my_new_dataset/categories.txt"

# Path to anchors for specific dataset (using K-mean clustering)
ANCHORS    = "./dataset/my_new_dataset/anchors.txt"
```

## Step 4: Fine-tune for your own dataset

* Training on your own datset 
```
python train.py  \
   --epochs        100 \
   --batch         8   \
   --learning_rate 0.00001 \
   --training_data /path/to/training_data.csv \
   --validation-data /path/to/testing_data.csv 
```

## Step 5: Evaluate your trained model on new dataset
```
python evaluate.py \
   ---weights yolov2.weights \
    ---csv_path ./dataset/my_new_dataset/testing_data.csv
```
