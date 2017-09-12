# Train on custom data set
----------------------------

## Step 1: Prepares training data

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
        
        
## Step 2: Generate dataset for training

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


## Step 3: Update parameters in `cfg.py`

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

## Step 4: Fine-tune for your own dataset

* Training on your own datset
```
python train.py  \
   --epochs        100
   --batch         8
   --learning_rate 0.00001
   --training_data your_own_data.txt 
```

## Step 5: Evaluate your finetuned model
```
python evaluate.py 
---weights yolov2.weights
---csv_path ./dataset/my_new_dataset/testing_data.csv
```
