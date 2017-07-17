"""
Convert LISA csv annotation dataset to txt format as following:

Text file:
---------
abs_image_path, x1, y1, x2, y2, label

Usage:
-----

python process_lisa.py -p absolute/path/to/lisa/training

eg:
python process_lisa.py -p /home/ubuntu/dataset/training


It will create a text file training.txt for training on YOLOv2
"""
import pandas as pd
import os
import glob
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description="Convert LISA Annotation into text file")
parser.add_argument('--lisa_path', '-p', type=str, help='path to training/testing lisa dataset')

def _main_():
    args = parser.parse_args()
    lisa_path = args.lisa_path    
    # Parse lisa and save into text file
    save_lisa_to_txt(lisa_path,save_file='./training.txt')
    print("A text file has been created.")

def save_lisa_to_txt(csv_path, save_file='./training.txt'):
    image_paths, labels = load_lisa_data(csv_path)
    df = pd.concat([image_paths, labels], axis=1)
    np.savetxt("training.txt", df.values, delimiter=',', fmt='%s')
    return image_paths, labels


def load_lisa_data(path=None):
    """
    Load LISA training data into pandas frames from a given path 

    Assumption:
        -  Path is contained an annotation ".csv" file 

    :param path: path to Data set
    :return: two lists:
        X : path_to_training images
        y : bounding box, labels
    """
    if path is None:
        raise ValueError("Please setup correct path to to data set")

    if not os.path.isdir(path):
        raise IOError("Path does not exist")

    # Get all .csv file(s) in given path (including sub-directories)
    csv_files = glob.glob(path + "*.csv")
    X = []
    Y = []
    for fl in csv_files:
        df = pd.read_csv(fl, sep=';|,', engine='python')  # Convert csv to panda data frames
        X.append(path + df['Filename'])  # Convert to absolute path
        Y.append(df.loc[:, 'Annotation tag': 'Lower right corner Y'])  # Extract only labels [class, x1, y1, x2, y2]

    X = pd.concat(X)
    Y = pd.concat(Y)
    # Re-arrange order of pandas frame
    Y = Y[['Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y', 'Annotation tag']]

    return X, Y

if __name__ == "__main__":
    _main_()
