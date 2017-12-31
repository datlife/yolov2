"""
Convert LISA csv annotation data set to txt format:

Usage:
-----
python convert_lisa_to_txt.py -p ../lisa/training

Return
------
It will create a text file:
    + 'training.txt'
    + 'categories.txt'

Next step
---------
Read create_custom_dataset.py

"""
import pandas as pd
import os
import glob
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description="Convert LISA Annotation into text file")
parser.add_argument('--path', '-p', type=str, help='path to training/testing lisa dataset')
parser.add_argument('--output_filename', '-o', type=str, default='training.txt', help='Output file name')

args = parser.parse_args()
lisa_path = args.path
out_fname = args.output_filename


def _main_():
    # Parse lisa and save into text file
    save_lisa_to_txt(lisa_path, save_file=out_fname)


def save_lisa_to_txt(csv_path, save_file='./training.txt'):
    image_paths, labels = load_lisa_data(csv_path)
    df = pd.concat([image_paths, labels], axis=1)
    data_file_path  = os.path.join(csv_path, save_file)
    categories_path = os.path.join(csv_path, 'categories.txt')

    # Save training data
    np.savetxt(data_file_path, df.values, delimiter=',', fmt='%s')
    print("A text file has been created at {}/{}".format(csv_path,  save_file))

    # Save labels in to categories.txt
    labels = df['Annotation tag'].unique()
    np.savetxt(categories_path, labels, delimiter=',', fmt='%s')
    print("A text file has been created at {}".format(categories_path))

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
    if len(csv_files) is 0:
        raise ValueError("No CSV were found in dataset.")
    else:
        print('Found %s CSV Files in dataset'%(len(csv_files)))

    print("Obtaining absolute path to dataset...")
    # Get absolute path to dataset
    path = os.path.abspath(path)
    print(path)
    X = []
    Y = []
    for fl in csv_files:
        df = pd.read_csv(fl, sep=';|,', engine='python')  # Convert csv to panda data frames
        X.append(path +'/'+df['Filename'])  # Convert to absolute path
        Y.append(df.loc[:, 'Annotation tag': 'Lower right corner Y'])  # Extract only labels [class, x1, y1, x2, y2]

    X = pd.concat(X)
    Y = pd.concat(Y)
    # Re-arrange order of pandas frame
    Y = Y[['Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y', 'Annotation tag']]
    return X, Y

if __name__ == "__main__":
    _main_()
