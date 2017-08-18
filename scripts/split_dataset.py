"""
Split dataset into training and testing
"""
import os
import csv
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

parser = ArgumentParser(description="Split data")
parser.add_argument('-p', '--path', help="Path to training text file ", type=str, default='./dataset/lisa/training.txt')


def split_data(dataset, training='training_data.csv', testing='testing_data.csv', ratio=0.1):
    with open(dataset) as txt_file:
        x = txt_file.read().splitlines()[1:]
        x_train, x_test = train_test_split(x, test_size=ratio)

    dataset_path = os.path.dirname(dataset)
    training_path = os.path.join(dataset_path, training)
    testing_path = os.path.join(dataset_path, testing)
    with open(training_path, 'w') as txt:

        writer = csv.writer(txt, delimiter=',', )
        writer.writerow(["Filename", "x1", "y1", "x2", "y2", "annotation tag"])
        for item in x_train:
            item = item.split(',')
            writer.writerow(item)

    with open(testing_path, 'w') as txt:
        writer = csv.writer(txt, delimiter=',')
        writer.writerow(["Filename", "x1", "y1", "x2", "y2", "annotation tag"])
        for item in x_test:
            item = item.split(',')
            writer.writerow(item)

    print(len(x_train))
    print(len(x_test))
    return training, testing


if __name__ == "__main__":
    args = parser.parse_args()
    PATH = args.path
    training, testing = split_data(PATH)
