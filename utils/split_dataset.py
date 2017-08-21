"""
Split dataset into training and testing
"""
from sklearn.model_selection import train_test_split
import csv


def split_data(dataset, training='training_data.csv', testing='testing_data.csv', ratio=0.1):
    with open(dataset) as txt_file:
        x = txt_file.read().splitlines()[1:]
        x_train, x_test = train_test_split(x, test_size=ratio)

    with open(training, 'w') as txt:

        writer = csv.writer(txt, delimiter=',', )
        writer.writerow(["Filename", "x1", "y1", "x2", "y2", "annotation tag"])
        for item in x_train:
            item = item.split(',')
            writer.writerow(item)

    with open(testing, 'w') as txt:
        writer = csv.writer(txt, delimiter=',')
        writer.writerow(["Filename", "annotation tag", "x1", "y1", "x2", "y2", ])
        for item in x_test:
            item = item.split(',')
            item[1], item[-1] = item[-1], item[1]
            writer.writerow(item)

    print(len(x_train))
    print(len(x_test))
    return training, testing

if __name__ == "__main__":
    training, testing = split_data('../scripts/training_cutai.txt')
