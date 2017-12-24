"""

"""
from __future__ import division
import os
import argparse
import matplotlib.pyplot as plt
import sys
import re
from copy import deepcopy
from collections import namedtuple

MatchStats = namedtuple("MatchStats", ["numAnnotations", "tpCount", "fpCount", "precision", "recall", "widthsFound"])


def computeMatchStatistics(annotations, detections, pascal=0.5, sizeMinimum=None):
    numAnnotations = len(annotations)
    workAnnotations = deepcopy(annotations)

    # Remove any annotations that are too small:
    for annotation in workAnnotations:
        annFields = annotation.split(';')
        if sizeMinimum != None and (
                            int(annFields[4]) - int(annFields[2]) < sizeMinimum[0] or int(annFields[5]) - int(
                    annFields[3]) <
                    sizeMinimum[1]):
            workAnnotations.remove(annotation)
            numAnnotations -= 1
    fpCount = 0
    tpCount = 0
    id = 0
    widthsFound = []
    falsePositives = []
    for detection in detections:
        fields = detection.split(',')
        potentialAnnotations = [line for line in workAnnotations if fields[0] in line]
        match = False
        for annotation in potentialAnnotations:
            annFields = annotation.split(',')

            # Compute intersection : IoU
            left = max(int(float(annFields[1])), int(float(fields[1])))
            right = min(int(float(annFields[3])), int(float(fields[3])))
            top = max(int(float(annFields[2])), int(float(fields[2])))
            bottom = min(int(float(annFields[4])), int(float(fields[4])))
            if left < right and top < bottom:
                intersectionArea = (right - left) * (bottom - top)
            else:
                intersectionArea = 0

            # Compute union as the combined area of the two rectangles minus the intersection
            unionArea = (int(float(annFields[3])) - int(float(annFields[1]))) * (
                int(float(annFields[4])) - int(float(annFields[2]))) + \
                        (int(float(fields[3])) - int(float(fields[1]))) * (
                            int(float(fields[4])) - int(float(fields[2]))) - \
                        intersectionArea

            # Compute Pascal measure
            pascalMeasure = intersectionArea / unionArea
            match = (pascalMeasure > pascal)
            if match:
                workAnnotations.remove(annotation)
                break

        if match:
            tpCount += 1
            widthsFound.append(int(float(fields[3])) - int(float(fields[1])))
        else:
            fpCount += 1
            falsePositives.append(fields)
        id += 1

    return [MatchStats(numAnnotations, tpCount, fpCount, tpCount / (tpCount + fpCount), tpCount / numAnnotations, []),
            # widthsFound),
            falsePositives,
            workAnnotations]


def printDetailedStats(falsePositives, falseNegatives):
    for fp in falsePositives:
        print("False positive in: %s" % fp[0])


def printFalseNegatives(falseNegatives, header):
    sys.stdout.write(header)
    sys.stdout.write('\n'.join([x.strip() for x in falseNegatives]))


def main(args):
    if not os.path.isfile(args.detectionPath):
        print("Error: The given detection file does not exist.")
        exit()
    detectionFile = open(os.path.abspath(args.detectionPath), 'r')

    if not os.path.isfile(args.truthPath):
        print("Error: The given annotation file does not exist.")
        exit()
    annotationFile = open(os.path.abspath(args.truthPath), 'r')

    if not (0 < args.pascal <= 1.0):
        print("Error: The Pascal overlap criterion must be > 0 and <= 1.0")
        exit()

    sizeMinimum = None
    if args.sizeMinimum != None and not re.match('[0-9]+x[0-9]+', args.sizeMinimum):
        print(
            "Error: The size must be in the format 20x20, where the numbers can be any integer, regex match [0-9]+x[0-9]+. First number is width.")
        exit()
    elif args.sizeMinimum != None:
        sizeMinimum = [int(args.sizeMinimum.partition('x')[0]),
                       int(args.sizeMinimum.partition('x')[2])]

    header = annotationFile.readline()  # Discard the header-line.

    detections = detectionFile.readlines()
    annotations = annotationFile.readlines()
    # from cfg import HIER_TREE
    # for line in annotations:
    #     line[1] = HIER_TREE.get_parent(line[1])
    # print(annotations)
    statistics, falsePositives, falseNegatives = computeMatchStatistics(annotations, detections, args.pascal,
                                                                        sizeMinimum)

    if args.printOnlyFalseNegatives:
        printFalseNegatives(falseNegatives, header)
        exit()
    if args.verbose:
        printDetailedStats(falsePositives, falseNegatives)

    print('------')
    print('Total Number of annotations:\t%d' % statistics.numAnnotations)
    print('------')
    print('Testing with a Pascal overlap measure of: %0.2f' % args.pascal)
    print("True positives:\t\t%d" % statistics.tpCount)
    print("False positives:\t%d" % statistics.fpCount)
    print("False negatives (miss):\t%d" % (statistics.numAnnotations - statistics.tpCount))
    print('------')
    print("Precision:\t\t%0.4f" % statistics.precision)
    print("Recall:\t\t\t%0.4f" % statistics.recall)

    if args.widthHistogram and statistics.tpCount > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(statistics.widthsFound, range=(10, 110), bins=20)
        ax.set_title('Histogram over detected bounding box widths in %s' % args.detectionPath)
        plt.gca().set_xlabel("Sign widths in pixels")
        plt.gca().set_ylabel("Number of detections")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate stats on detection performance, given a detection file and a ground truth file.')
    parser.add_argument('detectionPath', metavar='detections.csv', type=str,
                        help='The path to the csv-file containing the detections. Each line formatted as filenameNoPath;upperLeftX;upperLeftY;lowerRightX;lowerRightY. No header line.')
    parser.add_argument('truthPath', metavar='annotations.csv', type=str,
                        help='The path to the csv-file containing ground truth annotations.')
    parser.add_argument('-p', '--pascal', type=float, default=0.5, help='Define Pascal overlap fraction.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print false negatives along with the final summary.')
    parser.add_argument('-n', '--printOnlyFalseNegatives', action='store_true',
                        help='Print just the misses in an annotation file format. Useful for boosting.')
    parser.add_argument('-s', '--sizeMinimum', metavar='20x20', type=str,
                        help='Disregard any annotation smaller than the specified size. First number is width.')
    parser.add_argument('-w', '--widthHistogram', action='store_true',
                        help='Show a histogram of the widths of true positives.')
    args = parser.parse_args()

    main(args)
