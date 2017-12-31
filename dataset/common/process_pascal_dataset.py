"""
Convert Pascal Annotations into txt format

Usage:
-----

1. Create training.txt using [training] set of Pascal 2012

python convert_pascal_to_txt.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \

2. Create training.txt using [training+val] set of Pascal 2012+2007

python convert_pascal_to_txt.py \
        --data_dir=/home/user/VOCdevkit \
        --set=trainval
        --year=merged \


Return
------
A text file for `create_custom_dataset.py` to process
    + 'training.txt'

**NOTE**
Most of the codes in this file was modified from TensorFlow Object Detection repo.
Instead of generating tf.record, we modified the file to generate .txt file.

Reference:
https://github.com/tensorflow/models/blob/master/object_detection/create_pascal_tf_record.py

"""
import os
from lxml import etree
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations', '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    if FLAGS.year not in YEARS:
        raise ValueError('year must be in : {}'.format(YEARS))

    data_dir = FLAGS.data_dir
    years = ['VOC2007', 'VOC2012']
    if FLAGS.year != 'merged':
        years = [FLAGS.year]

    output_file = os.path.join(data_dir, '{}_{}.txt'.format(FLAGS.set, FLAGS.year))
    with open(output_file, 'w') as txt_file:
        items = 0

        for year in years:
            annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)

            examples_path   = os.path.join(data_dir, year, 'ImageSets', 'Main', 'aeroplane_' + FLAGS.set + '.txt')
            examples_list   = read_examples_list(examples_path)

            for idx, example in enumerate(examples_list):
                path = os.path.join(annotations_dir, example + '.xml')
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = recursive_parse_xml_to_dict(xml)['annotation']

                img_path = os.path.join(data_dir, year, 'JPEGImages', data['filename'])
                for obj in data['object']:
                    difficult = bool(int(obj['difficult']))
                    if FLAGS.ignore_difficult_instances and difficult:
                        continue

                    xmin = float(obj['bndbox']['xmin'])
                    ymin = float(obj['bndbox']['ymin'])
                    xmax = float(obj['bndbox']['xmax'])
                    ymax = float(obj['bndbox']['ymax'])
                    classes_text = obj['name']

                    # Write item to text file here
                    items += 1
                    instance = "{},{},{},{},{},{}\n".format(img_path, xmin, ymin, xmax, ymax, classes_text)
                    txt_file.write(instance)

        print('Total instances', items)


def recursive_parse_xml_to_dict(xml):
    """
    Citation: https://github.com/tensorflow/core/blob/master/object_detection/utils/dataset_util.py

    Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def read_examples_list(path):
    """
    Citation: https://github.com/tensorflow/core/blob/master/object_detection/utils/dataset_util.py

    Read list of training or validation examples.

    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.

    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

    Args:
      path: absolute path to examples list file.

    Returns:
      list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


if __name__ == '__main__':
    tf.app.run()
