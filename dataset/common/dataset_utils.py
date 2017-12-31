
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
    import tensorflow as tf
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]
