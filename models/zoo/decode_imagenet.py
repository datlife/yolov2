CLASS_INDEX_PATH = '../../dataset/imagenet.labels.list'
CLASS_NAMES_PATH = '../../dataset/imagenet.shortnames.list'


def decode_predictions(preds, top=5):
    """
    Decodes the prediction of an ImageNet model.
    """
    global CLASS_NAMES_PATH
    global CLASS_INDEX_PATH
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))

    with open(CLASS_NAMES_PATH) as f:
        LABELS = f.read().splitlines()

    with open(CLASS_INDEX_PATH) as f:
        IDS = f.read().splitlines()

    results = []

    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(IDS[i], LABELS[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results
