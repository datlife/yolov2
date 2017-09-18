from keras.models import load_model, Model


def densenet(include_top=True, model_path='./weights/feature_extractors/densenet201.h5'):
    base = load_model(model_path)

    if include_top:
        model = Model(base.inputs, base.outputs)
    else:
        model = Model(base.inputs, base.layers[-4].output)

    return model
