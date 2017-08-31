from keras.models import load_model, Model


def densenet(include_top=True):
    base = load_model('./weights/densenet201.h5')

    if include_top:
        model = Model(base.inputs, base.outputs)
    else:
        model = Model(base.inputs, base.layers[-4].output)

    return model
