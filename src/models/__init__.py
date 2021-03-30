import os
import pickle


def load_model(model_class, folder):
    """ Check https://github.com/davidADSP/GDL_code/blob/master/utils/loaders.py#L133
    So far it only works with the VAE I have created
    """

    with open(os.path.join(folder, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = model_class(*params)

    model.load_weights(os.path.join(folder, 'weights/weights.h5'))

    return model
