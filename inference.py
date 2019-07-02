import os
import glob
import pandas as pd
import pathlib

from PIL import Image
import numpy as np
import keras.backend as K

from config import TARGET_SIZE, BASEMODEL
from src.models import load_model
from src.util import get_latestname


def main(folderpath):
    dirs = os.listdir('images/test/')
    weights = get_latestname("__checkpoints__/model_", 1)
    n_classes = len(dirs)
    model = load_model(n_classes=n_classes, weights=weights,
                       freeze='inference', basemodel=BASEMODEL)

    df = pd.DataFrame()
    images_path = glob.glob(folderpath+'/*')
    for image_path in images_path:
        name = os.path.basename(image_path)
        image = prep_image(image_path)
        pred, uncert = prediction(model, n_classes, image)

        df[name+"_pred"] = pred
        df[name+"_var"] = uncert

    df.index = dirs
    df.to_csv('inference.csv')


def prediction(model, n_classes, image):
    image = image.reshape(TARGET_SIZE[0],
                          TARGET_SIZE[1], 3).reshape(1, TARGET_SIZE[0],
                                                     TARGET_SIZE[1], 3)
    f = K.function([model.layers[0].input, K.learning_phase()],
                   [model.layers[-1].output])

    predictioin, uncertainity = predict_with_uncertainty(f, n_classes, image)
    return(predictioin, uncertainity)


def predict_with_uncertainty(f, n_classes, x, n_iter=10):
    result = np.zeros((n_iter,) + (1, n_classes))

    for iter in range(n_iter):
        result[iter] = f([x, 1])

    prediction = result.mean(axis=0)[0]
    uncertainty = result.var(axis=0)[0]
    return prediction, uncertainty


def prep_image(image_path):
    image = Image.open(image_path)
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    image = np.array(image) / 255

    return image


if __name__ == '__main__':
    folderpath = 'inference'
    main(folderpath)
