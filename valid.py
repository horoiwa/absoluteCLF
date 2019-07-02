import os
import glob
import pandas as pd
import pathlib
import shutil
import sys

from PIL import Image
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import lime
from lime import lime_image

from src.models import load_model
from src.util import get_latestname
from config import TARGET_SIZE, BASEMODEL
NUM_SAMPLES = 100


def main(mode='simple'):
    homedir = '__checkpoints__/valid'
    if os.path.exists(homedir):
        shutil.rmtree(homedir)
    os.makedirs(homedir)

    categories = os.listdir('images/test')
    indices = range(len(categories))
    labels = {idx: label for idx, label in zip(indices, categories)}

    for category in categories:
        category_dir = os.path.join(homedir, category)
        os.makedirs(category_dir)
        debug_model(category, category_dir, labels, mode)


def debug_model(category, category_dir, labels, mode):
    print("Start:", category)
    images_path = glob.glob(os.path.join('images', 'test', category, '*.jpg'))

    n_classes = os.listdir('images/train')
    trained_weight = get_latestname("__checkpoints__/model_", 1)
    model = load_model(len(n_classes), trained_weight,
                       freeze='inference', basemodel=BASEMODEL)

    for image_path in images_path:
        image = prep_image(image_path)
        name = pathlib.Path(image_path).name

        prediction = model.predict(image)
        predicted_label = labels[np.argmax(prediction)]
        true_label = category

        if predicted_label == true_label:
            continue

        image = image.reshape(TARGET_SIZE[0], TARGET_SIZE[1], 3)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image, model.predict,
                                                 top_labels=len(n_classes),
                                                 hide_color=0,
                                                 num_samples=NUM_SAMPLES)

        if mode == 'RedGreen':
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=10,
                hide_rest=False)

        elif mode == 'simple':
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False)

        image = mark_boundaries(temp * 0.7 + 0.3, mask) * 255
        image = Image.fromarray(np.uint8(image))
        image.save(os.path.join(category_dir, "["+predicted_label+"]_"+name))


def prep_image(image_path):
    image = Image.open(image_path)
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    image = np.array(image) / 255
    image = image.reshape(TARGET_SIZE[0],
                          TARGET_SIZE[1], 3).reshape(1,
                                                     TARGET_SIZE[0],
                                                     TARGET_SIZE[1], 3)

    return image


if __name__ == '__main__':
    try:
        mode = sys.argv[1]
    except IndexError:
        mode = 'RedGreen'

    main(mode)
