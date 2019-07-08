import os
import glob
import pandas as pd
import pathlib

import numpy as np
from PIL import Image
import click

from src.models import load_model
from src.util import get_latestname
from config import TARGET_SIZE, BASEMODEL


@click.command()
@click.option('--folder', '-f', required=True, type=click.Path(exists=True))
@click.option('--out', '-o', default='predict.csv')
def main(folder, out):
    """起動はこのフォルダ内から行う
       フォルダに格納された画像に対して予測を行う
    """
    dirs = os.listdir('images/test/')
    weights = get_latestname("__checkpoints__/model_", 1)
    n_classes = len(dirs)
    model = load_model(n_classes=n_classes, weights=weights,
                       freeze='inference', basemodel=BASEMODEL)

    df = pd.DataFrame()
    images_path = glob.glob(os.path.join(folder, '*.jpg'))
    print("-----"*4)
    print(f"Detected {len(images_path)} images")
    if not len(images_path):
        print("No images detected")
        return

    for image_path in images_path:
        name = os.path.basename(image_path)
        image = prep_image(image_path)
        image = image.reshape(TARGET_SIZE[0],
                              TARGET_SIZE[1], 3).reshape(1, TARGET_SIZE[0],
                                                         TARGET_SIZE[1], 3)
        pred = model.predict(image)
        print(name, pred)
        df[name] = pred[0]

    df.index = dirs
    df = df.T
    df.to_csv(out)


def prep_image(image_path):
    image = Image.open(image_path)
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    image = np.array(image) / 255

    return image


if __name__ == '__main__':
    main()
