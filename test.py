import os
import glob
import pandas as pd
import pathlib

from PIL import Image
import numpy as np
import keras.backend as K

from src.models import load_model
from src.util import get_latestname
from src.generator import DummyGenerator
from config import TARGET_SIZE, BASEMODEL


def main(folder=None):
    if folder:
        inference(folder)
    else:
        inference_testdata()


def inference(folder):
    print(folder)


def inference_testdata():
    dirs = os.listdir('images/test/')
    weights = get_latestname("__checkpoints__/model_", 1)
    model = load_model(n_classes=len(dirs), weights=weights,
                       freeze='inference', basemodel=BASEMODEL)

    testGene = DummyGenerator(batch_size=1,
                              train_path='__dataset__',
                              image_folder='test',
                              aug_dict=None,
                              save_to_dir=None,
                              shuffle=False,
                              target_size=TARGET_SIZE,
                              image_color_mode="rgb",)

    filenames = testGene.filenames
    category_names = [pathlib.Path(fname).parts[-2] for fname in filenames]
    nb_samples = len(filenames)
    if not nb_samples:
        print("No images found")
        return

    predict = model.predict_generator(testGene, steps=nb_samples)

    columns = list(testGene.class_indices.keys())
    df_predict = pd.DataFrame(predict, columns=columns)
    df_predict['Pred'] = df_predict.idxmax(1)
    df_predict['True'] = category_names
    df_predict['Score'] = (df_predict['Pred'] == df_predict['True']).apply(int)
    df_predict.index = filenames
    df_predict.to_csv('__checkpoints__/test_result.csv')

    print("--"*20)
    print("")
    print("Total Score:", df_predict['Score'].mean())
    for label in set(list(df_predict["True"])):
        score = df_predict.loc[df_predict['True'] == label, "Score"].mean()
        print(f"{label} Score:", score)

    print("")
    print("--"*20)


if __name__ == '__main__':
    main()
