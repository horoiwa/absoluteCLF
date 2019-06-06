import glob
import os

from PIL import Image
import numpy as np

from src.util import folder_check, cleanup
from src.processing import preprocessing
from config import DATA_GEN_ARGS, CONFIGS


def main(prepare, train):
    categories = os.listdir('images/train')
    print("Detected classes:", categories)
    
    if prepare:
        print("Create Dataset")
        cleanup()
        folder_check()

        create_dataset(categories)

    if train:
        print("Run training")
        run_training()
        pass


def create_dataset(categories):
    os.makedirs('__dataset__')

    for category in categories:
        for datatype in ['train', 'valid']:
            p_images = glob.glob(f'images/{datatype}/{category}/*')
            os.makedirs(f'__dataset__/{datatype}/{category}')
            outdir = f'__dataset__/{datatype}/{category}'
            for p_image in p_images:
                preprocessing(p_image, outdir)


def run_training():
    pass




if __name__ == '__main__':
    main(prepare=False, train=True)
