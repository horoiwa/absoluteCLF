import os
import shutil

import numpy as np
from PIL import Image

from config import BATCH_SIZE, DATA_GEN_DEFAULT, TARGET_SIZE
from src.generator import customGenerator


def generator_test():
    """ 使う時はconfig.pyと同じディレクトリに移動して使用
    """
    dirname = 'config_test'
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    os.makedirs(dirname)
    trainGene = customGenerator(batch_size=50,
                                train_path='images',
                                image_folder='train',
                                aug_dict=DATA_GEN_DEFAULT,
                                save_to_dir=None,
                                target_size=TARGET_SIZE,
                                image_color_mode="rgb",
                                inference=False)

    images, labels = trainGene.__next__()
    for i in range(images.shape[0]):
        image = images[i, :, :, :] * 255
        image = Image.fromarray((np.uint8(image)))
        image.save(f'{dirname}/{i}.png')


if __name__ == '__main__':
    generator_test()
