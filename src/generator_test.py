import os
import shutil

import numpy as np
from PIL import Image

from config import BATCH_SIZE, DATA_GEN_ARGS
from src.generator import customGenerator


def generator_test():
    """ 使う時はconfig.pyと同じディレクトリに移動して使用
    """
    dirname = 'image_test'
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    os.makedirs(dirname)
    trainGene = customGenerator(batch_size=50,
                                train_path='images',
                                image_folder='train',
                                aug_dict=DATA_GEN_ARGS,
                                save_to_dir=None,
                                image_color_mode="rgb",
                                inference=False)

    images, labels = trainGene.__next__()
    for i in range(images.shape[0]):
        image = images[i, :, :, :] * 255
        image = Image.fromarray((np.uint8(image)))
        image.save(f'{dirname}/{i}.png')



if __name__ == '__main__':
    generator_test()
