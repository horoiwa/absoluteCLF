import os

from config import BATCH_SIZE, DATA_GEN_ARGS
from src.generator import customGenerator


def generator_test():
    """ 使う時はconfig.pyと同じディレクトリに移動して使用
    """
    os.makedirs('test_generator')
    trainGene = customGenerator(batch_size=BATCH_SIZE,
                                train_path='images',
                                image_folder='train',
                                aug_dict=DATA_GEN_ARGS,
                                save_to_dir='test_generator',
                                image_color_mode="rgb",)
    trainGene.__next__()


if __name__ == '__main__':
    generator_test()
