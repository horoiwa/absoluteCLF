import os
import shutil


def cleanup(mode='all'):
    folders = ['']

    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def folder_check():
    if 'train' not in os.listdir('images'):
        raise Exception("Error: train folder not exists")
    elif 'valid' not in os.listdir('images'):
        raise Exception("Error: valid folder not exists")
    elif 'test' not in os.listdir('images'):
        raise Exception("Error: test folder not exists")

    dirs_train = os.listdir('images/train')
    dirs_valid = os.listdir('images/valid')

    if dirs_train != dirs_valid:
        raise Exception("Error: train and valid not consisted")


if __name__ == '__main__':
    folder_check()