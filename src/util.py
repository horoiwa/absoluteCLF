import os
import shutil


def cleanup(mode='all'):
    folders = ['__dataset__', '__checkpoints__', 'image_test',
               'config_test']

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


def get_uniquename(name, n):
    uniquename = name + str(n) + '.hdf5'
    if os.path.exists(uniquename):
        uniquename = get_uniquename(name, n+1)
    else:
        return name + str(n)


def get_latestname(name, n):
    currentname = name + str(n) + '.hdf5'
    nextname = name + str(n+1) + '.hdf5'
    if os.path.exists(nextname):
        get_latestname(name, n+1)
    elif os.path.exists(currentname):
        return currentname
    else:
        return None


def make_defaultfolder():
    os.makedirs('images/test')
    os.makedirs('images/train')
    os.makedirs('images/valid')


if __name__ == '__main__':
    folder_check()
