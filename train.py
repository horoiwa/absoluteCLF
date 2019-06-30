import glob
import os
import shutil
import random

import matplotlib.pyplot as plt

from config import (BATCH_SIZE, CONFIGS, DATA_GEN_ARGS_MIN,
                    DATA_GEN_DEFAULT, EA_EPOCHS, FINAL_EPOCHS,
                    INITIAL_EPOCHS, SECOND_EPOCHS, TARGET_SIZE)
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from src.generator import customGenerator
from src.models import load_model
from src.processing import preprocessing
from src.util import cleanup, folder_check, get_latestname, get_uniquename


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

    num_images_classes = []
    for category in categories:
        for datatype in ['train']:
            p_images = glob.glob(f'images/{datatype}/{category}/*')
            num_images_classes.append(len(p_images))

    num_images_max = max(num_images_classes)
    for category in categories:
        for datatype in ['train']:
            p_images = glob.glob(f'images/{datatype}/{category}/*')
            num_images = len(p_images)
            os.makedirs(f'__dataset__/{datatype}/{category}')
            outdir = f'__dataset__/{datatype}/{category}'
            for p_image in p_images:
                preprocessing(p_image, outdir)
            if num_images_max - num_images > 0:
                for i in range(num_images_max - num_images):
                    p_image = random.choice(p_images)
                    preprocessing(p_image, outdir, rename=True)

        for datatype in ['valid', 'test']:
            p_images = glob.glob(f'images/{datatype}/{category}/*')
            num_images = len(p_images)
            os.makedirs(f'__dataset__/{datatype}/{category}')
            outdir = f'__dataset__/{datatype}/{category}'
            for p_image in p_images:
                preprocessing(p_image, outdir)


def run_training():
    acc_train = []
    acc_val = []

    trainGene = customGenerator(batch_size=BATCH_SIZE,
                                train_path='__dataset__',
                                image_folder='train',
                                aug_dict=DATA_GEN_DEFAULT,
                                save_to_dir=None,
                                target_size=TARGET_SIZE,
                                image_color_mode="rgb",)

    validGene = customGenerator(batch_size=BATCH_SIZE,
                                train_path='__dataset__',
                                image_folder='valid',
                                aug_dict=None,
                                save_to_dir=None,
                                target_size=TARGET_SIZE,
                                image_color_mode="rgb",)

    if os.path.exists("__checkpoints__"):
        shutil.rmtree("__checkpoints__")
    os.makedirs("__checkpoints__")

    hdfname = get_uniquename("__checkpoints__/model_", 1)
    model_checkpoint = ModelCheckpoint('{}.hdf5'.format(hdfname),
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=EA_EPOCHS, verbose=0, mode='auto')

    callbacks = [early_stopping, model_checkpoint]

    n_classes = len(os.listdir('__dataset__/train/'))
    n_train_images = len(glob.glob('__dataset__/train/*/*'))
    n_valid_images = len(glob.glob('__dataset__/valid/*/*'))

    print("ベースモデル凍結：訓練開始")

    trained_weight = get_latestname("__checkpoints__/model_", 1)
    model = load_model(n_classes, trained_weight, freeze='initial')

    history = model.fit_generator(
        trainGene,
        steps_per_epoch=n_train_images // BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        validation_data=validGene,
        validation_steps=n_valid_images // BATCH_SIZE,
        callbacks=callbacks)

    acc_train = acc_train + list(history.history['acc'])
    acc_val = acc_val + list(history.history['val_acc'])

    print("初期訓練の終了：モデルのリロードを開始")
    trained_weight = get_latestname("__checkpoints__/model_", 1)
    print("検出したモデル：", trained_weight)
    model = load_model(n_classes, weights=trained_weight, freeze='second')
    print("2つのinceptionブロックを解凍：訓練再開")

    history = model.fit_generator(
        trainGene,
        steps_per_epoch=n_train_images // BATCH_SIZE,
        epochs=SECOND_EPOCHS,
        validation_data=validGene,
        validation_steps=n_valid_images // BATCH_SIZE,
        callbacks=callbacks)

    acc_train = acc_train + list(history.history['acc'])
    acc_val = acc_val + list(history.history['val_acc'])

    print("第二次訓練の終了：モデルのリロードを開始")
    trained_weight = get_latestname("__checkpoints__/model_", 1)
    print("検出したモデル：", trained_weight)
    model = load_model(n_classes, weights=trained_weight, freeze='final')
    print("4つのinceptionブロックを解凍：訓練再開")

    trainGene = customGenerator(batch_size=BATCH_SIZE,
                                train_path='__dataset__',
                                image_folder='train',
                                aug_dict=DATA_GEN_DEFAULT,
                                save_to_dir=None,
                                target_size=TARGET_SIZE,
                                image_color_mode="rgb",)

    validGene = customGenerator(batch_size=BATCH_SIZE,
                                train_path='__dataset__',
                                image_folder='valid',
                                aug_dict=None,
                                save_to_dir=None,
                                target_size=TARGET_SIZE,
                                image_color_mode="rgb",)

    history = model.fit_generator(
        trainGene,
        steps_per_epoch=n_train_images // BATCH_SIZE,
        epochs=FINAL_EPOCHS,
        validation_data=validGene,
        validation_steps=n_valid_images // BATCH_SIZE,
        callbacks=callbacks)

    acc_train = acc_train + list(history.history['acc'])
    acc_val = acc_val + list(history.history['val_acc'])

    print("訓練の正常終了を確認")
    print("acc train:", acc_train)
    print("acc validation:", acc_val)
    epochs = range(1, len(acc_train) + 1)

    plt.plot(epochs, acc_train, label='train')
    plt.plot(epochs, acc_val, label='valid')
    plt.legend()
    plt.savefig('__checkpoints__/training_history.png')

    shutil.copy('config.py', '__checkpoints__/')


if __name__ == '__main__':
    main(prepare=False, train=True)
