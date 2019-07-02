import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from config import PCA_COLOR_RANGE


def customGenerator(batch_size, train_path, image_folder, aug_dict,
                    image_color_mode="rgb", shuffle=True,
                    image_save_prefix="image",
                    save_to_dir=None, target_size=None,
                    inference=False):
    #: test mode
    if not aug_dict:
        aug_dict = dict(rescale=1./255)

    image_folder_path = os.path.join(train_path, image_folder)
    classes = os.listdir(image_folder_path)

    custom_datagen = ImageDataGenerator(**aug_dict)
    custom_generator = custom_datagen.flow_from_directory(
        image_folder_path,
        classes=classes,
        class_mode="categorical",
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        shuffle=shuffle)

    for images, labels in custom_generator:
        #: 任意の処理を挟むことが可能
        for i in range(images.shape[0]):
            image = images[i, :, :, :]
            if not inference:
                image = image * 255
                image = pca_color_augmentation_modify(image)
                image = image / 255
            images[i, :, :, :] = image

        yield (images, labels)


def pca_color_augmentation_modify(image_array_input):
    """
        RGBカラー画像限定
        コピぺ：https://qiita.com/koshian2/items/78de8ccd09dd2998ddfc
    """
    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3

    img = image_array_input.reshape(-1, 3).astype(np.float32)
    # 分散を計算
    ch_var = np.var(img, axis=0)
    # 分散の合計が3になるようにスケーリング
    scaling_factor = np.sqrt(3.0 / sum(ch_var))
    # 平均で引いてスケーリング
    img = (img - np.mean(img, axis=0)) * scaling_factor

    cov = np.cov(img, rowvar=False)
    lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)

    while True:
        rand = np.random.randn(3) * 0.1
        if np.all(rand > PCA_COLOR_RANGE[0]):
            if np.all(rand < PCA_COLOR_RANGE[1]):
                break

    delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
    return img_out


def DummyGenerator(batch_size, train_path, image_folder, aug_dict,
                   image_color_mode="rgb", shuffle=True,
                   image_save_prefix="image",
                   save_to_dir=None, target_size=None,
                   inference=False):
    #: test mode
    if not aug_dict:
        aug_dict = dict(rescale=1./255)

    image_folder_path = os.path.join(train_path, image_folder)
    classes = os.listdir(image_folder_path)

    custom_datagen = ImageDataGenerator(**aug_dict)
    custom_generator = custom_datagen.flow_from_directory(
        image_folder_path,
        classes=classes,
        class_mode="categorical",
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        shuffle=shuffle)

    return custom_generator
