import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator


def customGenerator(batch_size, train_path, image_folder, aug_dict,
                    image_color_mode="grayscale",
                    image_save_prefix="image",
                    save_to_dir=None, target_size=(299, 299),
                    seed=np.random.randint(1, 9999)):
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
        shuffle=True,
        seed=seed)

    for images in custom_generator:
        #: 任意の処理を挟むことが可能
        yield images
