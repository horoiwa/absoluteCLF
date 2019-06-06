
CONFIGS = {"image_augmentation": True,
           "contrast_augmentation": True,
           "fine_tunining": True,
           "predict_proba": True}

DATA_GEN_ARGS = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')