
CONFIGS = {"image_augmentation": True,
           "contrast_augmentation": True,
           "fine_tunining": True,
           "predict_proba": True}

BATCH_SIZE = 10
EPOCHS = 10

DATA_GEN_ARGS = dict(rescale=1./255,
                     rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     brightness_range=[0.8, 1.0],
                     shear_range=0.05,
                     zoom_range=0.05,
                     vertical_flip=True,
                     horizontal_flip=True,
                     fill_mode='nearest')
