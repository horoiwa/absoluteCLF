""" Model configuration
"""
INCEPTION_WEIGHTS = 'src/inception_v3_notop.h5'
RESNET_WEIGHTS = 'src/resnet50_notop.h5'

TARGET_SIZE = (768, 768)

CONFIGS = {"image_augmentation": True,
           "contrast_augmentation": True,
           "fine_tunining": True,
           "predict_proba": True}

BATCH_SIZE = 3

INITIAL_EPOCHS = 30
SECOND_EPOCHS = 150
FINAL_EPOCHS = 150

EA_EPOCHS = 5

PCA_MIN = 0
DATA_GEN_DEFAULT = dict(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.0,
    zoom_range=0.1,
    vertical_flip=False,
    horizontal_flip=True,
    cval=0,
    fill_mode='constant')

DATA_GEN_ARGS_MIN = dict(rescale=1./255,
                         vertical_flip=False,
                         horizontal_flip=True,
                         fill_mode='nearest')
"""
Note:
カテゴリ少ないときはDenseは最低限かつDropoutきつくするのがよい
fill_modeを黒にすると結果が改善
入力画像サイズを大きくすると結果が大きく改善した
PCAコントラストaugmentationは結果を大きく改善した
"""
