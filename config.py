INCEPTION_WEIGHTS = 'src/inception_v3_notop.h5'
RESNET_WEIGHTS = 'src/resnet50_notop.h5'

CONFIGS = {"image_augmentation": True,
           "contrast_augmentation": True,
           "fine_tunining": True,
           "predict_proba": True}

BATCH_SIZE = 16

INITIAL_EPOCHS = 1
SECOND_EPOCHS = 1
FINAL_EPOCHS = 1

DATA_GEN_ARGS = dict(rescale=1./255,
                     rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     brightness_range=[0.9, 1.0],
                     shear_range=0.05,
                     zoom_range=0.05,
                     vertical_flip=False,
                     horizontal_flip=False,
                     fill_mode='nearest')

DATA_GEN_ARGS_MIN = dict(rescale=1./255,
                         vertical_flip=True,
                         horizontal_flip=True,
                         fill_mode='nearest')

DATA_GEN_DEFAULT = dict(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.9, 1.0],
    shear_range=0.05,
    zoom_range=0.05,
    vertical_flip=False,
    horizontal_flip=False,
    fill_mode='nearest')

"""
Note:
段階的な訓練を導入
初期訓練はearly stopping導入するほうがよいかも
ヒストリーの可視化
初期訓練が甘いと破綻?
やはりガチャはある
imagedatagenは過学習の原因になる
"""
