""" Model configuration
"""
INCEPTION_WEIGHTS = 'src/inception_v3_notop.h5'
RESNET_WEIGHTS = 'src/resnet50_notop.h5'

TARGET_SIZE = (512, 512)
CONFIGS = {"image_augmentation": True,
           "contrast_augmentation": True,
           "fine_tunining": True,
           "predict_proba": True}

BATCH_SIZE = 6

INITIAL_EPOCHS = 1
SECOND_EPOCHS = 1
FINAL_EPOCHS = 1

EA_EPOCHS = 10

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
段階的な訓練を導入
初期訓練はearly stopping導入するほうがよいかも
ヒストリーの可視化
初期訓練が甘いと破綻?
やはりガチャはある
imagedatagenは過学習の原因になる?
カテゴリ少ないときはDenseは最低限かつDropoutきつくするのがよい

train_genとvalid_genは同じaugmentationを行うべき
テスト時もこれを行うことでアンサンブル予測的になる？？

mixed_upとかいうやばいaugmentation
最後はsigmoidの方がぶれがでてよいかも？？

弱いESを入れたい

PCAコントラストaugmentationは結果を大きく改善した
"""
