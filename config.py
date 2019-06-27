INCEPTION_WEIGHTS = 'src/inception_v3_notop.h5'
RESNET_WEIGHTS = 'src/resnet50_notop.h5'

CONFIGS = {"image_augmentation": True,
           "contrast_augmentation": True,
           "fine_tunining": True,
           "predict_proba": True}

BATCH_SIZE = 6

INITIAL_EPOCHS = 10
SECOND_EPOCHS = 50
FINAL_EPOCHS = 50

DATA_GEN_ARGS = dict(rescale=1./255,
                     shear_range=0.2,
                     zoom_range=0.2,
                     vertical_flip=False,
                     horizontal_flip=True,
                     fill_mode='nearest')

DATA_GEN_ARGS_MIN = dict(rescale=1./255,
                         vertical_flip=False,
                         horizontal_flip=True,
                         fill_mode='nearest')

DATA_GEN_DEFAULT = dict(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.,
    height_shift_range=0.,
    brightness_range=[0.9, 1.0],
    shear_range=0.02,
    zoom_range=0.02,
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

"""
