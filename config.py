""" Model configuration
"""
#: inception or xception
BASEMODEL = 'xception'

INCEPTION_WEIGHTS = 'src/inception_v3_notop.h5'
XCEPTION_WEIGHTS = 'src/xception_notop.h5'
RESNET_WEIGHTS = 'src/resnet50_notop.h5'

TARGET_SIZE = (512, 512)

BATCH_SIZE = 6

INITIAL_EPOCHS = 50
SECOND_EPOCHS = 200
THIRD_EPOCHS = 200
FINAL_EPOCHS = 200

EA_EPOCHS = 10

#: 基本的にはこの設定値なら影響がない
PCA_COLOR_RANGE = (-0.3, 0.3)

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

画像サイズを大きくするとサチりにくい
バッチ数を大きくしたり画像サイズを大きくするとメモリにのりきらないので注意
(TARGET_SIZE=(512, 512), batch_size=3)ぐらいが8GBの限界
"""
