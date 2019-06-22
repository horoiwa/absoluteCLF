from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import SGD

from config import INCEPTION_WEIGHTS


def load_model(n_classes, weights=None):
    if weights:
        """重みが存在するときは途中からの学習なのでモデルを半解凍
           SGDを使用して細かく学習する
        """
        model = model_inceptionv3(n_classes, freeze=False)
        model.load_weights(weights)
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_crossentropy')
    else:
        """デフォルトはinceptionを凍結してtopのみがtrainable
        """
        model = model_inceptionv3(n_classes, freeze=True)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy')
    return model


def model_inceptionv3(n_classes, freeze=True):
    """ topなしresnet
        Note: Inception v3のデフォルト入力サイズは(299, 299)
    """
    input_tensor = Input(shape=(299, 299, 3))
    # create the base pre-trained model
    base_model = InceptionV3(weights=INCEPTION_WEIGHTS,
                             include_top=False,
                             input_tensor=input_tensor)

    #: top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    #: 最終モデル
    model = Model(inputs=base_model.input, outputs=predictions)

    # mixedレイヤーがブロックの区切り
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

    return model
