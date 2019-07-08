from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import SGD

from config import INCEPTION_WEIGHTS, XCEPTION_WEIGHTS, TARGET_SIZE


def load_model(n_classes, weights=None, freeze=None, basemodel=None):
    if basemodel == 'inception':
        print("Load inception_v3 model")
        model = load_inception(n_classes, weights=weights, freeze=freeze)
        return model
    elif basemodel == 'xception':
        print("Load xception model")
        model = load_xception(n_classes, weights=weights, freeze=freeze)
        return model
    else:
        raise Exception("NotImplementedModel")


def load_inception(n_classes, weights=None, freeze=None):
    if freeze == 'second' or freeze == 'third' or freeze == 'final':
        """重みが存在するときは途中からの学習なのでモデルを半解凍
           SGDを使用して細かく学習する
        """
        model = model_inceptionv3(n_classes, freeze=freeze)
        model.load_weights(weights)
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    elif freeze == 'initial':
        """デフォルトはinceptionを凍結してtopのみがtrainable
        """
        model = model_inceptionv3(n_classes, freeze=freeze)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
    elif freeze == 'inference':
        model = model_inceptionv3(n_classes, freeze='final')
        assert weights, "Error, No model found!!"
        model.load_weights(weights)
    else:
        print(freeze)
        raise Exception("freeze must not be None")

    return model


def model_inceptionv3(n_classes, freeze=None):
    """ inception_v3
        Note: Inception v3のデフォルト入力サイズは(TARGET_SIZE)
    """
    input_tensor = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    # create the base pre-trained model
    base_model = InceptionV3(weights=INCEPTION_WEIGHTS,
                             include_top=False,
                             input_tensor=input_tensor)

    #: top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    #: 最終モデル
    model = Model(inputs=base_model.input, outputs=predictions)

    # mixedレイヤーがブロックの区切り
    if freeze == 'initial':
        for layer in base_model.layers:
            layer.trainable = False
    elif freeze == "second":
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True
    elif freeze == 'third':
        for layer in model.layers[:197]:
            layer.trainable = False
        for layer in model.layers[197:]:
            layer.trainable = True
    elif freeze == 'final':
        for layer in model.layers:
            layer.trainable = True
    else:
        raise Exception('Kwyword error')

    return model


def load_xception(n_classes, weights=None, freeze=None):
    if freeze == 'second' or freeze == 'third' or freeze == 'final':
        model = model_xception(n_classes, freeze=freeze)
        model.load_weights(weights)
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    elif freeze == 'initial':
        model = model_xception(n_classes, freeze=freeze)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
    elif freeze == 'inference':
        model = model_xception(n_classes, freeze='final')
        assert weights, "Error, No model found!!"
        model.load_weights(weights)
    else:
        raise Exception('Keyword error')

    return model


def model_xception(n_classes, freeze=None):
    """ Xception
        Note: Inception v3のデフォルト入力サイズは(TARGET_SIZE)
    """
    input_tensor = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    # create the base pre-trained model
    base_model = Xception(weights=XCEPTION_WEIGHTS,
                          include_top=False,
                          input_tensor=input_tensor)

    #: top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    #: 最終モデル
    model = Model(inputs=base_model.input, outputs=predictions)

    #: addレイヤーがブロックの区切り
    #: 変更　125-105では二回目の学習効果が薄かった
    #: 上層：125 115 中層：105 下層：95
    if freeze == 'initial':
        for layer in base_model.layers:
            layer.trainable = False
    elif freeze == "second":
        for layer in model.layers[:115]:
            layer.trainable = False
        for layer in model.layers[115:]:
            layer.trainable = True
    elif freeze == 'third':
        for layer in model.layers[:105]:
            layer.trainable = False
        for layer in model.layers[105:]:
            layer.trainable = True
    elif freeze == 'final':
        for layer in model.layers:
            layer.trainable = True
    else:
        raise Exception('Kwyword error')

    return model
