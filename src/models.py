from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import SGD

from config import INCEPTION_WEIGHTS


def load_model(n_classes, weights=None, freeze=None):
    if freeze == 'second' or freeze == 'final':
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

    return model


def model_inceptionv3(n_classes, freeze=None):
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
    x = Dropout(0.7)(x)
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
    elif freeze == 'final':
        for layer in model.layers[:197]:
            layer.trainable = False
        for layer in model.layers[197:]:
            layer.trainable = True

    return model
