from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras import optimizers


def load_resnet50(n_classes, weights=None):
    model = resnet50(n_classes)
    if weights:
        model.load_weights(weights) 
    return model 


def resnet50(n_classes):
    """ topなしresnet
        Note:Resnetの出力をflattenすると131072
    """
    input_tensor = Input(shape=(256, 256, 3))
    resnet = ResNet50(include_top=False, weights=None,
                      input_tensor=input_tensor)
    
    resnet.load_weights('src/resnet50_notop.h5')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnet.output_shape[1:]))

    top_model.add(Dense(512, activation='softmax'))
    top_model.add(Dropout(0.4))
    top_model.add(Dense(n_classes, activation='softmax'))

    model = Model(input=ResNet50.input, output=top_model(resnet.output))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    return model
