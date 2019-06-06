import numpy as np
import keras.backend as K


def predict_with_uncertainty(f, x, n_iter=10):
    result = np.zeros((n_iter,) + x.shape)

    for iter in range(n_iter):
        result[iter] = f(x, 1)

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)
    return prediction, uncertainty


if __name__ == '__main__':
    model = 0
    f = K.function([model.layers[0].input, K.learning_phase()],
                   [model.layers[-1].output])
