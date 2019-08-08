import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    # xs is no. of bedrooms
    xs = np.array([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0],
            dtype=float)
    # ys is price of house(x10000)
    ys = np.array([-1.0,-.5,0.0,.5,1.0,1.5,2.0,2.5,3.0,3.5,4.5],
        dtype=float)

    model = tf.keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])
        ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1000)
    prediction = model.predict(y_new)[0]
    return prediction

if __name__ == '__main__':
    prediction = house_model([7.0])
    print('Answer: ' + str(prediction))
