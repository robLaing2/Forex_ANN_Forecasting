from flask import Flask, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import zeros, newaxis
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

labels = [
    'Jan', 'Feb', 'Mar', 'Apr',
    'May', 'Jun', 'Jul', 'Aug',
    'Sep', 'Oct', 'Nov', 'Dec'
]

values = [
    967.67, 1190.89, 1079.75, 1349.19,
    2328.91, 2504.28, 2873.83, 4764.86,
    4349.29, 6458.30, 9907, 16297
]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

input = np.array([1.6386027, 1.60650694, 1.68153863, 1.82628306, 2.02199241, 2.01750265, 1.76344654,
                      1.76777452,
                      1.84826674,
                      1.88859375,
                      1.97464762,
                      1.92250969,
                      1.71640511,
                      1.49507172,
                      1.61676059,
                      1.68930471,
                      1.62853106,
                      1.48789214,
                      1.22168131,
                      0.94374045])

def predict():
    new_model = keras.models.load_model('MultiStepModel.h5')

    b = np.tile(input, (30, 1))
    b = np.array([b])
    b = b.reshape(30, 20, 1)
    b = tf.constant(b)

    y = new_model.predict(b)[0]
    return y


@app.route('/line')
def line():
    line_labels = labels
    prediction = predict()

    history = input[:6]
    predictions = predict()

    history = (history * 0.1634) + 1.308
    predictions = (predictions * 0.1634) + 1.308

    history = np.concatenate((history, predictions))

    title = 'Forex Forecast.'
    return render_template('graph.html', title=title, max=30, labels=line_labels, history=history)


@app.route('/')
def hello_world():
    return render_template('home.html')


if __name__ == '__main__':
    app.run()
