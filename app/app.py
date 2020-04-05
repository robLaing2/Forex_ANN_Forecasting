from flask import Flask, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import zeros, newaxis
import os
import quandl
import pandas as pd
from collections import OrderedDict

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

START_DATE = '2001-01-01'
END_DATE = '2020-02-01'


def getMovingAverages(data, windowSize):
    movingAverages = []

    for x in range(len(data)):
        if (x < windowSize):
            window = data[:x + 1]
        else:
            window = data[x - (windowSize - 1):x + 1]

        total = sum(window)
        average = total / len(window)
        movingAverages.append(average)

    return movingAverages


EPOCHS = 10
EVALUATION_INTERVAL = 100
VALIDATION_STEPS = 50
TIME_LAGS = 600
PREDICTION_HORIZON = 180
STEP = 30
BATCH_SIZE = 30
FOLDS = 4
FUTURE_STEP = 30


def getIndices(currentIndex, steps):

    indices = []

    index = currentIndex + 22

    for i in range(1, steps + 1):
        if i % 4 == 0:
            index = index - 21
            indices.append(index)
        else:
            index = index - 22
            indices.append(index)

    return indices


def predict():
    new_model = keras.models.load_model('MultiStepModel.h5')

    quandl.ApiConfig.api_key = "VXqfuyrbTE8xxYZzqePw"
    dataGbpEurRate = quandl.get("BOE/XUDLERS", start_date=START_DATE, end_date=END_DATE, returns="numpy")
    forexDataN = dataGbpEurRate.Value

    forex_mean = forexDataN.mean()
    forex_std = forexDataN.std()
    forexDataN = (forexDataN - forex_mean) / forex_std

    dates = []
    for x in dataGbpEurRate.Date:
        dates.append(pd.Timestamp(x))

    averaged = getMovingAverages(forexDataN, 30)

    currentIndex = len(averaged) - 1

    indices = getIndices(currentIndex, 20)

    dates = np.asarray(dates)
    mov = np.asarray(averaged)

    mov = mov[indices]

    #print(dates[indices])

    #print()
    #print(input)

    b = np.tile(mov, (30, 1))
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
