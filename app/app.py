from flask import Flask, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import quandl
import pandas as pd
from dbnomics import fetch_series

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

labels = [
    'Jan', 'Feb', 'Mar', 'Apr',
    'May', 'Jun', 'Jul', 'Aug',
    'Sep', 'Oct', 'Nov', 'Dec'
]

dataInput = []

START_DATE = '2001-01-01'

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

    indices = list(reversed(indices))

    return indices


def getCPI():
    ukCPI = fetch_series('IMF/CPI/M.GB.PCPIHA_PC_CP_A_PT')
    euCPI = fetch_series('IMF/CPI/M.U2.PCPIHA_PC_CP_A_PT')

    dbnomicsQuery = "period >= '" + START_DATE + "'"

    ukCPI = ukCPI.query(dbnomicsQuery)
    euCPI = euCPI.query(dbnomicsQuery)

    ukCPIDict = {ukCPI.period.iloc[i]: ukCPI.value.iloc[i] for i in range(len(ukCPI))}
    euCPIDict = {euCPI.period.iloc[i]: euCPI.value.iloc[i] for i in range(len(euCPI))}

    dates = []
    ukCPIarr = []
    euCPIarr = []

    for k, v in ukCPIDict.items():
        match = euCPIDict.get(k, 0)

        ukCPIarr.append(v)
        euCPIarr.append(match)
        dates.append(k)

    ukCPIarr = np.array(ukCPIarr, dtype=np.float)
    euCPIarr = np.array(euCPIarr, dtype=np.float)

    ukEuCPIRatio = ukCPIarr / euCPIarr

    cpi_mean = 0.3957793542714463
    cpi_std = 0.9813408912397396
    ukEuCPIRatio = (ukEuCPIRatio - cpi_mean) / cpi_std

    cpiDict = {dates[i]: ukEuCPIRatio[i] for i in range(len(dates))}

    return cpiDict

def getForex():
    quandl.ApiConfig.api_key = "VXqfuyrbTE8xxYZzqePw"
    dataGbpEurRate = quandl.get("BOE/XUDLERS", start_date=START_DATE, returns="numpy")
    forexDataN = dataGbpEurRate.Value

    print(forexDataN)
    print("--")

    forex_mean = 1.308
    forex_std = 0.1634
    forexDataN = (forexDataN - forex_mean) / forex_std

    dates = []
    for x in dataGbpEurRate.Date:
        dates.append(pd.Timestamp(x))

    print(dataGbpEurRate.Date)

    averaged = getMovingAverages(forexDataN, 30)

    currentIndex = len(averaged) - 1

    indices = getIndices(currentIndex, 24)

    dates = np.asarray(dates)
    mov = np.asarray(averaged)

    mov = mov[indices]
    dates = dates[indices]

    dataInput.append(mov)

    return dates, mov


def predict():
    new_model = keras.models.load_model('finalModel.h5')

    cpiDict = getCPI()
    recentCpi = 0

    dates, forex = getForex()

    dataDf = pd.DataFrame(columns=['Date','forex','cpi'])

    for x in range(len(forex)):

        date = dates[x]
        dateRounded = date.replace(day=1)
        cpi = cpiDict.get(dateRounded, recentCpi)
        recentCpi = cpi

        dataDf = dataDf.append({
            'Date':date,
            'forex':forex[x],
            'cpi':cpi},
            ignore_index=True)

    features = ['forex','cpi']
    dataSet = dataDf[features]
    dataSet = dataSet.values

    recent = dates[-1]
    dataTf = tf.constant([dataSet])

    y = new_model.predict(dataTf)[0]
    print(y)
    return y, recent


@app.route('/')
def line():
    predictions, recent = predict()

    recentDate = str(recent)[:10]

    recent = recent.month

    if recent > 6:
        x = 18 - recent
    else:
        x = 6 - recent

    months = (labels[-x:] + labels[:-x])

    history = dataInput[0][-6:]

    history = (history * 0.1634) + 1.308
    predictions = (predictions * 0.1634) + 1.308

    history = np.concatenate((history, predictions))
    history = ['%.3f' % elem for elem in history]

    title = 'Forex Forecast.'

    return render_template('graph.html', title=title, max=30, labels=months, history=history, predictions=predictions, recentDate=recentDate)


if __name__ == '__main__':
    app.run()
