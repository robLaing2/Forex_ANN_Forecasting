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
# labels for the months of the year
labels = [
    'Jan', 'Feb', 'Mar', 'Apr',
    'May', 'Jun', 'Jul', 'Aug',
    'Sep', 'Oct', 'Nov', 'Dec'
]

dataInput = []
START_DATE = '2017-01-01'

# Function to calculate moving averages
def getMovingAverages(data, windowSize):
    movingAverages = []

    # For each datapoint calculate moving average based on window size
    for x in range(len(data)):
        if (x < windowSize):
            window = data[:x + 1]
        else:
            window = data[x - (windowSize - 1):x + 1]

        total = sum(window)
        average = total / len(window)
        movingAverages.append(average)

    return movingAverages

# Function to get indices of past data to be used to predict the models
def getIndices(currentIndex, steps):

    indices = []

    index = currentIndex + 22

    # For three months go back 22 and one month 21
    for i in range(1, steps + 1):
        if i % 4 == 0:
            index = index - 21
            indices.append(index)
        else:
            index = index - 22
            indices.append(index)

    indices = list(reversed(indices))

    return indices

# Function to fetch all inflation data and transform it for the models
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

    # Match the dates
    for k, v in ukCPIDict.items():
        match = euCPIDict.get(k, 0)

        ukCPIarr.append(v)
        euCPIarr.append(match)
        dates.append(k)

    ukCPIarr = np.array(ukCPIarr, dtype=np.float)
    euCPIarr = np.array(euCPIarr, dtype=np.float)

    ukEuCPIRatio = ukCPIarr / euCPIarr

    # The mean and std of the inflation data that was used in training
    # This keeps the normalisation consistent
    cpi_mean = 0.3957793542714463
    cpi_std = 0.9813408912397396
    ukEuCPIRatio = (ukEuCPIRatio - cpi_mean) / cpi_std

    #  Return a dictionary of all inflation data
    cpiDict = {dates[i]: ukEuCPIRatio[i] for i in range(len(dates))}

    return cpiDict

# Function to get the most recent FOREX data
def getForex():
    quandl.ApiConfig.api_key = "VXqfuyrbTE8xxYZzqePw"
    dataGbpEurRate = quandl.get("BOE/XUDLERS", start_date=START_DATE, returns="numpy")
    forexDataN = dataGbpEurRate.Value

    # Use the mean and std to keep normalisation consistent with the trained models
    forex_mean = 1.308
    forex_std = 0.1634
    forexDataN = (forexDataN - forex_mean) / forex_std

    # Convert dates to pandas Timestamps
    dates = []
    for x in dataGbpEurRate.Date:
        dates.append(pd.Timestamp(x))

    # Get moving average values for FOREX
    averaged = getMovingAverages(forexDataN, 10)

    currentIndex = len(averaged) - 1

    # Get indicies of FOREX to be used in the model
    indices = getIndices(currentIndex, 24)

    dates = np.asarray(dates)
    forex = np.asarray(averaged)

    forex = forex[indices]
    dates = dates[indices]

    dataInput.append(forex)

    return dates, forex

# Function to make the predictions using the newest economic data
def predict():
    # the exported model
    new_model = keras.models.load_model('finalModel.h5')

    cpiDict = getCPI()
    recentCpi = 0

    dates, forex = getForex()

    # Create dataframe with all necessary data
    dataDf = pd.DataFrame(columns=['Date','forex','cpi'])

    # Match each FOREX value with the equivalent inflation value
    for x in range(len(forex)):

        date = dates[x]
        # Round down the FOREXs day so it can match with inflation date
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

    # Make predictions
    y = new_model.predict(dataTf)[0]

    # Return predictions and most recent FOREX date
    return y, recent


@app.route('/')
def home():
    predictions, recent = predict()

    # First 10 characters in the string represent the date to be displayed
    recentDate = str(recent)[:10]
    # Shift the labels so that the current month is displayed in the middle of the graph
    recent = recent.month
    if recent > 6:
        x = 18 - recent
    else:
        x = 6 - recent
    months = (labels[-x:] + labels[:-x])

    history = dataInput[0][-6:]

    history = (history * 0.1634) + 1.308
    predictions = (predictions * 0.1634) + 1.308

    forex = np.concatenate((history, predictions))
    forex = ['%.3f' % elem for elem in forex]

    title = 'Forex Forecast.'

    # Pass the front-end HTML the title, graph labels, both history and predicted FOREX values, most recent date
    return render_template('graph.html', title=title, max=30, labels=months, forex=forex, recentDate=recentDate)


if __name__ == '__main__':
    app.run()
