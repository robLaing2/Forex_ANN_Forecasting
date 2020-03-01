from flask import Flask, render_template
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

labels = [
    'JAN', 'FEB', 'MAR', 'APR',
    'MAY', 'JUN', 'JUL', 'AUG',
    'SEP', 'OCT', 'NOV', 'DEC'
]

values = [
    967.67, 1190.89, 1079.75, 1349.19,
    2328.91, 2504.28, 2873.83, 4764.87,
    4349.29, 6458.30, 9907, 16297
]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]


def predict():
    new_model = keras.models.load_model('newmodel.h5')
    x = tf.constant([[-0.99668541, -0.88890724, -0.86432626, -0.87188964, -0.83596358, -0.72062204]])
    y = new_model.predict(x)
    return y


@app.route('/line')
def line():
    line_labels=labels
    line_values=values
    prediction = predict()
    title = 'Prediction: ' + str(prediction)
    return render_template('graph.html', title=title, max=17000, labels=line_labels, values=line_values)


@app.route('/')
def hello_world():
    return render_template('home.html')


if __name__ == '__main__':
    app.run()
