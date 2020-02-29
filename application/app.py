from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return "<h1>Predictions<h1>"


if __name__ == '__main__':
    app.run()
