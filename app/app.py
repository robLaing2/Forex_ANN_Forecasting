from flask import Flask, render_template

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



@app.route('/line')
def line():
    line_labels=labels
    line_values=values
    return render_template('graph.html', title='Bitcoin Monthly Price in USD', max=17000, labels=line_labels, values=line_values)


@app.route('/')
def hello_world():
    return render_template('home.html')


if __name__ == '__main__':
    app.run()
