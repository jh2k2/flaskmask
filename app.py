from flask import Flask, request, render_template
from predictor import predictor
from jinja2 import Template

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        file = request.files['file'].read()
        result = predictor.predict(file)
        return render_template('index.html', str=result)

    else:
        return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__ == '__main__':
    app.run(debug=True)
