from flask import Flask, request, render_template
from predictor import predictor

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
	if request.method == 'POST':
		file = request.files['file'].read()
		result = predictor.predict(file)
		return result[0]
	else:
		return render_template('index.html')

if __name__ == '__main__':
	app.run()
