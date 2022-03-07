from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def home():
	try:
		if request.method == "POST":
			print(request.form)
			f = request.files['face_image']
			print(f.filename)
			return render_template("index.html")
		else:
			return render_template("index.html")
	except Exception as e:
		return "helo"
