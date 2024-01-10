from flask import Flask, request, render_template, Blueprint
from classify import classify_text, info

main = Blueprint('main', __name__)


@main.route('/')
def index():
    infor = info()
    return render_template('index.html', results=infor)

@main.route('/classify')
def classify():
    q = request.args.get('q')
    language = classify_text(q)
    return render_template('result.html', results=language)


# app = Flask(__name__)
app = Flask(__name__,template_folder='./templates',static_folder='./static')

app.register_blueprint(main)
