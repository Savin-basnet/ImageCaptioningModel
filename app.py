# app.py

from flask import Flask, redirect, url_for
from blueprint.ImageCaption import bp_imageCaption, load_models
from blueprint.HomePage import bp_home
from blueprint.auth import auth

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.secret_key = 'swaruptharu117'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load models at startup
load_models()

# Register Blueprints with prefixes
app.register_blueprint(bp_home, url_prefix='/homepage')
app.register_blueprint(bp_imageCaption)  # Caption routes at /caption
app.register_blueprint(auth)

# Root URL redirects to /homepage
@app.route('/')
def index():
    return redirect(url_for('home.index'))  # 'home' is the Blueprint name

if __name__ == '__main__':
    import os
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
