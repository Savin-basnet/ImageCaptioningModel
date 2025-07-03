from flask import Blueprint, render_template

bp_home = Blueprint('home', __name__)  # The first arg is the Blueprint name!

@bp_home.route('/')
def index():
    return render_template('HomePage.html')
