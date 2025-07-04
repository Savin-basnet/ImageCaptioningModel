from flask import Blueprint, render_template
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask import Flask, redirect, url_for

import mysql.connector

bp_profile = Blueprint('profile', __name__)  # The first arg is the Blueprint name!

# Connect to your DB
def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='swarupdb'
    )
    return conn


@bp_profile.route('/profile')
def profile():
    id = session.get('id')
    if not id:
        return redirect(url_for('auth.login_page'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Get user info and photo name from projectusers
    cursor.execute("SELECT * FROM projectusers WHERE id = %s", (id,))
    userdata = cursor.fetchone()

    cursor.close()
    conn.close()

    # Get the uploaded image filename
    photo_filename = userdata['uploadeduserpic'] if userdata and userdata['uploadeduserpic'] else None

    # Build the static path
    photo_url = None
    if photo_filename:
        photo_url = url_for('static', filename=f'uploads/{photo_filename}')

    return render_template('profile.html', userinfo=userdata, userphoto=photo_url)




@bp_profile.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login_page'))

