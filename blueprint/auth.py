from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import mysql.connector

auth = Blueprint('auth', __name__)

# Connect to your DB
def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='userbd'
    )
    return conn

# Login route
@auth.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute('SELECT * FROM users WHERE email = %s AND password = %s', (email, password))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        session['user_id'] = user['id']
        session['email'] = user['email']
        return redirect(url_for('auth.profile'))
    else:
        flash('Invalid email or password')
        return redirect(url_for('auth.login_page'))

# Signup route
@auth.route('/signup', methods=['POST'])
def signup():
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        flash('Passwords do not match!')
        return redirect(url_for('auth.login_page'))

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if email exists
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    if cursor.fetchone():
        flash('Email already registered!')
        return redirect(url_for('auth.login_page'))

    cursor.execute('INSERT INTO users (email, password) VALUES (%s, %s)', (email, password))
    conn.commit()

    cursor.close()
    conn.close()

    flash('Signup successful! Please log in.')
    return redirect(url_for('auth.login_page'))

# Login page
@auth.route('/auth')
def login_page():
    return render_template('LoginSignup.html')

# Profile page
@auth.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('auth.login_page'))
    return render_template('profile.html', email=session['email'])
