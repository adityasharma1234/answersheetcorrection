from flask import render_template, url_for, flash, redirect, request
from autograde import app, db, bcrypt
from autograde.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
#def init_db():
    #db.init_app(app)
    #db.app = app
    #db.create_all()



@app.route("/")
@login_required
def hello():
  return render_template("student.html")

@app.route("/home")
@login_required
def home():
  return render_template("essay1.html")


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("Assign1.html")
    if request.method == 'POST':
        email = request.form.get("email")
        pwd = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, pwd):
            login_user(user)
            return redirect(url_for('hello'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
            return redirect(url_for('login'))

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template("AutoGrader.html")
    if request.method == 'POST':
        name = request.form.get("name")
        email = request.form.get("email")
        pwd = request.form.get("password")
        hashed_password = bcrypt.generate_password_hash(pwd).decode('utf-8')
        val1 = User.query.filter_by(email=email).first()
        val2 = User.query.filter_by(username=name).first()
        if val1:
            flash('Email already exists', 'info')
            return redirect(url_for('signup'))
        if val2 :
            flash('Username already exists', 'info')
            return redirect(url_for('signup'))
        user = User(username=name, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))

@app.route("/profile")
@login_required
def profile():
    name = current_user.username
    email = current_user.email
    posts = current_user.posts
    return render_template("profile.html", email=email, name=name, posts=posts)
    
@app.route("/post", methods=['POST'])
@login_required
def Post():
    title = request.form.get("checklist")
    content = request.form.get("essay")
    post = Post(title=title, content=content, author=current_user)
    db.session.add(post)
    db.session.commit()
    flash('Your post has been created!', 'success')
    return redirect(url_for('home'))

    
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('hello'))
