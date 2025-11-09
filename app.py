from flask import Flask, render_template, redirect, url_for, request, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from io import BytesIO
import pandas as pd
from models import db, User, Result
from utils import calculate_grade, compute_semester_cgpa
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif','xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------- Routes --------------------

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = request.form['role']
        matric_no = request.form.get('matric_no') if role=='student' else None

        if User.query.filter_by(email=email).first():
            flash("Email already registered","danger")
            return redirect(url_for('register'))

        user = User(username=username,email=email,password=password,role=role,matric_no=matric_no)
        db.session.add(user)
        db.session.commit()
        flash("Registered successfully! Please login","success")
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password,password):
            login_user(user)
            flash("Login Successful","success")
            return redirect(url_for('dashboard_student' if user.role=='student' else 'dashboard_officer'))
        else:
            flash("Invalid Credentials","danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully","success")
    return redirect(url_for('login'))

# -------------------- Student Dashboard --------------------

@app.route("/dashboard_student", methods=["GET","POST"])
@login_required
def dashboard_student():
    if current_user.role != 'student':
        return "Unauthorized",403

    if request.method=="POST":
        file = request.files.get('photo')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            current_user.photo = filename
            db.session.commit()
            flash("Photo uploaded","success")

    results = Result.query.filter_by(student_id=current_user.id).all()
    if results:
        # Compute grades & CGPA
        for r in results:
            r.grade, r.remark = calculate_grade(r.score)
        cgpa = compute_semester_cgpa(results)
        for r in results:
            r.semester_cgpa = cgpa
        db.session.commit()
    return render_template("dashboard_student.html", results=results)

# -------------------- Officer Dashboard --------------------

@app.route("/dashboard_officer", methods=["GET","POST"])
@login_required
def dashboard_officer():
    if current_user.role != 'officer':
        return "Unauthorized",403

    if request.method=="POST":
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(path)
            df = pd.read_excel(path)
            # Expected columns: MatricNo, CourseCode, CourseTitle, Score
            for _, row in df.iterrows():
                student = User.query.filter_by(matric_no=row['MatricNo']).first()
                if student:
                    grade, remark = calculate_grade(row['Score'])
                    res = Result(student_id=student.id,
                                 course_code=row['CourseCode'],
                                 course_title=row['CourseTitle'],
                                 score=row['Score'],
                                 grade=grade,
                                 remark=remark)
                    db.session.add(res)
            db.session.commit()
            flash("Results uploaded","success")

    total_students = User.query.filter_by(role='student').count()
    total_results = Result.query.count()
    pending_students = total_students - Result.query.with_entities(Result.student_id).distinct().count()
    recent_results = Result.query.order_by(Result.id.desc()).limit(10).all()
    return render_template("dashboard_officer.html",
                           total_students=total_students,
                           total_results=total_results,
                           pending_students=pending_students,
                           recent_results=recent_results)

# -------------------- PDF Download --------------------

@app.route("/download/<int:student_id>")
@login_required
def download(student_id):
    if current_user.role != 'student' or current_user.id != student_id:
        return "Unauthorized",403
    results = Result.query.filter_by(student_id=student_id).all()
    output = BytesIO()
    df = pd.DataFrame([{'Course Code': r.course_code,
                        'Course Title': r.course_title,
                        'Score': r.score,
                        'Grade': r.grade,
                        'Remark': r.remark} for r in results])
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="results.csv", mimetype="text/csv")

# -------------------- Run --------------------

if __name__ == "__main__":
    app.run(debug=True)
