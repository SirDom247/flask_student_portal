# app.py
from flask import Flask, render_template, redirect, url_for, request, flash, send_file, jsonify
from flask_migrate import Migrate
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from io import BytesIO
import pandas as pd
from datetime import datetime
from models import db, User, Result
from utils import calculate_grade, compute_semester_cgpa
from config import Config
from sqlalchemy import extract
import calendar
import logging
from functools import wraps
import secrets

# -------------------- App Initialization --------------------
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# CSRF Protection
csrf = CSRFProtect()
csrf.init_app(app)

# Rate Limiting - Fixed initialization
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'xlsx', 'xls'}

# WTForms (for validation)
from wtforms import Form, StringField, FloatField, IntegerField, validators

class ResultForm(Form):
    course_code = StringField('Course Code', [
        validators.Length(min=2, max=20),
        validators.Regexp(r'^[A-Z0-9]+$')
    ])
    course_title = StringField('Course Title', [
        validators.Length(min=2, max=100)
    ])
    score = FloatField('Score', [
        validators.NumberRange(min=0, max=100)
    ])
    credit_unit = IntegerField('Credit Unit', [
        validators.NumberRange(min=1, max=10)
    ])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- Security Headers --------------------
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# -------------------- Database Error Handler --------------------
def handle_db_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            db.session.rollback()
            logger.exception(f"Database error in {f.__name__}: {str(e)}")
            flash("A database error occurred. Please try again or contact the admin.", "danger")
            return redirect(url_for('dashboard_officer'))
    return decorated_function

# -------------------- Excel Processing Functions --------------------
def process_excel_results(file):
    """Process Excel file and return validated DataFrame or error."""
    try:
        df = pd.read_excel(file)

        # Normalize column names (safe replace of spaces -> underscores)
        try:
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        except Exception:
            # fallback if some columns are not string-like
            df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

        column_mapping = {
            'matricno': 'matric_no', 'matric_no': 'matric_no', 'matric_number': 'matric_no',
            'registration_number': 'matric_no', 'student_id': 'matric_no', 'id': 'matric_no',
            'matric': 'matric_no',
            'student_name': 'student_name', 'studentname': 'student_name', 'name': 'student_name',
            'full_name': 'student_name', 'fullname': 'student_name', 'student': 'student_name',
            'coursecode': 'course_code', 'course_code': 'course_code', 'course': 'course_code',
            'subject': 'course_code', 'subject_code': 'course_code', 'code': 'course_code',
            'coursetitle': 'course_title', 'course_title': 'course_title', 'title': 'course_title',
            'course_name': 'course_title', 'subject_name': 'course_title',
            'score': 'score', 'marks': 'score', 'percentage': 'score', 'mark': 'score', 'grade_score': 'score',
            'credit_unit': 'credit_unit', 'creditunit': 'credit_unit', 'credit': 'credit_unit',
            'units': 'credit_unit', 'unit': 'credit_unit'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Required columns: matric_no, course_code, score (student_name optional)
        required = ['matric_no', 'course_code', 'score']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Missing required columns: {', '.join(missing)}"

        # Additional validation
        validation_errors = validate_excel_data(df)
        if validation_errors:
            return None, "; ".join(validation_errors)

        df = clean_dataframe(df)
        return df, None
    except Exception as e:
        logger.exception("Error reading Excel file")
        return None, f"Error reading Excel file: {str(e)}"

def validate_excel_data(df):
    """Additional validation for Excel data"""
    errors = []
    
    # Check for duplicate entries
    duplicates = df[df.duplicated(['matric_no', 'course_code'], keep=False)]
    if not duplicates.empty:
        errors.append(f"Found {len(duplicates)} duplicate entries")
    
    # Validate course codes format (if course_code exists)
    if 'course_code' in df.columns:
        invalid_courses = df[~df['course_code'].astype(str).str.match(r'^[A-Z]{2,4}\d{3,4}$', na=False)]
        if not invalid_courses.empty:
            errors.append(f"Found {len(invalid_courses)} invalid course codes")
    
    # Validate matric numbers (basic format check)
    if 'matric_no' in df.columns:
        invalid_matric = df[~df['matric_no'].astype(str).str.match(r'^[A-Z0-9]{5,20}$', na=False)]
        if not invalid_matric.empty:
            errors.append(f"Found {len(invalid_matric)} invalid matric numbers")
    
    return errors

def clean_dataframe(df):
    """
    Safely clean and normalize the DataFrame coming from Excel uploads.
    """
    # drop rows that are completely empty
    df = df.dropna(how='all')

    # Columns we expect to treat as strings
    string_columns = ['matric_no', 'student_name', 'course_code', 'course_title']

    for col in string_columns:
        if col in df.columns:
            # Convert to string safely, replace common "null" markers, strip whitespace
            df[col] = df[col].fillna("").astype(str).str.strip()
            # Convert markers that represent missing data to pandas.NA
            df[col] = df[col].replace(['nan', 'None', 'NaN', ''], pd.NA)

    # Ensure 'score' exists and convert to numeric
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
    else:
        df['score'] = pd.NA

    # Drop rows missing any required fields
    df = df.dropna(subset=['matric_no', 'course_code', 'score'])

    # After dropna, ensure matric_no and course_code are non-empty strings
    df['matric_no'] = df['matric_no'].astype(str).str.strip()
    df = df[df['matric_no'] != ""]

    df['course_code'] = df['course_code'].astype(str).str.strip()
    df = df[df['course_code'] != ""]

    # At this point score is numeric (or NaN). Drop NaN scores and enforce 0-100 range
    df = df.dropna(subset=['score'])
    df = df[(df['score'] >= 0) & (df['score'] <= 100)]

    # credit_unit default to 3 if missing or invalid
    if 'credit_unit' in df.columns:
        df['credit_unit'] = pd.to_numeric(df['credit_unit'], errors='coerce').fillna(3).astype(int)
    else:
        df['credit_unit'] = 3

    # Optionally trim long whitespace in course_title and student_name if present
    if 'student_name' in df.columns:
        df['student_name'] = df['student_name'].astype(str).str.strip().replace(pd.NA, None)
    if 'course_title' in df.columns:
        df['course_title'] = df['course_title'].astype(str).str.strip().replace(pd.NA, None)

    # Reset index for consistent iteration later
    df = df.reset_index(drop=True)

    return df


# -------------------- Enhanced Search Features --------------------

@app.route("/search/students")
@login_required
def search_students():
    """Advanced student search page"""
    if current_user.role != 'officer':
        return "Unauthorized", 403
    
    query = request.args.get('q', '').strip()
    department = request.args.get('department', '').strip()
    status = request.args.get('status', '').strip()
    
    students_query = User.query.filter_by(role='student')
    
    if query:
        like_query = f"%{query}%"
        students_query = students_query.filter(
            (User.matric_no.ilike(like_query)) |
            (User.username.ilike(like_query)) |
            (User.email.ilike(like_query)) |
            (User.first_name.ilike(like_query)) |
            (User.last_name.ilike(like_query))
        )
    
    if department and hasattr(User, 'department'):
        students_query = students_query.filter(User.department.ilike(f"%{department}%"))
    
    if status == 'with_results':
        students_query = students_query.join(Result).group_by(User.id).having(db.func.count(Result.id) > 0)
    elif status == 'without_results':
        students_query = students_query.outerjoin(Result).group_by(User.id).having(db.func.count(Result.id) == 0)
    
    students = students_query.order_by(User.matric_no).all()
    
    return render_template("search_students.html", 
                         students=students, 
                         search_query=query,
                         department_filter=department,
                         status_filter=status)

@app.route("/search/results")
@login_required
def search_results():
    """Advanced results search page"""
    if current_user.role != 'officer':
        return "Unauthorized", 403
    
    # Get filter parameters
    matric_no = request.args.get('matric_no', '').strip()
    course_code = request.args.get('course_code', '').strip()
    grade = request.args.get('grade', '').strip()
    min_score = request.args.get('min_score', '').strip()
    max_score = request.args.get('max_score', '').strip()
    semester = request.args.get('semester', '').strip()
    
    results_query = Result.query
    
    # Apply filters
    if matric_no:
        results_query = results_query.filter(Result.matric_no.ilike(f"%{matric_no}%"))
    
    if course_code:
        results_query = results_query.filter(
            (Result.course_code.ilike(f"%{course_code}%")) |
            (Result.course_title.ilike(f"%{course_code}%"))
        )
    
    if grade:
        results_query = results_query.filter(Result.grade == grade)
    
    if min_score:
        try:
            results_query = results_query.filter(Result.score >= float(min_score))
        except ValueError:
            flash("Invalid minimum score", "warning")
    
    if max_score:
        try:
            results_query = results_query.filter(Result.score <= float(max_score))
        except ValueError:
            flash("Invalid maximum score", "warning")
    
    if semester:
        results_query = results_query.filter(Result.semester == semester)
    
    results = results_query.order_by(Result.date_uploaded.desc()).all()
    
    # Get unique values for filter dropdowns
    all_grades = [g[0] for g in db.session.query(Result.grade).distinct().all() if g[0]]
    all_courses = [c[0] for c in db.session.query(Result.course_code).distinct().all() if c[0]]
    
    return render_template("search_results.html",
                         results=results,
                         matric_no_filter=matric_no,
                         course_code_filter=course_code,
                         grade_filter=grade,
                         min_score_filter=min_score,
                         max_score_filter=max_score,
                         semester_filter=semester,
                         all_grades=all_grades,
                         all_courses=all_courses)

# -------------------- Enhanced Chart Features --------------------

def get_advanced_chart_data():
    """Get comprehensive data for charts and analytics"""
    
    # Grade Distribution
    grade_distribution = db.session.query(
        Result.grade, 
        db.func.count(Result.id)
    ).group_by(Result.grade).all()
    
    # Course Performance (Average scores per course)
    course_performance = db.session.query(
        Result.course_code,
        Result.course_title,
        db.func.avg(Result.score).label('average_score'),
        db.func.count(Result.id).label('student_count')
    ).group_by(Result.course_code, Result.course_title).all()
    
    # Student CGPA Distribution
    student_cgpas = db.session.query(
        Result.student_id,
        db.func.avg(Result.semester_cgpa).label('avg_cgpa')
    ).group_by(Result.student_id).all()
    
    # Monthly Performance Trends
    monthly_trends = db.session.query(
        db.func.extract('month', Result.date_uploaded).label('month'),
        db.func.avg(Result.score).label('avg_score'),
        db.func.count(Result.id).label('result_count')
    ).group_by('month').order_by('month').all()
    
    # Pass/Fail Statistics by Course
    course_pass_fail = []
    for course in course_performance:
        passed = Result.query.filter(
            Result.course_code == course.course_code,
            Result.score >= 50
        ).count()
        failed = Result.query.filter(
            Result.course_code == course.course_code,
            Result.score < 50
        ).count()
        course_pass_fail.append({
            'course_code': course.course_code,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
        })
    
    return {
        'grade_distribution': dict(grade_distribution),
        'course_performance': course_performance,
        'student_cgpas': [cgpa.avg_cgpa for cgpa in student_cgpas if cgpa.avg_cgpa],
        'monthly_trends': monthly_trends,
        'course_pass_fail': course_pass_fail
    }

@app.route("/analytics/dashboard")
@login_required
def analytics_dashboard():
    """Comprehensive analytics dashboard with charts"""
    if current_user.role != 'officer':
        return "Unauthorized", 403
    
    chart_data = get_advanced_chart_data()
    stats = get_dashboard_stats()
    
    return render_template("analytics_dashboard.html",
                         chart_data=chart_data,
                         stats=stats)

@app.route("/api/analytics/grade-distribution")
@login_required
def api_grade_distribution():
    """API endpoint for grade distribution chart data"""
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    grade_data = db.session.query(
        Result.grade, 
        db.func.count(Result.id).label('count')
    ).group_by(Result.grade).all()
    
    return jsonify({
        'labels': [grade[0] for grade in grade_data],
        'data': [grade[1] for grade in grade_data],
        'colors': ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5', '#fef3c7']  # Green shades
    })

@app.route("/api/analytics/course-performance")
@login_required
def api_course_performance():
    """API endpoint for course performance chart data"""
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    course_data = db.session.query(
        Result.course_code,
        db.func.avg(Result.score).label('average_score')
    ).group_by(Result.course_code).having(db.func.count(Result.id) >= 5).all()
    
    return jsonify({
        'labels': [course[0] for course in course_data],
        'data': [float(course[1]) for course in course_data]
    })

@app.route("/api/analytics/student-progress/<matric_no>")
@login_required
def api_student_progress(matric_no):
    """API endpoint for individual student progress chart"""
    student = User.query.filter_by(matric_no=matric_no, role='student').first()
    if not student:
        return jsonify({"error": "Student not found"}), 404
    
    results = Result.query.filter_by(student_id=student.id).order_by(Result.date_uploaded).all()
    
    return jsonify({
        'labels': [result.course_code for result in results],
        'scores': [result.score for result in results],
        'grades': [result.grade for result in results],
        'cgpa': [result.semester_cgpa for result in results]
    })

# -------------------- Enhanced Officer Dashboard with Charts --------------------

# Update the existing dashboard_officer route to include chart data
@app.route("/dashboard_officer", methods=["GET", "POST"])
@login_required
def dashboard_officer():
    if current_user.role != 'officer':
        return "Unauthorized", 403

    # handle Excel upload
    if request.method == "POST" and 'file' in request.files:
        file = request.files.get('file')
        success_count, error_rows, error_msg = handle_excel_upload(file, db.session, current_user)
        if error_msg:
            flash(error_msg, "danger")
        elif error_rows:
            flash(f"Uploaded {success_count} results. Errors rows: {', '.join(map(str, error_rows[:10]))}" + ("..." if len(error_rows) > 10 else ""), "warning")
        else:
            flash(f"Successfully uploaded {success_count} results", "success")
        return redirect(url_for('dashboard_officer'))

    # filters & stats
    query = Result.query
    recent_results = apply_dashboard_filters(query, request.args)
    stats = get_dashboard_stats()
    
    # Get basic chart data for dashboard
    chart_data = get_advanced_chart_data()

    return render_template("dashboard_officer.html",
                           recent_results=recent_results,
                           search_query=request.args.get('search', ''),
                           student_filter=request.args.get('student_filter', ''),
                           course_filter=request.args.get('course_filter', ''),
                           grade_filter=request.args.get('grade_filter', ''),
                           date_from=request.args.get('date_from', ''),
                           date_to=request.args.get('date_to', ''),
                           chart_data=chart_data,
                           **stats)

# -------------------- Quick Search Endpoints --------------------

@app.route("/api/quick-search/students")
@login_required
def api_quick_search_students():
    """Quick search API for students (for autocomplete)"""
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify([])
    
    students = User.query.filter(
        (User.matric_no.ilike(f"%{query}%")) |
        (User.first_name.ilike(f"%{query}%")) |
        (User.last_name.ilike(f"%{query}%")) |
        (User.email.ilike(f"%{query}%"))
    ).filter_by(role='student').limit(10).all()
    
    return jsonify([{
        'id': student.id,
        'matric_no': student.matric_no,
        'name': student.full_name,
        'email': student.email
    } for student in students])

@app.route("/api/quick-search/courses")
@login_required
def api_quick_search_courses():
    """Quick search API for courses (for autocomplete)"""
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify([])
    
    courses = Result.query.filter(
        (Result.course_code.ilike(f"%{query}%")) |
        (Result.course_title.ilike(f"%{query}%"))
    ).distinct().limit(10).all()
    
    return jsonify([{
        'code': course.course_code,
        'title': course.course_title
    } for course in courses])



def get_or_create_student(matric_no, db_session):
    """
    Returns a User student object.
    If there's no student with that matric_no, create a minimal student entry.
    """
    student = User.query.filter_by(matric_no=matric_no, role='student').first()
    if student:
        return student

    # create minimal placeholder student with random password
    username = matric_no
    first_name = "Student"
    last_name = matric_no
    email = f"{matric_no.lower()}@portal.local"
    # Generate random password
    random_password = secrets.token_urlsafe(16)
    password = generate_password_hash(random_password)
    student = User(
        first_name=first_name,
        last_name=last_name,
        username=username,
        other_names=None,
        email=email,
        password=password,
        role='student',
        matric_no=matric_no
    )
    db_session.add(student)
    db_session.flush()  # assign id
    logger.info(f"Created placeholder student {matric_no} (id={student.id})")
    return student

def get_student_full_name_from_user(user):
    if not user:
        return ""
    return user.full_name

# -------------------- Performance Optimization --------------------
def bulk_update_cgpa(student_ids):
    """Update CGPA for multiple students efficiently"""
    updated_count = 0
    for student_id in student_ids:
        results = Result.query.filter_by(student_id=student_id).all()
        if results:
            cgpa = compute_semester_cgpa(results)
            # Use single update query
            Result.query.filter_by(student_id=student_id).update(
                {'semester_cgpa': cgpa}, 
                synchronize_session=False
            )
            updated_count += 1
    if updated_count > 0:
        db.session.commit()
    return updated_count

# -------------------- User Loader --------------------
@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

# -------------------- Home --------------------
@app.route("/")
def home():
    return render_template("home.html")

# -------------------- Register/Login/Logout --------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        other_names = request.form.get('other_names', '').strip() or None
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password_raw = request.form.get('password', '').strip()
        role = request.form.get('role', 'student')
        matric_no = request.form.get('matric_no', '').strip() or None

        if not first_name or not last_name or not username or not email or not password_raw or not role:
            flash("Please fill all required fields.", "danger")
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for('register'))

        if matric_no and User.query.filter_by(matric_no=matric_no).first():
            flash("Matric number already registered.", "danger")
            return redirect(url_for('register'))

        user = User(
            first_name=first_name,
            last_name=last_name,
            other_names=other_names,
            username=username,
            email=email,
            password=generate_password_hash(password_raw),
            role=role,
            matric_no=matric_no
        )
        db.session.add(user)
        db.session.commit()
        flash("Registered successfully! Please login.", "success")
        return redirect(url_for('login'))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    if request.method == "POST":
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            if user.role == 'student':
                return redirect(url_for('dashboard_student'))
            return redirect(url_for('dashboard_officer'))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

# -------------------- Student Dashboard --------------------
@app.route("/dashboard_student")
@login_required
def dashboard_student():
    if current_user.role != 'student':
        return "Unauthorized", 403

    results = Result.query.filter_by(student_id=current_user.id).order_by(Result.id.desc()).all()
    if results:
        cgpa = compute_semester_cgpa(results)
        for r in results:
            r.grade, r.remark = calculate_grade(r.score)
            r.semester_cgpa = cgpa
        db.session.commit()

    return render_template("dashboard_student.html", results=results)

# -------------------- Officer Helpers --------------------
def handle_excel_upload(file, db_session, uploader_user):
    """Process uploaded Excel file and insert results. Returns (success_count, error_rows, error_msg)."""
    if not file or file.filename == '':
        return 0, [], "No file selected."

    if not allowed_file(file.filename):
        return 0, [], "Invalid file type."

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return 0, [], f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB."

    df, error = process_excel_results(file)
    if error:
        return 0, [], error

    if df.empty:
        return 0, [], "No valid records found."

    success_count = 0
    error_rows = []
    affected_student_ids = set()

    for idx, row in df.iterrows():
        try:
            matric_no = str(row['matric_no']).strip()
            course_code = str(row['course_code']).strip()
            course_title = str(row.get('course_title', f"Course {course_code}"))
            score = float(row['score'])
            credit_unit = int(row.get('credit_unit', 3))

            # find or create student
            student = get_or_create_student(matric_no, db_session)
            student_full_name = student.full_name
            # create Result
            result = Result(
                student_id=student.id,
                matric_no=student.matric_no,
                student_name=student_full_name,
                course_code=course_code,
                course_title=course_title,
                credit_unit=credit_unit,
                score=score,
                grade=calculate_grade(score)[0],
                remark=calculate_grade(score)[1],
                semester_cgpa=0.0,
                date_uploaded=datetime.utcnow(),
                uploaded_by=uploader_user.id
            )
            db_session.add(result)
            success_count += 1
            affected_student_ids.add(student.id)

        except Exception as e:
            logger.exception(f"Error processing row {idx + 2}")
            error_rows.append(idx + 2)
            continue

    try:
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        return success_count, error_rows, f"Database error during upload: {str(e)}"

    # recompute CGPA for affected students using bulk operation
    if affected_student_ids:
        bulk_update_cgpa(affected_student_ids)

    return success_count, error_rows, None

def apply_dashboard_filters(query, request_args):
    search_query = request_args.get('search', '').strip()
    student_filter = request_args.get('student_filter', '').strip()
    course_filter = request_args.get('course_filter', '').strip()
    grade_filter = request_args.get('grade_filter', '').strip()
    date_from = request_args.get('date_from', '').strip()
    date_to = request_args.get('date_to', '').strip()

    if search_query:
        like = f"%{search_query}%"
        query = query.filter(
            (Result.matric_no.ilike(like)) |
            (Result.student_name.ilike(like)) |
            (Result.course_code.ilike(like)) |
            (Result.course_title.ilike(like))
        )

    if student_filter:
        query = query.filter(Result.matric_no.ilike(f"%{student_filter}%"))

    if course_filter:
        query = query.filter(
            (Result.course_code.ilike(f"%{course_filter}%")) |
            (Result.course_title.ilike(f"%{course_filter}%"))
        )

    if grade_filter:
        query = query.filter(Result.grade == grade_filter)

    if date_from:
        try:
            df = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(Result.date_uploaded >= df)
        except ValueError:
            flash("Invalid start date format", "warning")

    if date_to:
        try:
            dt = datetime.strptime(date_to, '%Y-%m-%d')
            dt = dt.replace(hour=23, minute=59, second=59)
            query = query.filter(Result.date_uploaded <= dt)
        except ValueError:
            flash("Invalid end date format", "warning")

    return query.order_by(Result.id.desc()).all()

def get_dashboard_stats():
    total_students = User.query.filter_by(role='student').count()
    total_results = Result.query.count()
    students_with_results = db.session.query(Result.student_id).distinct().count()
    pending_students = max(0, total_students - students_with_results)
    students = User.query.filter_by(role='student').all()

    all_grades = [g[0] for g in db.session.query(Result.grade).distinct().all() if g[0]]
    all_courses = [c[0] for c in db.session.query(Result.course_code).distinct().all() if c[0]]

    passed = Result.query.filter(Result.score >= 50).count()
    failed = total_results - passed

    departments = []
    dept_avg_scores = []
    if hasattr(User, 'department'):
        departments = [d[0] for d in db.session.query(User.department).distinct() if d[0]]
        for dept in departments:
            student_ids = [s.id for s in User.query.filter_by(department=dept).all()]
            results_in_dept = Result.query.filter(Result.student_id.in_(student_ids)).all()
            avg = round(sum(r.score for r in results_in_dept) / len(results_in_dept), 2) if results_in_dept else 0
            dept_avg_scores.append(avg)

    trend_labels = []
    trend_data = []
    for month in range(1, 13):
        count = Result.query.filter(extract('month', Result.date_uploaded) == month).count()
        trend_labels.append(calendar.month_abbr[month])
        trend_data.append(count)

    return {
        "total_students": total_students,
        "total_results": total_results,
        "pending_students": pending_students,
        "students": students,
        "all_grades": all_grades,
        "all_courses": all_courses,
        "chart_pass": passed,
        "chart_fail": failed,
        "chart_departments": departments,
        "chart_dept_scores": dept_avg_scores,
        "chart_trend_labels": trend_labels,
        "chart_trend_data": trend_data
    }

@app.route("/export/students")
@login_required
def export_students():
    if current_user.role != 'officer':
        return "Unauthorized", 403
    
    # Get the same filters as search_students
    query = request.args.get('q', '').strip()
    department = request.args.get('department', '').strip()
    status = request.args.get('status', '').strip()
    
    students_query = User.query.filter_by(role='student')
    
    # Apply the same filters as search_students
    if query:
        like_query = f"%{query}%"
        students_query = students_query.filter(
            (User.matric_no.ilike(like_query)) |
            (User.username.ilike(like_query)) |
            (User.email.ilike(like_query)) |
            (User.first_name.ilike(like_query)) |
            (User.last_name.ilike(like_query))
        )
    
    if status == 'with_results':
        students_query = students_query.join(Result).group_by(User.id).having(db.func.count(Result.id) > 0)
    elif status == 'without_results':
        students_query = students_query.outerjoin(Result).group_by(User.id).having(db.func.count(Result.id) == 0)
    
    students = students_query.order_by(User.matric_no).all()
    
    # Create CSV
    output = BytesIO()
    csv_content = "Matric No,Full Name,Username,Email,Results Count\n"
    
    for student in students:
        result_count = len(student.results) if student.results else 0
        csv_content += f'"{student.matric_no}","{student.full_name}","{student.username}","{student.email}",{result_count}\n'
    
    output.write(csv_content.encode('utf-8'))
    output.seek(0)
    
    return send_file(output, as_attachment=True, download_name=f"students_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")

# -------------------- Admin Student Management --------------------
@app.route("/students/edit/<int:student_id>", methods=["GET", "POST"])
@login_required
@handle_db_errors
def edit_student(student_id):
    if current_user.role != 'officer':
        return "Unauthorized", 403

    student = User.query.filter_by(id=student_id, role='student').first()
    if not student:
        flash("Student not found", "danger")
        return redirect(url_for('dashboard_officer'))

    if request.method == "POST":
        first_name = request.form.get('first_name', '').strip()
        other_names = request.form.get('other_names', '').strip() or None
        last_name = request.form.get('last_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        new_matric_no = request.form.get('matric_no', '').strip() or None

        if not first_name or not last_name or not username or not email:
            flash("All name, username and email fields are required.", "danger")
            return redirect(url_for('edit_student', student_id=student_id))

        # uniqueness checks excluding current user
        if User.query.filter(User.email == email, User.id != student.id).first():
            flash("Email already used by another account.", "danger")
            return redirect(url_for('edit_student', student_id=student_id))

        if new_matric_no and User.query.filter(User.matric_no == new_matric_no, User.id != student.id).first():
            flash("Matric number already in use.", "danger")
            return redirect(url_for('edit_student', student_id=student_id))

        try:
            old_matric = student.matric_no
            # update user first
            student.first_name = first_name
            student.other_names = other_names
            student.last_name = last_name
            student.username = username
            student.email = email
            student.matric_no = new_matric_no

            db.session.commit()

            # update all results' stored matric_no and student_name if changed
            needs_update = False
            if old_matric != new_matric_no:
                Result.query.filter_by(student_id=student.id).update({'matric_no': new_matric_no}, synchronize_session=False)
                needs_update = True

            # always sync student_name field in results (in case firstname/lastname/other changed)
            full_name = student.full_name
            Result.query.filter_by(student_id=student.id).update({'student_name': full_name}, synchronize_session=False)
            needs_update = True

            if needs_update:
                db.session.commit()

            # recompute CGPA for the student
            bulk_update_cgpa([student.id])

            flash("Student updated successfully!", "success")
            return redirect(url_for('dashboard_officer'))

        except Exception as e:
            db.session.rollback()
            logger.exception(f"Error updating student {student_id}")
            flash(f"Error updating student: {str(e)}", "danger")
            return redirect(url_for('edit_student', student_id=student_id))

    return render_template("edit_student.html", student=student)

@app.route("/students/delete/<int:student_id>", methods=["POST"])
@login_required
@handle_db_errors
def delete_student(student_id):
    if current_user.role != 'officer':
        return "Unauthorized", 403

    student = User.query.filter_by(id=student_id, role='student').first()
    if not student:
        flash("Student not found", "danger")
        return redirect(url_for('dashboard_officer'))

    try:
        # delete results first
        Result.query.filter_by(student_id=student.id).delete()
        db.session.delete(student)
        db.session.commit()
        flash("Student and their results deleted.", "success")
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Error deleting student {student_id}")
        flash(f"Error deleting student: {str(e)}", "danger")

    return redirect(url_for('dashboard_officer'))

# -------------------- Result CRUD --------------------
@app.route("/results/edit/<int:result_id>", methods=["GET", "POST"])
@login_required
@handle_db_errors
def edit_result(result_id):
    if current_user.role != 'officer':
        return "Unauthorized", 403

    result = Result.query.get(result_id)
    if not result:
        flash("Result not found", "danger")
        return redirect(url_for('dashboard_officer'))

    # Create form for validation
    form = ResultForm(request.form) if request.method == 'POST' else ResultForm()

    if request.method == "POST":
        course_code = request.form.get('course_code', '').strip()
        course_title = request.form.get('course_title', '').strip()
        score = request.form.get('score', '').strip()
        credit_unit = request.form.get('credit_unit', '').strip()

        if not course_code or not course_title or not score or not credit_unit:
            flash("All fields are required.", "danger")
            return redirect(url_for('edit_result', result_id=result_id))

        try:
            # Validate using form (server-side validation)
            form = ResultForm(request.form)
            if not form.validate():
                for field, errors in form.errors.items():
                    for error in errors:
                        flash(f"{getattr(form, field).label.text}: {error}", "danger")
                return redirect(url_for('edit_result', result_id=result_id))

            result.course_code = course_code
            result.course_title = course_title
            result.score = float(score)
            result.credit_unit = int(credit_unit)
            result.grade, result.remark = calculate_grade(result.score)
            db.session.commit()

            # recompute CGPA for the student
            bulk_update_cgpa([result.student_id])

            flash("Result updated successfully.", "success")
            return redirect(url_for('dashboard_officer'))
        except Exception as e:
            db.session.rollback()
            logger.exception(f"Error updating result {result_id}")
            flash(f"Error updating result: {str(e)}", "danger")
            return redirect(url_for('edit_result', result_id=result_id))

    return render_template("edit_result.html", result=result, form=form)

@app.route("/results/delete/<int:result_id>", methods=["POST"])
@login_required
@handle_db_errors
def delete_result(result_id):
    if current_user.role != 'officer':
        return "Unauthorized", 403

    result = Result.query.get(result_id)
    if not result:
        flash("Result not found", "danger")
        return redirect(url_for('dashboard_officer'))

    sid = result.student_id
    try:
        db.session.delete(result)
        db.session.commit()

        # recompute CGPA
        bulk_update_cgpa([sid])

        flash("Result deleted.", "success")
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Error deleting result {result_id}")
        flash(f"Error deleting result: {str(e)}", "danger")

    return redirect(url_for('dashboard_officer'))

# -------------------- Export / Template Download --------------------
@app.route("/download_template")
@login_required
def download_template():
    if current_user.role != 'officer':
        return "Unauthorized", 403

    sample = {
        'matric_no': ['CS2025001', 'CS2025002'],
        'student_name': ['Alice Smith', 'Bob Jones'],
        'course_code': ['CSC101', 'MAT201'],
        'course_title': ['Intro to CS', 'Calculus'],
        'score': [85, 72],
        'credit_unit': [3, 3]
    }
    df = pd.DataFrame(sample)
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="results_template.xlsx")

@app.route("/export_results")
@login_required
def export_results():
    if current_user.role != 'officer':
        return "Unauthorized", 403
    results = Result.query.all()
    df = pd.DataFrame([{
        'student_id': r.student_id,
        'matric_no': r.matric_no,
        'student_name': r.student_name,
        'course_code': r.course_code,
        'course_title': r.course_title,
        'score': r.score,
        'grade': r.grade,
        'credit_unit': r.credit_unit,
        'date_uploaded': r.date_uploaded
    } for r in results])
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f"results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx")

# -------------------- API Routes --------------------
@app.route("/api/v1/results")
@login_required
def api_get_results():
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    results = Result.query.all()
    return jsonify([{
        'id': r.id,
        'matric_no': r.matric_no,
        'course_code': r.course_code,
        'score': r.score,
        'grade': r.grade,
        'date_uploaded': r.date_uploaded.isoformat() if r.date_uploaded else None
    } for r in results])

@app.route("/api/v1/students")
@login_required
def api_get_students():
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    students = User.query.filter_by(role='student').all()
    return jsonify([{
        'id': s.id,
        'matric_no': s.matric_no,
        'full_name': s.full_name,
        'email': s.email
    } for s in students])

# -------------------- Health --------------------
@app.route("/health")
def health_check():
    try:
        db.session.execute("SELECT 1")
        return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)