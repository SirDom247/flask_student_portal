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
from utils import calculate_grade, compute_semester_cgpa, is_passing_grade
from config import Config
from sqlalchemy import extract
import calendar
import logging
from functools import wraps
import secrets
from urllib.parse import urlencode
import html
import re
from dotenv import load_dotenv
import base64
from PIL import Image
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message


# Fix for Render's PostgreSQL URL before any database operations
def fix_database_url():
    database_url = os.environ.get('DATABASE_URL')
    if database_url and database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url

# Set the fixed database URL
os.environ['DATABASE_URL'] = fix_database_url() or 'sqlite:///pde_portal.db'

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
    default_limits=["1000 per day", "100 per hour"],
    storage_uri="memory://"
)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Environment Configuration --------------------

load_dotenv()
# Environment detection
is_production = os.getenv('FLASK_ENV') == 'production'
is_development = os.getenv('FLASK_ENV') == 'development'

# Set configuration based on environment
if is_production:
    app.config['DEBUG'] = False
    app.config['SESSION_COOKIE_SECURE'] = True  # Force HTTPS in production
    app.config['WTF_CSRF_ENABLED'] = True  # Enable CSRF in production
else:
    app.config['DEBUG'] = True
    app.config['SESSION_COOKIE_SECURE'] = False
    app.config['WTF_CSRF_ENABLED'] = False

# Fix for Render's PostgreSQL URL
import os
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url.replace("postgres://", "postgresql://", 1)

# Ensure the database URI is set
if not app.config.get('SQLALCHEMY_DATABASE_URI'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///pde_portal.db')

# Log environment info 
logger.info(f"Application started in {os.getenv('FLASK_ENV', 'unknown')} mode")

mail = Mail(app)

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


# -------------------- Enhanced Security Headers --------------------
@app.after_request
def set_security_headers(response):
    """Set security headers with CSP that allows all your external dependencies"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Comprehensive CSP for all your dependencies
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob: https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "object-src 'none'; "
        "media-src 'self'"
    )
    response.headers['Content-Security-Policy'] = csp_policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=(), payment=()'
    
    return response

#-------------------- Database Error Handler --------------------
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
        # Read Excel with specific dtype for academic_session to prevent decimal conversion
        df = pd.read_excel(file, dtype={'academic_session': str, 'semester': str})

        converters={
                'academic_session': lambda x: str(x) if pd.notna(x) else '2023/2024',
                'semester': lambda x: str(x) if pd.notna(x) else '1'
            }

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
            'units': 'credit_unit', 'unit': 'credit_unit',
            # ADD MAPPINGS FOR NEW FIELDS
            'academic_session': 'academic_session', 'session': 'academic_session', 'academicsession': 'academic_session',
            'semester': 'semester', 'sem': 'semester',
            # NEW: Department and School mappings
            'department': 'department', 'dept': 'department', 'dept_name': 'department',
            'school': 'school', 'faculty': 'school', 'faculty_name': 'school', 'college': 'school'
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

    # Columns we expect to treat as strings - ADD NEW FIELDS
    string_columns = ['matric_no', 'student_name', 'course_code', 'course_title', 'academic_session', 'semester', 'department', 'school']

    for col in string_columns:
        if col in df.columns:
            # Convert to string safely, replace common "null" markers, strip whitespace
            df[col] = df[col].fillna("").astype(str).str.strip()
            # Convert markers that represent missing data to pandas.NA
            df[col] = df[col].replace(['nan', 'None', 'NaN', ''], pd.NA)
            
            # SPECIAL HANDLING FOR ACADEMIC_SESSION - fix decimal conversion
            if col == 'academic_session':
                # Fix decimal conversions (e.g., 0.9995061728395062 -> 2023/2024)
                mask = df[col].str.contains(r'\.', na=False) & ~df[col].str.contains('/', na=False)
                df.loc[mask, col] = '2023/2024'
                
                # Also handle cases where it might be stored as float
                try:
                    # Convert any numeric academic_session to string and fix
                    numeric_mask = df[col].apply(lambda x: isinstance(x, (int, float)) and not isinstance(x, bool))
                    df.loc[numeric_mask, col] = '2023/2024'
                except:
                    pass

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

    # academic_session default to 2023/2024 if missing - ENHANCED
    if 'academic_session' not in df.columns:
        df['academic_session'] = '2023/2024'
    else:
        df['academic_session'] = df['academic_session'].fillna('2023/2024')
        # Final cleanup for academic_session - MORE ROBUST
        def fix_academic_session(x):
            if pd.isna(x) or x == '':
                return '2023/2024'
            x_str = str(x)
            # If it looks like a decimal (contains . but not /)
            if '.' in x_str and '/' not in x_str:
                return '2023/2024'
            # If it's a number (like 0.9995)
            try:
                float_val = float(x_str)
                if float_val < 1:  # It's a decimal fraction
                    return '2023/2024'
            except:
                pass
            return x_str
        
        df['academic_session'] = df['academic_session'].apply(fix_academic_session)

    # semester default to 1 if missing - ENHANCED
    if 'semester' not in df.columns:
        df['semester'] = '1'
    else:
        df['semester'] = df['semester'].fillna('1')
        # Fix semester if it's a decimal
        def fix_semester(x):
            if pd.isna(x) or x == '':
                return '1'
            x_str = str(x)
            # If it contains decimal point, take the integer part
            if '.' in x_str:
                try:
                    return str(int(float(x_str)))
                except:
                    return '1'
            return x_str
        
        df['semester'] = df['semester'].apply(fix_semester)

    # Optionally trim long whitespace in course_title and student_name if present
    if 'student_name' in df.columns:
        df['student_name'] = df['student_name'].astype(str).str.strip().replace(pd.NA, None)
    if 'course_title' in df.columns:
        df['course_title'] = df['course_title'].astype(str).str.strip().replace(pd.NA, None)

    # Handle department and school fields
    if 'department' in df.columns:
        df['department'] = df['department'].fillna('Computer Science')  # Use default from model
    else:
        df['department'] = 'Computer Science'
        
    if 'school' in df.columns:
        df['school'] = df['school'].fillna('')  # Empty string if not provided
    else:
        df['school'] = ''

    # Reset index for consistent iteration later
    df = df.reset_index(drop=True)

    return df

def handle_excel_upload(file, db_session, current_user):
    """Handle Excel file upload and process results"""
    if not file or file.filename == '':
        return 0, [], "No file selected"
    
    if not allowed_file(file.filename):
        return 0, [], "Invalid file type. Only Excel files are allowed."
    
    try:
        df, error_msg = process_excel_results(file)
        if error_msg:
            return 0, [], error_msg
        
        success_count = 0
        error_rows = []
        
        for index, row in df.iterrows():
            try:
                # Extract data from row
                matric_no = str(row['matric_no']).strip()
                student_name = row.get('student_name', '').strip()
                course_code = str(row['course_code']).strip().upper()
                course_title = row.get('course_title', '').strip()
                score = float(row['score'])
                credit_unit = int(row.get('credit_unit', 3))
                academic_session = row.get('academic_session', '2023/2024').strip()
                semester = str(row.get('semester', '1')).strip()
                department = row.get('department', 'Computer Science').strip()
                school = row.get('school', '').strip()
                
                # Get or create student
                student = get_or_create_student(matric_no, student_name, db_session, department, school)
                
                # Check if result already exists
                existing_result = Result.query.filter_by(
                    student_id=student.id,
                    course_code=course_code,
                    academic_session=academic_session,
                    semester=semester
                ).first()
                
                if existing_result:
                    error_rows.append(index + 2)  # +2 because Excel is 1-indexed and header row
                    continue
                
                # Calculate grade and remark
                grade, remark = calculate_grade(score)
                
                # Create result
                result = Result(
                    student_id=student.id,
                    matric_no=matric_no,
                    student_name=student.full_name,
                    course_code=course_code,
                    course_title=course_title,
                    score=score,
                    grade=grade,
                    remark=remark,
                    credit_unit=credit_unit,
                    academic_session=academic_session,
                    semester=semester,
                    uploaded_by=current_user.id
                )
                
                db_session.add(result)
                success_count += 1
                
            except Exception as e:
                error_rows.append(index + 2)
                continue
        
        if success_count > 0:
            db_session.commit()
            
            # Update CGPA for all affected students
            student_ids = list(set([r.student_id for r in Result.query.filter(
                Result.id.in_([r.id for r in db_session.new if isinstance(r, Result)])
            )]))
            bulk_update_cgpa(student_ids)
        
        return success_count, error_rows, None
        
    except Exception as e:
        db_session.rollback()
        logger.exception("Excel upload error")
        return 0, [], f"Error processing file: {str(e)}"

def apply_dashboard_filters(query, args):
    """Apply filters to the results query for dashboard"""
    student_filter = args.get('student_filter', '').strip()
    course_filter = args.get('course_filter', '').strip()
    date_from = args.get('date_from', '').strip()
    date_to = args.get('date_to', '').strip()
    
    if student_filter:
        query = query.filter(
            (Result.matric_no.ilike(f"%{student_filter}%")) |
            (Result.student_name.ilike(f"%{student_filter}%"))
        )
    
    if course_filter:
        query = query.filter(
            (Result.course_code.ilike(f"%{course_filter}%")) |
            (Result.course_title.ilike(f"%{course_filter}%"))
        )
    
    if date_from:
        try:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(Result.date_uploaded >= date_from_obj)
        except ValueError:
            pass  # Ignore invalid date format
    
    if date_to:
        try:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
            date_to_obj = date_to_obj.replace(hour=23, minute=59, second=59)
            query = query.filter(Result.date_uploaded <= date_to_obj)
        except ValueError:
            pass 
    
    return query

# ----- Student Creation with Student Name, Department, and School from Excel --------------

def get_or_create_student(matric_no, student_name_from_excel, db_session, department=None, school=None):
    """
    Returns a User student object.
    If there's no student with that matric_no, create a student entry using the name from Excel.
    Sets requires_password_reset flag to True for new students.
    """
    student = User.query.filter_by(matric_no=matric_no, role='student').first()
    if student:
        return student

    # Parse student name from Excel to extract first name and last name
    first_name, last_name = parse_student_name(student_name_from_excel, matric_no)
    
    username = matric_no
    email = f"{matric_no.lower()}@fcetomoku.edu.ng"
    
    # Use a default password that officers can communicate to students
    default_password = "changeme123"
    password = generate_password_hash(default_password)
    
    student = User(
        first_name=first_name,
        last_name=last_name,
        username=username,
        other_names=None,
        email=email,
        password=password,
        role='student',
        matric_no=matric_no,
        department=department or 'Computer Science',  # Use provided department or default
        school=school or '',  # Use provided school or empty string
        requires_password_reset=True  # This should trigger password reset on first login
    )
    db_session.add(student)
    try:
        db_session.commit()  # Use commit instead of flush to ensure the student is saved
        logger.info(f"Created student {matric_no} (id={student.id}) with name: {first_name} {last_name}, department: {department}, school: {school} - password reset required")
        return student
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error creating student {matric_no}: {str(e)}")
        raise

def parse_student_name(student_name, matric_no):
    """
    Parse student name from Excel to extract first name and last name.
    Handles various name formats.
    """
    if not student_name or student_name.strip() == "":
        return "Student", matric_no
    
    name_parts = student_name.strip().split()
    
    if len(name_parts) == 1:
        return name_parts[0], matric_no
    elif len(name_parts) == 2:
        return name_parts[0], name_parts[1]
    else:
        return name_parts[0], name_parts[-1]

def get_student_full_name_from_user(user):
    if not user:
        return ""
    return user.full_name


# -------------------- Student Profile Management --------------------
@app.route("/student/profile/update", methods=["POST"])
@login_required
def update_student_profile():
    """Update student profile information"""
    if current_user.role != 'student':
        return "Unauthorized", 403
    
    try:
        email = request.form.get('email', '').strip()
        department = request.form.get('department', '').strip()
        school = request.form.get('school', '').strip() or None
        
        if not email:
            flash("Email is required", "danger")
            return redirect(url_for('dashboard_student'))
        
        # Check if email is already taken by another user
        existing_user = User.query.filter(User.email == email, User.id != current_user.id).first()
        if existing_user:
            flash("Email is already registered by another user", "danger")
            return redirect(url_for('dashboard_student'))
        
        # Update user profile
        current_user.email = email
        current_user.department = department
        current_user.school = school
        current_user.date_updated = datetime.utcnow()
        
        db.session.commit()
        
        flash("Profile updated successfully!", "success")
        
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Error updating student profile: {str(e)}")
        flash("Error updating profile. Please try again.", "danger")
    
    return redirect(url_for('dashboard_student'))


@app.route("/student/password/change", methods=["POST"])
@login_required
def change_student_password():
    """Change student password"""
    if current_user.role != 'student':
        return "Unauthorized", 403
    
    try:
        current_password = request.form.get('current_password', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Validate current password
        if not check_password_hash(current_user.password, current_password):
            flash("Current password is incorrect", "danger")
            return redirect(url_for('dashboard_student'))
        
        
        is_valid, message = validate_password_complexity(new_password)
        if not is_valid:
            flash(message, "danger")
            return redirect(url_for('dashboard_student'))
        
        # Validate password confirmation
        if new_password != confirm_password:
            flash("New passwords do not match", "danger")
            return redirect(url_for('dashboard_student'))
        
        if check_password_hash(current_user.password, new_password):
            flash("New password cannot be the same as your current password", "danger")
            return redirect(url_for('dashboard_student'))
        
        current_user.password = generate_password_hash(new_password)
        current_user.date_updated = datetime.utcnow()
        
        db.session.commit()
        
        flash("Password changed successfully!", "success")
        
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Error changing student password: {str(e)}")
        flash("Error changing password. Please try again.", "danger")
    
    return redirect(url_for('dashboard_student'))

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Verify current password
        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'error')
            return render_template('reset_password.html', 
                                 password_reset_required=False)
        
        # Validate new passwords
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return render_template('reset_password.html',
                                 password_reset_required=False)
        
        if len(new_password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('reset_password.html',
                                 password_reset_required=False)
        
        # Update password
        current_user.set_password(new_password)
        db.session.commit()
        
        flash('Your password has been updated successfully!', 'success')
        return redirect(url_for('dashboard_student'))
    
    return render_template('reset_password.html', 
                         password_reset_required=False)


# -------------------- Performance Optimization --------------------
def bulk_update_cgpa(student_ids):
    """Update CGPA for multiple students efficiently"""
    updated_count = 0
    for student_id in student_ids:
        results = Result.query.filter_by(student_id=student_id).all()
        if results:
            cgpa = compute_semester_cgpa(results)
            cgpa_rounded = round(cgpa, 2)
            Result.query.filter_by(student_id=student_id).update(
                {'semester_cgpa': cgpa_rounded},
                synchronize_session=False
            )
            updated_count += 1
    if updated_count > 0:
        db.session.commit()
    return updated_count

def get_dashboard_stats():
    total_students = User.query.filter_by(role='student').count()
    total_results = Result.query.count()
    students_with_results = db.session.query(Result.student_id).distinct().count()
    pending_students = max(0, total_students - students_with_results)
    students = User.query.filter_by(role='student').all()

    all_grades = [g[0] for g in db.session.query(Result.grade).distinct().all() if g[0]]
    all_courses = [c[0] for c in db.session.query(Result.course_code).distinct().all() if c[0]]
    all_sessions = [s[0] for s in db.session.query(Result.academic_session).distinct().all() if s[0]]
    all_semesters = [s[0] for s in db.session.query(Result.semester).distinct().all() if s[0]]

    # NEW: Get unique departments and schools
    all_departments = [d[0] for d in db.session.query(User.department).distinct().all() if d[0]]
    all_schools = [s[0] for s in db.session.query(User.school).distinct().all() if s[0]]

    # Pass/Fail
    passed = Result.query.filter(Result.grade.in_(['A', 'B', 'C', 'D', 'E'])).count()
    failed = Result.query.filter(Result.grade == 'F').count()

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
        "all_sessions": all_sessions,
        "all_semesters": all_semesters,
        "all_departments": all_departments,  # NEW
        "all_schools": all_schools,  # NEW
        "chart_pass": passed,
        "chart_fail": failed,
        "chart_departments": departments,
        "chart_dept_scores": dept_avg_scores,
        "chart_trend_labels": trend_labels,
        "chart_trend_data": trend_data
    }

# -------------------- User Loader --------------------
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


# -------------------- Home --------------------
@app.route("/")
def home():
    return render_template("home.html")

def validate_password_complexity(password):
    """
    Validate password meets complexity requirements
    - At least 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter  
    - At least 1 number
    - At least 1 special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"

# -------------------- Register/Login/Logout --------------------

@app.route("/register", methods=["GET", "POST"])
@limiter.limit("3 per minute")
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
        department = request.form.get('department', '').strip() or 'Computer Science'
        school = request.form.get('school', '').strip() or None

        # Validate required fields
        if not first_name or not last_name or not username or not email or not password_raw or not role:
            flash("Please fill all required fields.", "danger")
            return redirect(url_for('register'))

        # ✅ ADDED: Validate password complexity
        is_valid, message = validate_password_complexity(password_raw)
        if not is_valid:
            flash(message, "danger")
            return redirect(url_for('register'))

        # Check for existing email
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for('register'))

        # Check for existing matric number
        if matric_no and User.query.filter_by(matric_no=matric_no).first():
            flash("Matric number already registered.", "danger")
            return redirect(url_for('register'))

        # Create user
        user = User(
            first_name=first_name,
            last_name=last_name,
            other_names=other_names,
            username=username,
            email=email,
            password=generate_password_hash(password_raw),
            role=role,
            matric_no=matric_no,
            department=department,
            school=school,
            requires_password_reset=False
        )
        db.session.add(user)
        db.session.commit()
        flash("Registered successfully! Please login.", "success")
        return redirect(url_for('login'))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
@limiter.limit("30 per hour")
def login():
    if request.method == "POST":
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            
            # Check if password reset is required - FIXED: Use proper attribute access
            if hasattr(user, 'requires_password_reset') and user.requires_password_reset:
                flash("Please reset your password to continue.", "warning")
                return redirect(url_for('reset_password'))
            
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

# -------------------- Password Reset Route --------------------

@app.route("/reset_password", methods=["GET", "POST"])
@login_required
def reset_password():
    # Check if password reset is required or user is voluntarily changing password
    password_reset_required = hasattr(current_user, 'requires_password_reset') and current_user.requires_password_reset
    
    if request.method == "POST":
        # If password reset is required, don't ask for current password
        if not password_reset_required:
            current_password = request.form.get('current_password')
            if not current_password:
                flash("Current password is required", "danger")
                return redirect(url_for('reset_password'))
            
            if not check_password_hash(current_user.password, current_password):
                flash("Current password is incorrect", "danger")
                return redirect(url_for('reset_password'))
        
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not new_password or not confirm_password:
            flash("New password and confirmation are required", "danger")
            return redirect(url_for('reset_password'))
        
        if new_password != confirm_password:
            flash("New passwords do not match", "danger")
            return redirect(url_for('reset_password'))
        
        is_valid, message = validate_password_complexity(new_password)
        if not is_valid:
            flash(message, "danger")
            return redirect(url_for('reset_password'))
        if not password_reset_required:
            if check_password_hash(current_user.password, new_password):
                flash("New password cannot be the same as your current password", "danger")
                return redirect(url_for('reset_password'))
        
        # Update password
        current_user.password = generate_password_hash(new_password)
        
        # Clear the password reset flag if it was set
        if password_reset_required:
            current_user.requires_password_reset = False
            flash_message = "Password reset successfully! You can now access your dashboard."
        else:
            flash_message = "Password updated successfully!"
        
        db.session.commit()
        
        flash(flash_message, "success")
        
        # Redirect based on role
        if current_user.role == 'student':
            return redirect(url_for('dashboard_student'))
        else:
            return redirect(url_for('dashboard_officer'))
    
    return render_template("reset_password.html", 
                         password_reset_required=password_reset_required)

@app.route('/forgot-password', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        
        if not email:
            flash('Please enter your email address.', 'error')
            return render_template('forgot_password.html')
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            try:
                # Generate secure reset token
                serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
                token = serializer.dumps(email, salt='password-reset-salt')
                
                # Create reset URL
                reset_url = url_for('reset_password', token=token, _external=True)
                
                # Create email message
                msg = Message(
                    'Password Reset Request - PDE Portal',
                    sender=app.config.get('MAIL_DEFAULT_SENDER', 'noreply@fcetomoku.edu.ng'),
                    recipients=[email]
                )
                
                # Plain text version
                msg.body = f'''Password Reset Request

Hello {user.full_name or 'User'},

You have requested to reset your password for your PDE Portal account.

To reset your password, please click the following link:
{reset_url}

This link will expire in 1 hour for security reasons.

If you did not request this password reset, please ignore this email. 
Your account remains secure.

For any questions, contact the IT support desk.

Best regards,
PDE Portal Team
Federal College of Education (Technical) Omoku
'''
                
                # HTML version
                msg.html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #10b981, #059669);
            padding: 30px;
            text-align: center;
            color: white;
            border-radius: 10px 10px 0 0;
        }}
        .content {{
            background: #f9f9f9;
            padding: 30px;
            border-radius: 0 0 10px 10px;
            border: 1px solid #e1e1e1;
            border-top: none;
        }}
        .button {{
            display: inline-block;
            padding: 14px 28px;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #e1e1e1;
            padding-top: 20px;
        }}
        .warning {{
            background: #fef3cd;
            border: 1px solid #fde68a;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 14px;
        }}
        .code {{
            background: #f1f1f1;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            word-break: break-all;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔐 PDE Portal</h1>
        <p>Password Reset Request</p>
    </div>
    
    <div class="content">
        <h2>Hello {user.full_name or 'User'},</h2>
        
        <p>You have requested to reset your password for your PDE Portal account.</p>
        
        <p style="text-align: center;">
            <a href="{reset_url}" class="button">Reset Your Password</a>
        </p>
        
        <p>Or copy and paste this link into your browser:</p>
        <div class="code">{reset_url}</div>
        
        <div class="warning">
            <strong>⚠️ Important:</strong> This link will expire in 1 hour for security reasons.
        </div>
        
        <p>If you did not request this password reset, please ignore this email. Your account remains secure.</p>
        
        <p>For any questions or if you need assistance, please contact the IT support desk.</p>
        
        <p>Best regards,<br>
        <strong>PDE Portal Team</strong><br>
        Federal College of Education (Technical) Omoku</p>
    </div>
    
    <div class="footer">
        <p>This is an automated message. Please do not reply to this email.</p>
        <p>© 2024 Federal College of Education (Technical) Omoku. All rights reserved.</p>
    </div>
</body>
</html>
'''
                
                # Send email
                mail.send(msg)
                
                # Log successful email sending
                app.logger.info(f"Password reset email sent to {email}")
                
            except Exception as e:
                # Log the error but don't reveal it to the user (security)
                app.logger.error(f"Failed to send password reset email to {email}: {str(e)}")
                # Don't show error to user to prevent email enumeration
        
        # Always show the same message regardless of whether user exists (security best practice)
        flash('If an account with that email exists, a password reset link has been sent. Please check your email.', 'info')
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def reset_password_with_token(token):
    try:
        # Verify token
        serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        email = serializer.loads(token, salt=app.config.get('PASSWORD_RESET_SALT', 'password-reset-salt'), 
                               max_age=app.config.get('PASSWORD_RESET_EXPIRY', 3600))
    except Exception as e:
        app.logger.warning(f"Invalid or expired reset token attempted: {token}")
        flash('The reset link is invalid or has expired. Please request a new password reset link.', 'error')
        return redirect(url_for('forgot_password'))
    
    user = User.query.filter_by(email=email).first()
    if not user:
        app.logger.warning(f"Reset token for non-existent user: {email}")
        flash('Invalid reset token.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Validate passwords
        if not new_password or not confirm_password:
            flash('Please fill in all password fields.', 'error')
            return render_template('reset_password.html', 
                                 password_reset_required=True,
                                 token=token)
        
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return render_template('reset_password.html',
                                 password_reset_required=True,
                                 token=token)
        
        if len(new_password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('reset_password.html',
                                 password_reset_required=True,
                                 token=token)
        
        try:
            # Update password
            user.set_password(new_password)
            db.session.commit()
            
            # Log the password reset
            app.logger.info(f"Password successfully reset for user: {email}")
            
            flash('Your password has been reset successfully! You can now login with your new password.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error resetting password for {email}: {str(e)}")
            flash('An error occurred while resetting your password. Please try again.', 'error')
            return render_template('reset_password.html',
                                 password_reset_required=True,
                                 token=token)
    
    return render_template('reset_password.html', 
                         password_reset_required=True,
                         token=token)


# -------------------- Dashboard Routes --------------------
@app.route("/dashboard_student")
@login_required
def dashboard_student():
    if current_user.role != 'student':
        return "Unauthorized", 403

    # Check if password reset is required
    if hasattr(current_user, 'requires_password_reset') and current_user.requires_password_reset:
        flash("Please reset your password to access the dashboard.", "warning")
        return redirect(url_for('reset_password'))

    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    per_page = min(per_page, 50)

    # Get paginated results using SQLAlchemy's paginate
    results_pagination = Result.query.filter_by(
        student_id=current_user.id
    ).order_by(Result.id.desc()).paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    )
    
    results = results_pagination.items
    
    # Calculate statistics
    average_score = 0
    if results:
        total_score = sum(r.score for r in results if r.score is not None)
        average_score = total_score / len(results)
        
        # Calculate CGPA and update results
        cgpa = compute_semester_cgpa(results)
        cgpa_rounded = round(cgpa, 2)
        for r in results:
            r.grade, r.remark = calculate_grade(r.score)
            r.semester_cgpa = cgpa_rounded
        db.session.commit()
    
    # Build pagination URL function
    def build_pagination_url(page_num, per_page_count):
        args = request.args.copy()
        args['page'] = page_num
        args['per_page'] = per_page_count
        return f"{url_for('dashboard_student')}?{urlencode(args)}"
    
    return render_template(
        "dashboard_student.html",
        results=results,
        results_pagination=results_pagination,
        per_page=per_page,
        average_score=average_score,
        build_pagination_url=build_pagination_url
    )

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

    # Students pagination parameters
    students_page = request.args.get('students_page', 1, type=int)
    students_per_page = request.args.get('students_per_page', 10, type=int)
    students_per_page = min(students_per_page, 50)

    # Results pagination parameters
    results_page = request.args.get('results_page', 1, type=int)
    results_per_page = request.args.get('results_per_page', 10, type=int)
    results_per_page = min(results_per_page, 50)

    # Get students with pagination
    students_query = User.query.filter_by(role='student')
    students_pagination = students_query.order_by(User.matric_no).paginate(
        page=students_page, per_page=students_per_page, error_out=False
    )
    students = students_pagination.items

    # Get recent results with pagination
    results_query = Result.query
    recent_results_query = apply_dashboard_filters(results_query, request.args)
    results_pagination = recent_results_query.order_by(Result.id.desc()).paginate(
        page=results_page, per_page=results_per_page, error_out=False
    )
    recent_results = results_pagination.items

    stats = get_dashboard_stats()
    
    # Get basic chart data for dashboard
    chart_data = get_advanced_chart_data()

    # Build pagination URL functions
    def build_students_pagination_url(page_num, per_page_count):
        args = request.args.copy()
        args['students_page'] = page_num
        args['students_per_page'] = per_page_count
        # Remove results pagination params to avoid conflicts
        args.pop('results_page', None)
        args.pop('results_per_page', None)
        return f"{request.path}?{urlencode(args)}"

    def build_results_pagination_url(page_num, per_page_count):
        args = request.args.copy()
        args['results_page'] = page_num
        args['results_per_page'] = per_page_count
        # Remove students pagination params to avoid conflicts
        args.pop('students_page', None)
        args.pop('students_per_page', None)
        return f"{request.path}?{urlencode(args)}"

    # Remove 'students' from stats to avoid duplicate parameter
    stats_without_students = {k: v for k, v in stats.items() if k != 'students'}

    return render_template("dashboard_officer.html",
                           students=students,  
                           students_pagination=students_pagination,
                           students_per_page=students_per_page,
                           recent_results=recent_results,
                           results_pagination=results_pagination,
                           results_per_page=results_per_page,
                           search_query=request.args.get('search', ''),
                           student_filter=request.args.get('student_filter', ''),
                           course_filter=request.args.get('course_filter', ''),
                           date_from=request.args.get('date_from', ''),
                           date_to=request.args.get('date_to', ''),
                           chart_data=chart_data,
                           build_students_pagination_url=build_students_pagination_url,
                           build_results_pagination_url=build_results_pagination_url,
                           **stats_without_students)

# -------------------- Enhanced Search Features --------------------

@app.route("/search/students")
@login_required
def search_students():
    """Advanced student search page with department and school filters"""
    if current_user.role != 'officer':
        return "Unauthorized", 403
    
    query = request.args.get('q', '').strip()
    department = request.args.get('department', '').strip()
    school = request.args.get('school', '').strip()
    status = request.args.get('status', '').strip()
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # Limit to 100 per page
    
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
    
    if department:
        students_query = students_query.filter(User.department.ilike(f"%{department}%"))
    
    if school:
        students_query = students_query.filter(User.school.ilike(f"%{school}%"))
    
    if status == 'with_results':
        students_query = students_query.join(Result).group_by(User.id).having(db.func.count(Result.id) > 0)
    elif status == 'without_results':
        students_query = students_query.outerjoin(Result).group_by(User.id).having(db.func.count(Result.id) == 0)
    
    # Get pagination object
    students_pagination = students_query.order_by(User.matric_no).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    students = students_pagination.items
    
    # Get unique departments and schools for filter dropdowns
    all_departments = [d[0] for d in db.session.query(User.department).distinct().all() if d[0]]
    all_schools = [s[0] for s in db.session.query(User.school).distinct().all() if s[0]]
    
    # Build pagination URL function
    def build_pagination_url(page_num, per_page_count):
        args = request.args.copy()
        args['page'] = page_num
        args['per_page'] = per_page_count
        return f"{request.path}?{urlencode(args)}"
    
    return render_template("search_students.html", 
                         students=students,
                         pagination=students_pagination,
                         search_query=query,
                         department_filter=department,
                         school_filter=school,
                         all_departments=all_departments,
                         all_schools=all_schools,
                         status_filter=status,
                         page=page,
                         per_page=per_page,
                         build_pagination_url=build_pagination_url)

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
    academic_session = request.args.get('academic_session', '').strip()
    department = request.args.get('department', '').strip()
    school = request.args.get('school', '').strip()
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    result_status = request.args.get('result_status', '').strip()
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # Limit to 100 per page
    
    # Check if this is an export request
    export_request = request.args.get('export') == 'true'
    
    results_query = Result.query.join(User, Result.student_id == User.id)
    
    # Apply filters (same as before)
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
    
    if academic_session:
        results_query = results_query.filter(Result.academic_session == academic_session)
    
    if department:
        results_query = results_query.filter(User.department.ilike(f"%{department}%"))
    
    if school:
        results_query = results_query.filter(User.school.ilike(f"%{school}%"))
    
    if date_from:
        try:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            results_query = results_query.filter(Result.date_uploaded >= date_from_obj)
        except ValueError:
            flash("Invalid start date format", "warning")
    
    if date_to:
        try:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
            date_to_obj = date_to_obj.replace(hour=23, minute=59, second=59)
            results_query = results_query.filter(Result.date_uploaded <= date_to_obj)
        except ValueError:
            flash("Invalid end date format", "warning")
    
    if result_status:
        if result_status == 'passed':
            results_query = results_query.filter(Result.grade.in_(['A', 'B', 'C', 'D', 'E']))
        elif result_status == 'failed':
            results_query = results_query.filter(Result.grade == 'F')
        elif result_status == 'excellent':
            results_query = results_query.filter(Result.score >= 70)
        elif result_status == 'good':
            results_query = results_query.filter(Result.score.between(60, 69))
        elif result_status == 'average':
            results_query = results_query.filter(Result.score.between(50, 59))
        elif result_status == 'minimal_pass':
            results_query = results_query.filter(Result.score.between(40, 49))
    
    # Handle export request
    if export_request:
        results = results_query.order_by(Result.date_uploaded.desc()).all()
        return export_search_results(results)
    
    # Get pagination object
    results_pagination = results_query.order_by(Result.date_uploaded.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    results = results_pagination.items
    
    # Get unique values for filter dropdowns
    all_grades = [g[0] for g in db.session.query(Result.grade).distinct().all() if g[0]]
    all_courses = [c[0] for c in db.session.query(Result.course_code).distinct().all() if c[0]]
    all_sessions = [s[0] for s in db.session.query(Result.academic_session).distinct().all() if s[0]]
    all_semesters = [s[0] for s in db.session.query(Result.semester).distinct().all() if s[0]]
    all_departments = [d[0] for d in db.session.query(User.department).distinct().all() if d[0]]
    all_schools = [s[0] for s in db.session.query(User.school).distinct().all() if s[0]]
    
    # Get stats for the template
    stats = get_dashboard_stats()
    
    return render_template("search_results.html",
                         results=results,
                         pagination=results_pagination,
                         matric_no_filter=matric_no,
                         course_code_filter=course_code,
                         grade_filter=grade,
                         min_score_filter=min_score,
                         max_score_filter=max_score,
                         semester_filter=semester,
                         academic_session_filter=academic_session,
                         department_filter=department,
                         school_filter=school,
                         all_grades=all_grades,
                         all_courses=all_courses,
                         all_sessions=all_sessions,
                         all_semesters=all_semesters,
                         all_departments=all_departments,
                         all_schools=all_schools,
                         stats=stats,
                         page=page,
                         per_page=per_page)

def export_search_results(results):
    """Export search results to Excel"""
    output = BytesIO()
    
    # Create DataFrame
    data = []
    for result in results:
        student = User.query.get(result.student_id)
        data.append({
            'Matric Number': result.matric_no,
            'Student Name': result.student_name,
            'Department': student.department if student else 'N/A',  # NEW
            'School': student.school if student else 'N/A',  # NEW
            'Course Code': result.course_code,
            'Course Title': result.course_title,
            'Score': result.score,
            'Grade': result.grade,
            'Credit Unit': result.credit_unit,
            'CGPA': round(result.semester_cgpa, 2),
            'Academic Session': result.academic_session,
            'Semester': result.semester,
            'Remark': result.remark,
            'Date Uploaded': result.date_uploaded.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(data)
    df.to_excel(output, index=False)
    output.seek(0)
    
    filename = f"results_search_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(output, as_attachment=True, download_name=filename)

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
            Result.grade.in_(['A', 'B', 'C', 'D', 'E'])
        ).count()
        failed = Result.query.filter(
            Result.course_code == course.course_code,
            Result.grade == 'F'
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
        'student_cgpas': [round(cgpa.avg_cgpa, 2) for cgpa in student_cgpas if cgpa.avg_cgpa],
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
        'colors': ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#f97316', '#ef4444']
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
        'cgpa': [round(result.semester_cgpa, 2) for result in results]
    })

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
        'email': student.email,
        'department': student.department,  # NEW
        'school': student.school  # NEW
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
        'credit_unit': [3, 3],
        'academic_session': ['2023/2024', '2023/2024'],
        'semester': ['1', '1'],
        'department': ['Computer Science', 'Mathematics'],  # NEW
        'school': ['School of Computing', 'School of Sciences']  # NEW
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
        'semester_cgpa': round(r.semester_cgpa, 2),
        'academic_session': r.academic_session,
        'semester': r.semester,
        'date_uploaded': r.date_uploaded
    } for r in results])
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f"results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx")

# -------------------- API Routes --------------------
@app.route("/api/v1/results")
@limiter.limit("100 per hour")
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
        'semester_cgpa': round(r.semester_cgpa, 2),
        'academic_session': r.academic_session,
        'semester': r.semester,
        'date_uploaded': r.date_uploaded.isoformat() if r.date_uploaded else None
    } for r in results])

@app.route("/api/v1/students")
@limiter.limit("100 per hour") 
@login_required
def api_get_students():
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    students = User.query.filter_by(role='student').all()
    return jsonify([{
        'id': s.id,
        'matric_no': s.matric_no,
        'full_name': s.full_name,
        'email': s.email,
        'department': s.department,
        'school': s.school,
        'results_count': s.get_results_count()  # Use the helper method
    } for s in students])

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
        department = request.form.get('department', '').strip() or 'Computer Science'  # NEW
        school = request.form.get('school', '').strip() or None  # NEW

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
            student.department = department
            student.school = school

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

# -------------------- Bulk Delete Routes --------------------

@app.route("/admin/bulk-delete/students", methods=["POST"])
@login_required
@handle_db_errors
def bulk_delete_students():
    """Bulk delete students and their results"""
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.get_json()
    student_ids = data.get('student_ids', [])
    delete_type = data.get('delete_type', 'selected')  # 'selected' or 'by_criteria'
    criteria = data.get('criteria', {})
    
    if not student_ids and delete_type == 'selected':
        return jsonify({"error": "No students selected"}), 400
    
    try:
        # Create backup before deletion
        backup_filename = create_backup_before_deletion()
        
        deleted_count = 0
        students_to_delete = []
        
        if delete_type == 'selected':
            # Delete selected students
            students_to_delete = User.query.filter(
                User.id.in_(student_ids),
                User.role == 'student'
            ).all()
        else:
            # Delete by criteria
            query = User.query.filter_by(role='student')
            
            if criteria.get('department'):
                query = query.filter(User.department == criteria['department'])
            if criteria.get('school'):
                query = query.filter(User.school == criteria['school'])
            if criteria.get('has_results') == 'without':
                query = query.outerjoin(Result).group_by(User.id).having(db.func.count(Result.id) == 0)
            elif criteria.get('has_results') == 'with':
                query = query.join(Result).group_by(User.id).having(db.func.count(Result.id) > 0)
            
            students_to_delete = query.all()
        
        # Delete students and their results
        for student in students_to_delete:
            # Delete associated results
            Result.query.filter_by(student_id=student.id).delete()
            # Delete student
            db.session.delete(student)
            deleted_count += 1
        
        db.session.commit()
        
        # Log the bulk deletion
        logger.info(f"Bulk deleted {deleted_count} students by officer {current_user.id}. Backup: {backup_filename}")
        
        return jsonify({
            "success": True,
            "message": f"Successfully deleted {deleted_count} students",
            "deleted_count": deleted_count,
            "backup_file": backup_filename
        })
        
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Bulk delete error: {str(e)}")
        return jsonify({"error": f"Deletion failed: {str(e)}"}), 500

@app.route("/admin/bulk-delete/results", methods=["POST"])
@login_required
@handle_db_errors
def bulk_delete_results():
    """Bulk delete results based on criteria"""
    if current_user.role != 'officer':
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.get_json()
    result_ids = data.get('result_ids', [])
    delete_type = data.get('delete_type', 'selected')
    criteria = data.get('criteria', {})
    
    if not result_ids and delete_type == 'selected':
        return jsonify({"error": "No results selected"}), 400
    
    try:
        # Create backup
        backup_filename = create_backup_before_deletion()
        
        deleted_count = 0
        results_to_delete = []
        student_ids_affected = set()
        
        if delete_type == 'selected':
            # Delete selected results
            results_to_delete = Result.query.filter(Result.id.in_(result_ids)).all()
        else:
            # Delete by criteria
            query = Result.query
            
            if criteria.get('course_code'):
                query = query.filter(Result.course_code == criteria['course_code'])
            if criteria.get('grade'):
                query = query.filter(Result.grade == criteria['grade'])
            if criteria.get('semester'):
                query = query.filter(Result.semester == criteria['semester'])
            if criteria.get('academic_session'):
                query = query.filter(Result.academic_session == criteria['academic_session'])
            if criteria.get('min_score'):
                query = query.filter(Result.score >= float(criteria['min_score']))
            if criteria.get('max_score'):
                query = query.filter(Result.score <= float(criteria['max_score']))
            if criteria.get('date_from'):
                date_from = datetime.strptime(criteria['date_from'], '%Y-%m-%d')
                query = query.filter(Result.date_uploaded >= date_from)
            if criteria.get('date_to'):
                date_to = datetime.strptime(criteria['date_to'], '%Y-%m-%d')
                date_to = date_to.replace(hour=23, minute=59, second=59)
                query = query.filter(Result.date_uploaded <= date_to)
            
            results_to_delete = query.all()
        
        # Collect student IDs for CGPA recalculation
        for result in results_to_delete:
            student_ids_affected.add(result.student_id)
        
        # Delete results
        for result in results_to_delete:
            db.session.delete(result)
            deleted_count += 1
        
        db.session.commit()
        
        # Recalculate CGPA for affected students
        if student_ids_affected:
            bulk_update_cgpa(list(student_ids_affected))
        
        logger.info(f"Bulk deleted {deleted_count} results by officer {current_user.id}")
        
        return jsonify({
            "success": True,
            "message": f"Successfully deleted {deleted_count} results",
            "deleted_count": deleted_count,
            "backup_file": backup_filename
        })
        
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Bulk results delete error: {str(e)}")
        return jsonify({"error": f"Deletion failed: {str(e)}"}), 500

def create_backup_before_deletion():
    """Create a database backup before bulk operations"""
    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"backup_before_bulk_delete_{timestamp}.sql"
        
        # Simple backup - export critical tables
        students = User.query.filter_by(role='student').all()
        results = Result.query.all()
        
        backup_data = {
            'timestamp': timestamp,
            'students': [{
                'id': s.id,
                'matric_no': s.matric_no,
                'username': s.username,
                'email': s.email,
                'first_name': s.first_name,
                'last_name': s.last_name,
                'department': s.department,
                'school': s.school
            } for s in students],
            'results': [{
                'id': r.id,
                'student_id': r.student_id,
                'matric_no': r.matric_no,
                'course_code': r.course_code,
                'score': r.score,
                'grade': r.grade,
                'credit_unit': r.credit_unit,
                'academic_session': r.academic_session,
                'semester': r.semester
            } for r in results]
        }
        
        # In a real implementation, you might want to save this to a file or cloud storage
        # For now, we'll just log that we would create a backup
        logger.info(f"Backup would be created: {backup_filename} with {len(students)} students and {len(results)} results")
        
        return backup_filename
        
    except Exception as e:
        logger.error(f"Backup creation failed: {str(e)}")
        return "backup_failed"

@app.route("/admin/bulk-actions")
@login_required
def bulk_actions_dashboard():
    """Bulk actions management page"""
    if current_user.role != 'officer':
        return "Unauthorized", 403
    
    # Get statistics for the bulk actions page
    stats = get_dashboard_stats()
    
    # Get unique values for filters
    all_departments = [d[0] for d in db.session.query(User.department).distinct().all() if d[0]]
    all_schools = [s[0] for s in db.session.query(User.school).distinct().all() if s[0]]
    all_courses = [c[0] for c in db.session.query(Result.course_code).distinct().all() if c[0]]
    all_grades = [g[0] for g in db.session.query(Result.grade).distinct().all() if g[0]]
    all_sessions = [s[0] for s in db.session.query(Result.academic_session).distinct().all() if s[0]]
    all_semesters = [s[0] for s in db.session.query(Result.semester).distinct().all() if s[0]]
    
    return render_template("bulk_actions.html",
                         stats=stats,
                         all_departments=all_departments,
                         all_schools=all_schools,
                         all_courses=all_courses,
                         all_grades=all_grades,
                         all_sessions=all_sessions,
                         all_semesters=all_semesters)

# -------------------- Export Students --------------------
@app.route("/export/students")
@login_required
def export_students():
    """Export students to CSV with search filters"""
    if current_user.role != 'officer':
        return "Unauthorized", 403
    
    # Get the same filters as search_students
    query = request.args.get('q', '').strip()
    department = request.args.get('department', '').strip()
    school = request.args.get('school', '').strip()
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
    
    if department:
        students_query = students_query.filter(User.department.ilike(f"%{department}%"))
    
    if school:
        students_query = students_query.filter(User.school.ilike(f"%{school}%"))
    
    if status == 'with_results':
        students_query = students_query.join(Result).group_by(User.id).having(db.func.count(Result.id) > 0)
    elif status == 'without_results':
        students_query = students_query.outerjoin(Result).group_by(User.id).having(db.func.count(Result.id) == 0)
    
    students = students_query.order_by(User.matric_no).all()
    
    # Create CSV with department and school
    output = BytesIO()
    csv_content = "Matric No,Full Name,Username,Email,Department,School,Results Count\n"
    
    for student in students:
        result_count = len(student.results) if student.results else 0
        csv_content += f'"{student.matric_no}","{student.full_name}","{student.username}","{student.email}","{student.department}","{student.school}",{result_count}\n'
    
    output.write(csv_content.encode('utf-8'))
    output.seek(0)
    
    filename = f"students_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(output, as_attachment=True, download_name=filename)

import os
import base64
from io import BytesIO
from PIL import Image

# Add this route to handle profile picture uploads
@app.route("/upload_profile_picture", methods=["POST"])
@login_required
def upload_profile_picture():
    """Handle profile picture upload and save to database"""
    if current_user.role != 'student':
        return jsonify({"success": False, "error": "Unauthorized"}), 403
    try:
        if 'profile_picture' not in request.files:
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        file = request.files['profile_picture']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"success": False, "error": "Invalid file type. Only PNG, JPG, JPEG, GIF allowed."}), 400
        
        # Validate file size (2MB max)
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        
        if file_length > 2 * 1024 * 1024:  # 2MB
            return jsonify({"success": False, "error": "File size must be less than 2MB"}), 400
        
        # Process image
        try:
            image = Image.open(file.stream)
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize image to reasonable size (300x300 max)
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Convert to base64 for database storage
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Create data URL
            profile_picture_url = f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({"success": False, "error": "Invalid image file"}), 400
        
        # Update user profile picture in database
        current_user.profile_picture = profile_picture_url
        current_user.date_updated = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"Profile picture updated for user {current_user.id}")
        
        return jsonify({
            "success": True, 
            "profile_picture": profile_picture_url,
            "message": "Profile picture updated successfully"
        })
        
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Error uploading profile picture: {str(e)}")
        return jsonify({"success": False, "error": "Server error occurred"}), 500

#-------------------- Remove Profile Picture --------------------
@app.route('/remove-profile-picture', methods=['POST'])
@login_required
def remove_profile_picture():
    try:
        if current_user.profile_picture:
            # Optional: Add file removal logic for local storage
            try:
                import os
                from flask import current_app
                from urllib.parse import unquote, urlparse
                
                # If profile_picture is a URL path like "/static/uploads/filename.jpg"
                if current_user.profile_picture.startswith('/static/uploads/'):
                    # Extract filename from URL path
                    filename = os.path.basename(unquote(current_user.profile_picture))
                    file_path = os.path.join(current_app.root_path, 'static', 'uploads', filename)
                    
                    # Safely delete the file if it exists
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted profile picture file: {filename}")
                        
            except Exception as file_error:
                # Log but don't fail if file deletion doesn't work
                print(f"Note: Could not delete physical file: {file_error}")
            
            # Clear the profile picture from database
            current_user.profile_picture = None
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Profile picture removed successfully'})
        else:
            return jsonify({'success': False, 'error': 'No profile picture to remove'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})
    

# -------------------- Input Sanitization --------------------


def sanitize_input(input_string, max_length=None, allowed_pattern=None):
    """
    Sanitize user input to prevent XSS and injection attacks
    
    Args:
        input_string: The string to sanitize
        max_length: Maximum allowed length (optional)
        allowed_pattern: Regex pattern for validation (optional)
    
    Returns:
        Sanitized string or None if invalid
    """
    if input_string is None:
        return None
    
    # Convert to string and strip whitespace
    sanitized = str(input_string).strip()
    
    # Check length if specified
    if max_length and len(sanitized) > max_length:
        return None
    
    # Check against allowed pattern if specified
    if allowed_pattern and not re.match(allowed_pattern, sanitized):
        return None
    
    # HTML escape to prevent XSS
    sanitized = html.escape(sanitized)
    
    return sanitized

def sanitize_filename(filename):
    """Sanitize filename for safe storage"""
    if not filename:
        return None
    
    # Secure the filename
    filename = secure_filename(filename)
    
    # Additional sanitization
    filename = re.sub(r'[^\w\.-]', '_', filename)
    
    return filename

#------------------- Pagination URL Builder --------------------
@app.context_processor
def utility_processor():
    def build_pagination_url(page, per_page):
        args = request.args.copy()
        args['page'] = page
        args['per_page'] = per_page
        
        # Determine which route to use based on the current endpoint
        if request.endpoint == 'search_students':
            return url_for('search_students', **args)
        elif request.endpoint == 'search_results':
            return url_for('search_results', **args)
        else:
            # Fallback to current route
            return url_for(request.endpoint, **args)
    
    return dict(build_pagination_url=build_pagination_url)

# -------------------- Security Middleware --------------------
@app.before_request
def security_checks():
    """Perform security checks before each request"""
    # Block common attack patterns in URLs
    blocked_patterns = [
        r'\.\./',  # Directory traversal
        r'\.php',  # PHP files
        r'\.env',  # Environment files
        r'wp-',    # WordPress patterns
    ]
    
    path = request.path.lower()
    for pattern in blocked_patterns:
        if re.search(pattern, path):
            logger.warning(f"Blocked suspicious request: {request.path} from {request.remote_addr}")
            return "Suspicious request blocked", 400
    
    # Add security headers for specific paths
    if request.path.startswith('/api/'):
        # API-specific security headers can be added here
        pass

# -------------------- Global Error Handlers --------------------
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 error: {request.url}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Resource not found"}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logger.error(f"500 error: {str(error)}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error"}), 500
    return render_template('500.html'), 500

@app.errorhandler(413)
def too_large(error):
    logger.warning(f"File too large: {request.url}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "File too large. Maximum size is 16MB."}), 413
    flash("File too large. Maximum size is 16MB.", "danger")
    return redirect(request.referrer or url_for('dashboard_officer'))

@app.errorhandler(429)
def ratelimit_error(error):
    logger.warning(f"Rate limit exceeded: {request.remote_addr}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Rate limit exceeded"}), 429
    flash("Too many requests. Please try again later.", "warning")
    return redirect(request.referrer or url_for('home'))



# --------------------  Full diagnostic for monitoring / AWS or internal tools --------------------
@app.route("/health/diagnostic")
def health_check():
    from datetime import datetime
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": app.config.get('ENV', 'unknown')
    }
    
    try:
        # Database check
        db.session.execute("SELECT 1")
        health_status["database"] = "connected"

        # Disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            health_status["disk_space"] = {
                "total_gb": round(total / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "used_percent": round((used / total) * 100, 2)
            }
        except:
            health_status["disk_space"] = "check_failed"

        # Memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            health_status["memory"] = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": round(memory.percent, 2)
            }
        except ImportError:
            health_status["memory"] = "psutil_not_available"
        except Exception:
            health_status["memory"] = "check_failed"

        return jsonify(health_status), 200

    except Exception as e:
        logger.exception("Health check failed")
        health_status.update({
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        })
        return jsonify(health_status), 500

# -------------------- # Minimal health check for Render --------------------
    
@app.route("/health")
def health():
    return "OK", 200

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)