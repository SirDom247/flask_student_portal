from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(db.Model, UserMixin):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), nullable=False)
    other_names = db.Column(db.String(150), nullable=True)  
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False) 
    matric_no = db.Column(db.String(50), unique=True, nullable=True)
    photo = db.Column(db.String(100))
    profile_picture = db.Column(db.Text, nullable=True)
    must_change_password = db.Column(db.Boolean, default=False)
    department = db.Column(db.String(100), default="Computer Science")
    
    # NEW FIELDS
    school = db.Column(db.String(100), nullable=True)  # Faculty/School
    requires_password_reset = db.Column(db.Boolean, default=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    date_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship: all results of this student - IMPROVED
    results = db.relationship(
        'Result',
        backref=db.backref('student_ref', lazy=True),  # Changed backref name to avoid conflict
        lazy=True,
        cascade="all, delete-orphan",
        order_by="Result.date_uploaded.desc()"  # Added ordering
    )

    def get_id(self):
        """Use the primary key for Flask-Login."""
        return str(self.id)

    @property
    def full_name(self):
        """Return full name including other names if present."""
        if self.other_names:
            return f"{self.first_name} {self.other_names} {self.last_name}"
        return f"{self.first_name} {self.last_name}"

    def get_results_count(self):
        """Helper method to get results count."""
        return len(self.results) if self.results else 0


class Result(db.Model):
    __tablename__ = "result"

    id = db.Column(db.Integer, primary_key=True)

    # Foreign key to User.id for referential integrity
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # Store matric_no for easier search/display without joining
    matric_no = db.Column(db.String(50), nullable=False)

    student_name = db.Column(db.String(200), nullable=True)  # full name from User
    course_code = db.Column(db.String(20), nullable=False)
    course_title = db.Column(db.String(100), nullable=False)
    credit_unit = db.Column(db.Integer, nullable=False, default=3)
    score = db.Column(db.Float, nullable=False)
    grade = db.Column(db.String(2))
    remark = db.Column(db.String(10))
    academic_session = db.Column(db.String(20), nullable=False, default="2024/2025")
    semester = db.Column(db.String(10), nullable=False, default="First")  # e.g., "First", "Second"
    semester_cgpa = db.Column(db.Float, default=0.0)
    date_uploaded = db.Column(db.DateTime, default=datetime.utcnow)

    # Who uploaded this result (User.id of officer)
    uploaded_by = db.Column(db.Integer, nullable=False)

    # Indexes for better performance - ADD THESE
    __table_args__ = (
        db.Index('idx_matric_no', 'matric_no'),
        db.Index('idx_course_code', 'course_code'),
        db.Index('idx_academic_session', 'academic_session'),
        db.Index('idx_semester', 'semester'),
        db.Index('idx_date_uploaded', 'date_uploaded'),
    )

    def update_matric_no(self):
        """Sync matric_no with the student record if changed."""
        if self.student_ref:  # Changed from self.student to self.student_ref
            self.matric_no = self.student_ref.matric_no

    def update_student_name(self):
        """Sync student_name with the student's full name."""
        if self.student_ref:  # Changed from self.student to self.student_ref
            self.student_name = self.student_ref.full_name

    def to_dict(self):
        """Convert result to dictionary for API responses."""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'matric_no': self.matric_no,
            'student_name': self.student_name,
            'course_code': self.course_code,
            'course_title': self.course_title,
            'score': self.score,
            'grade': self.grade,
            'credit_unit': self.credit_unit,
            'academic_session': self.academic_session,
            'semester': self.semester,
            'semester_cgpa': self.semester_cgpa,
            'date_uploaded': self.date_uploaded.isoformat() if self.date_uploaded else None
        }