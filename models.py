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
    other_names = db.Column(db.String(150), nullable=True)  # optional middle/other names
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'student' or 'officer'
    matric_no = db.Column(db.String(50), unique=True, nullable=True)  # only for students
    photo = db.Column(db.String(100))
    must_change_password = db.Column(db.Boolean, default=False)
    department = db.Column(db.String(100), default="Computer Science")

    # Relationship: all results of this student
    results = db.relationship(
        'Result',
        backref='student',
        lazy=True,
        cascade="all, delete-orphan"
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
    semester_cgpa = db.Column(db.Float, default=0.0)
    date_uploaded = db.Column(db.DateTime, default=datetime.utcnow)

    # Who uploaded this result (User.id of officer)
    uploaded_by = db.Column(db.Integer, nullable=False)

    def update_matric_no(self):
        """Sync matric_no with the student record if changed."""
        if self.student:
            self.matric_no = self.student.matric_no

    def update_student_name(self):
        """Sync student_name with the student's full name."""
        if self.student:
            self.student_name = self.student.full_name
