# 🎓 Student Result Portal

A modern **Student Result Management System** built with **Flask**, **PostgreSQL**, and **Tailwind CSS**.  
This portal enables **Exam Officers** to upload results in Excel format while allowing **Students** to securely access, view, and download their individual results as PDF/CSV.  

It also computes and tracks each student’s **semester CGPA** automatically.

---

## 🚀 Features

### 👨‍🏫 Exam Officer
- Secure **registration and login**
- Upload student results via **Excel file**
- Automatically assigns grades and remarks based on scores
- View analytics (number of students, results uploaded, pending uploads)

### 👩‍🎓 Student
- Secure **registration and login**
- View only their **own results**
- See **detailed results** with:
  - Course Code  
  - Course Title  
  - Score  
  - Grade (A–F)  
  - Remark (Pass/Fail)
- View **computed CGPA** for the semester
- Download results as **PDF/CSV**
- Upload and display **profile photos**

### ⚙️ Backend Features
- **Role-based authentication** (Student / Officer)
- **PostgreSQL** database integration
- **File uploads** (Excel, Images)
- **Flask-Migrate** for database versioning
- **Responsive UI** powered by Tailwind CSS
- **Secure password hashing** using Werkzeug
- **PDF/CSV export** using `pandas`

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML5, Tailwind CSS |
| **Backend** | Python (Flask) |
| **Database** | PostgreSQL |
| **ORM** | SQLAlchemy |
| **Authentication** | Flask-Login |
| **File Processing** | Pandas, OpenPyXL |
| **Deployment** | Render.com (compatible with Heroku and VPS) |

---

## 📂 Project Structure

![Project Structure](project_structure.png)


## 🧠 Installation & Setup
Follow these steps to set up the project locally:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/student-result-portal.git
cd student-result-portal

2️⃣ Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Database
Create a PostgreSQL database:
CREATE DATABASE student_portal_db;
Then, edit your config.py:


SQLALCHEMY_DATABASE_URI = 'postgresql://username:password@localhost:5432/student_portal_db'
5️⃣ Initialize Database
flask db init
flask db migrate
flask db upgrade


6️⃣ Run the App
python app.py
The app should now be live at 👉 http://127.0.0.1:5000

🧾 Excel Upload Format
Exam Officers should upload results in .xlsx format with the following column headers:
MatricNo	CourseCode	CourseTitle	Score
STU001	CSC101	Introduction to Computing	85
STU002	MTH101	Calculus I	72

The system will automatically:
Assign grades (A–F)
Mark remarks (Pass/Fail)
Compute the student’s CGPA

📈 CGPA Computation Logic
Grade	Score Range	Grade Point
A	70–100	5
B	60–69	4
C	50–59	3
D	45–49	2
E	40–44	1
F	< 40	0

CGPA = (Sum of Grade Points) / (Number of Courses)

🌟 Screenshots (Preview)
Add screenshots of student dashboard, officer dashboard, and result table here.

☁️ Deployment (Render.com)
Push your project to GitHub

Create a new Web Service on Render

Connect your GitHub repo

Set:

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

Add environment variables:

SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://username:password@hostname:port/dbname
Render will automatically deploy and host your Flask app online 🌍

🧩 Future Enhancements
Add email verification and password reset

Enable semester-by-semester result views

Include GPA trend visualization

Integrate admin dashboard analytics

Add support for bulk student registration via CSV

👨‍💻 Author
Dominic Udousoro
Cybersecurity Researcher & Software Engineer
📧 dominicudousoro@gmail.com
🌐 www.rigeltechkonsuktz.com
💼 www.github.com/SirDom247
💼 www.linkedin.com/in/dominic-udousoro-88245576
