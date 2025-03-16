import streamlit as st
import sqlite3
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import hashlib
import pandas as pd

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # Create Users Table with Role
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user'
        )
    """)

    # Create Scores Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            username TEXT,
            score REAL
        )
    """)

    # Create Answer Key Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS answer_key (
            question INTEGER PRIMARY KEY,
            correct_option INTEGER
        )
    """)

    # Ensure admin user exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                  ("admin", hash_password("admin123"), "admin"))
    
    conn.commit()
    conn.close()

# Function to register a new user
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Function to check login credentials
def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[0] == hash_password(password):
        return user[1]  # Return role (admin/user)
    return None

# Function to get the answer key
def get_answer_key():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT question, correct_option FROM answer_key")
    answer_key = dict(c.fetchall())
    conn.close()
    return answer_key

# Initialize database
init_db()

# Streamlit UI
st.title("🔐 Login to OMR Scanner")

menu = ["Login", "Register"]
if "logged_in" in st.session_state:
    menu = ["OMR Scanner", "Leaderboard", "Logout"]
    if st.session_state["role"] == "admin":
        menu.insert(1, "Admin Panel")  # Admin gets extra option

choice = st.sidebar.selectbox("Menu", menu)

if choice == "Login":
    st.subheader("📥 User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        role = authenticate_user(username, password)
        if role:
            st.success(f"Welcome {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.rerun()
        else:
            st.error("Invalid username or password")

elif choice == "Register":
    st.subheader("📝 Register New User")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        if register_user(new_username, new_password):
            st.success("Registration successful! You can now login.")
        else:
            st.error("Username already exists. Try another one.")

elif choice == "Admin Panel" and st.session_state.get("role") == "admin":
    st.title("🔑 Admin Panel - Set Answer Key")
    num_questions = st.number_input("Enter the number of questions:", min_value=1, step=1)
    answer_key = {i: st.selectbox(f"Correct answer for Question {i+1}", options=[0, 1, 2, 3, 4], key=f"q{i}") for i in range(num_questions)}
    
    if st.button("Save Answer Key"):
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("DELETE FROM answer_key")  # Clear previous answer key
        for q, ans in answer_key.items():
            c.execute("INSERT INTO answer_key (question, correct_option) VALUES (?, ?)", (q, ans))
        conn.commit()
        conn.close()
        st.success("Answer key updated successfully!")

elif choice == "OMR Scanner":
    st.title("📄 OMR Scanner and Test Grader")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="🖼 Original Image", use_container_width=True)

        # Process Image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        st.image(edged, caption="📌 Edge Detected Image", use_container_width=True)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        docCnt = None

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    docCnt = approx
                    break

        if docCnt is not None:
            paper = four_point_transform(image, docCnt.reshape(4, 2))
            st.image(paper, caption="📃 Warped Paper (Top-Down View)", use_container_width=True)

            # Extract and grade OMR
            answer_key = get_answer_key()
            correct = 0
            total_questions = len(answer_key)

            for i in range(total_questions):
                marked_answer = np.random.choice([0, 1, 2, 3, 4])  # Simulated detection
                if answer_key.get(i) == marked_answer:
                    correct += 1

            score = (correct / total_questions) * 100
            st.success(f"✅ Your Score: {score:.2f}%")

            # Store score
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("INSERT INTO scores (username, score) VALUES (?, ?)", (st.session_state["username"], score))
            conn.commit()
            conn.close()
        else:
            st.error("❌ No valid document detected!")

elif choice == "Leaderboard":
    st.title("🏆 Leaderboard - Student Scores")
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username, score FROM scores ORDER BY score DESC")
    records = c.fetchall()
    conn.close()

    df = pd.DataFrame(records, columns=["Username", "Score (%)"])
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("No scores available yet.")

elif choice == "Logout":
    st.session_state.clear()
    st.rerun()
