# import streamlit as st
# import cv2
# import numpy as np
# import imutils
# from imutils.perspective import four_point_transform
# from imutils import contours
# import random

# # Function to display images in Streamlit
# def show_images(images, titles):
#     for index, image in enumerate(images):
#         st.image(image, caption=titles[index], use_column_width=True)

# # ANSWER_KEY mapping, similar to what you've already defined
# ANSWER_KEY = {
#     0: 1,
#     1: 4,
#     2: 0,
#     3: 3,
#     4: 1
# }

# # Streamlit UI components
# st.title("OMR Scanner and Test Grader")
# st.write("Upload the image of the OMR sheet to get the grade.")

# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Convert the uploaded image to OpenCV format
#     image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#     # Edge detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 75, 200)
#     st.image(edged, caption="Edge Detected Image", use_column_width=True)

#     # Find contours
#     cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     docCnt = None

#     # Drawing contours for visualization
#     allContourImage = image.copy()
#     cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
#     st.image(allContourImage, caption="Contours Detected", use_column_width=True)

#     # Find the document contour
#     if len(cnts) > 0:
#         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#         for c in cnts:
#             peri = cv2.arcLength(c, closed=True)
#             approx = cv2.approxPolyDP(c, epsilon=peri * 0.02, closed=True)
#             if len(approx) == 4:
#                 docCnt = approx
#                 break

#     # Getting the bird's eye view (top view)
#     paper = four_point_transform(image, docCnt.reshape(4, 2))
#     warped = four_point_transform(gray, docCnt.reshape(4, 2))
#     st.image(paper, caption="Warped Paper", use_column_width=True)
#     st.image(warped, caption="Warped Gray Image", use_column_width=True)

#     # Threshold the document
#     thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     st.image(thresh, caption="Threshold Image", use_column_width=True)

#     # Finding contours in the threshold image
#     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     questionCnts = []

#     # Filtering contours for questions
#     for c in cnts:
#         (x, y, w, h) = cv2.boundingRect(c)
#         ar = w / float(h)
#         if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
#             questionCnts.append(c)

#     # Sorting question contours and checking answers
#     questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
#     correct = 0
#     questionsContourImage = paper.copy()

#     for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
#         cnts = contours.sort_contours(questionCnts[i: i + 5])[0]
#         cv2.drawContours(questionsContourImage, cnts, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
#         bubbled = None

#         for (j, c) in enumerate(cnts):
#             mask = np.zeros(thresh.shape, dtype="uint8")
#             cv2.drawContours(mask, [c], -1, 255, -1)
#             mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#             total = cv2.countNonZero(mask)

#             if bubbled is None or total > bubbled[0]:
#                 bubbled = (total, j)

#         color = (0, 0, 255)
#         k = ANSWER_KEY[q]

#         if k == bubbled[1]:
#             color = (0, 255, 0)
#             correct += 1

#         cv2.drawContours(paper, [cnts[k]], -1, color, 3)

#     st.image(questionsContourImage, caption="Contours with Colored Answers", use_column_width=True)

#     # Final score calculation
#     score = (correct / 5.0) * 100
#     st.write(f"Score: {score:.2f}%")

#     # Display final image with score
#     cv2.putText(paper, f"Score: {score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#     st.image(paper, caption="Final Result with Score", use_column_width=True)


# import streamlit as st
# import sqlite3
# import bcrypt
# import cv2
# import numpy as np
# import imutils
# from imutils.perspective import four_point_transform
# from imutils import contours
# import random
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyzbar.pyzbar import decode
# from reportlab.pdfgen import canvas
# import qrcode

# # Database setup
# conn = sqlite3.connect("students.db", check_same_thread=False)
# cursor = conn.cursor()

# # Create tables
# cursor.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)""")
# cursor.execute("""CREATE TABLE IF NOT EXISTS scores (username TEXT, score REAL)""")
# conn.commit()

# # Hashing functions
# def hash_password(password):
#     return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# def verify_password(password, hashed_password):
#     return bcrypt.checkpw(password.encode(), hashed_password.encode())

# # Function to display images in Streamlit
# def show_image(image, title):
#     """Helper function to display images with Streamlit"""
#     st.image(image, caption=title, use_container_width=True)

# # Streamlit UI
# st.title("📝 OMR Scanner and Test Grader")
# st.sidebar.title("🔐 Login / Register")

# # Authentication state
# if "authenticated" not in st.session_state:
#     st.session_state.authenticated = False
#     st.session_state.username = ""

# # Register function
# def register():
#     st.subheader("Create a New Account")
#     new_user = st.text_input("Username")
#     new_pass = st.text_input("Password", type="password")
    
#     if st.button("Register"):
#         hashed_pass = hash_password(new_pass)
#         try:
#             cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, hashed_pass))
#             conn.commit()
#             st.success("🎉 Account Created! Please log in.")
#         except sqlite3.IntegrityError:
#             st.error("⚠ Username already exists! Choose another.")

# # Login function
# def login():
#     st.subheader("Login to Your Account")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
    
#     if st.button("Login"):
#         cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
#         result = cursor.fetchone()
        
#         if result and verify_password(password, result[0]):
#             st.session_state.authenticated = True
#             st.session_state.username = username
#             st.success(f"✅ Welcome, {username}!")
#         else:
#             st.error("❌ Incorrect Username or Password!")

# # Logout function
# def logout():
#     st.session_state.authenticated = False
#     st.session_state.username = ""

# # Sidebar options
# auth_option = st.sidebar.radio("Select", ["Login", "Register", "QR Code Login"])

# if not st.session_state.authenticated:
#     if auth_option == "Login":
#         login()
#     elif auth_option == "Register":
#         register()
#     elif auth_option == "QR Code Login":
#         uploaded_qr = st.file_uploader("Upload QR Code Image", type=["png", "jpg", "jpeg"])
#         if uploaded_qr:
#             image = cv2.imdecode(np.frombuffer(uploaded_qr.read(), np.uint8), cv2.IMREAD_COLOR)
#             decoded_objects = decode(image)
#             if decoded_objects:
#                 qr_data = decoded_objects[0].data.decode("utf-8")
#                 cursor.execute("SELECT username FROM users WHERE username=?", (qr_data,))
#                 if cursor.fetchone():
#                     st.session_state.authenticated = True
#                     st.session_state.username = qr_data
#                     st.success(f"✅ QR Login Successful! Welcome, {qr_data}")
#                 else:
#                     st.error("⚠ QR Code is not registered in the system.")
#             else:
#                 st.error("❌ No QR Code found in the image.")
# else:
#     st.sidebar.write(f"👤 Logged in as **{st.session_state.username}**")
#     if st.sidebar.button("Logout"):
#         logout()

# # OMR Processing (Only if logged in)
# if st.session_state.authenticated:
#     st.subheader("📤 Upload OMR Sheet for Grading")
#     uploaded_file = st.file_uploader("Choose an OMR image", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#     # Convert the uploaded image to OpenCV format
#         image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#     # Edge detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 75, 200)
#     st.image(edged, caption="Edge Detected Image", use_column_width=True)

#     # Find contours
#     cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     docCnt = None

#     # Drawing contours for visualization
#     allContourImage = image.copy()
#     cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
#     st.image(allContourImage, caption="Contours Detected", use_column_width=True)

#     # Find the document contour
#     if len(cnts) > 0:
#         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#         for c in cnts:
#             peri = cv2.arcLength(c, closed=True)
#             approx = cv2.approxPolyDP(c, epsilon=peri * 0.02, closed=True)
#             if len(approx) == 4:
#                 docCnt = approx
#                 break

#     # Getting the bird's eye view (top view)
#     paper = four_point_transform(image, docCnt.reshape(4, 2))
#     warped = four_point_transform(gray, docCnt.reshape(4, 2))
#     st.image(paper, caption="Warped Paper", use_column_width=True)
#     st.image(warped, caption="Warped Gray Image", use_column_width=True)

#     # Threshold the document
#     thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     st.image(thresh, caption="Threshold Image", use_column_width=True)

#     # Finding contours in the threshold image
#     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     questionCnts = []

#     # Filtering contours for questions
#     for c in cnts:
#         (x, y, w, h) = cv2.boundingRect(c)
#         ar = w / float(h)
#         if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
#             questionCnts.append(c)

#     # Sorting question contours and checking answers
#     questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
#     correct = 0
#     questionsContourImage = paper.copy()

#     for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
#         cnts = contours.sort_contours(questionCnts[i: i + 5])[0]
#         cv2.drawContours(questionsContourImage, cnts, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
#         bubbled = None

#         for (j, c) in enumerate(cnts):
#             mask = np.zeros(thresh.shape, dtype="uint8")
#             cv2.drawContours(mask, [c], -1, 255, -1)
#             mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#             total = cv2.countNonZero(mask)

#             if bubbled is None or total > bubbled[0]:
#                 bubbled = (total, j)

#         color = (0, 0, 255)
#         k = ANSWER_KEY[q]

#         if k == bubbled[1]:
#             color = (0, 255, 0)
#             correct += 1

#         cv2.drawContours(paper, [cnts[k]], -1, color, 3)

#     st.image(questionsContourImage, caption="Contours with Colored Answers", use_column_width=True)

#     # Final score calculation
#     score = (correct / 5.0) * 100
#     st.write(f"Score: {score:.2f}%")

#     # Display final image with score
#     cv2.putText(paper, f"Score: {score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#     st.image(paper, caption="Final Result with Score", use_column_width=True)

    

#             # Store score
#             cursor.execute("INSERT INTO scores (username, score) VALUES (?, ?)", (st.session_state.username, score))
#             conn.commit()

#     # Leaderboard
#     st.subheader("🏆 Leaderboard - Top 5 Students")
#     cursor.execute("SELECT username, MAX(score) FROM scores GROUP BY username ORDER BY MAX(score) DESC LIMIT 5")
#     results = cursor.fetchall()
#     if results:
#         df = pd.DataFrame(results, columns=["Student", "Highest Score"])
#         st.dataframe(df)


#main....


# import streamlit as st
# import sqlite3
# import cv2
# import numpy as np
# import imutils
# from imutils.perspective import four_point_transform
# from imutils import contours
# import hashlib

# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # Database setup
# def init_db():
#     conn = sqlite3.connect("users.db")
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             username TEXT UNIQUE,
#             password TEXT
#         )
#     """)
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS scores (
#             username TEXT,
#             score REAL
#         )
#     """)
#     conn.commit()
#     conn.close()

# # Function to register a new user
# def register_user(username, password):
#     conn = sqlite3.connect("users.db")
#     c = conn.cursor()
#     try:
#         c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
#         conn.commit()
#         return True
#     except sqlite3.IntegrityError:
#         return False
#     finally:
#         conn.close()

# # Function to check login credentials
# def authenticate_user(username, password):
#     conn = sqlite3.connect("users.db")
#     c = conn.cursor()
#     c.execute("SELECT password FROM users WHERE username = ?", (username,))
#     user = c.fetchone()
#     conn.close()
#     if user and user[0] == hash_password(password):
#         return True
#     return False

# # Initialize database
# init_db()

# # Streamlit UI
# st.title("🔐 Login to OMR Scanner")

# menu = ["Login", "Register"]
# choice = st.sidebar.selectbox("Menu", menu)

# if choice == "Login":
#     st.subheader("📥 User Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if authenticate_user(username, password):
#             st.success(f"Welcome {username}!")
#             st.session_state["logged_in"] = True
#             st.session_state["username"] = username
#         else:
#             st.error("Invalid username or password")

# elif choice == "Register":
#     st.subheader("📝 Register New User")
#     new_username = st.text_input("Choose a Username")
#     new_password = st.text_input("Choose a Password", type="password")
#     if st.button("Register"):
#         if register_user(new_username, new_password):
#             st.success("Registration successful! You can now login.")
#         else:
#             st.error("Username already exists. Try another one.")

# # Check if user is logged in before showing OMR grader
# if "logged_in" in st.session_state and st.session_state["logged_in"]:
#     st.title("📄 OMR Scanner and Test Grader")
#     uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#         st.image(image, caption="🖼 Original Image", use_container_width=True)
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edged = cv2.Canny(blurred, 75, 200)
#         st.image(edged, caption="📌 Edge Detected Image", use_container_width=True)
        
#         # Find contours
#         cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         docCnt = None
        
#         if len(cnts) > 0:
#             cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#             for c in cnts:
#                 peri = cv2.arcLength(c, True)
#                 approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#                 if len(approx) == 4:
#                     docCnt = approx
#                     break
        
#         if docCnt is not None:
#             paper = four_point_transform(image, docCnt.reshape(4, 2))
#             warped = four_point_transform(gray, docCnt.reshape(4, 2))
#             st.image(paper, caption="📃 Warped Paper (Top-Down View)", use_container_width=True)
            
#             thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#             st.image(thresh, caption="⚪ Thresholded Image", use_container_width=True)
            
#             # Find contours for questions
#             cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             cnts = imutils.grab_contours(cnts)
#             questionCnts = [c for c in cnts if cv2.boundingRect(c)[2] >= 20 and cv2.boundingRect(c)[3] >= 20]
            
#             # Sort question contours
#             questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
#             correct = 0
            
#             # Grading the OMR Sheet
#             ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 2}  # Example answer key
#             for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
#                 cnts = contours.sort_contours(questionCnts[i: i + 5])[0]
#                 bubbled = max([(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=cv2.drawContours(np.zeros(thresh.shape, dtype="uint8"), [c], -1, 255, -1)))), j] for j, c in enumerate(cnts))
#                 color = (0, 0, 255)
#                 if ANSWER_KEY.get(q, -1) == bubbled[1]:
#                     color = (0, 255, 0)
#                     correct += 1
#                 cv2.drawContours(paper, [cnts[ANSWER_KEY.get(q, 0)]], -1, color, 3)
            
#             score = (correct / len(ANSWER_KEY)) * 100
#             cv2.putText(paper, f"Score: {score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#             st.image(paper, caption=f"✅ Graded OMR Sheet - Score: {score:.2f}%")
            
#             # Store score in database
#             conn = sqlite3.connect("users.db")
#             c = conn.cursor()
#             c.execute("INSERT INTO scores (username, score) VALUES (?, ?)", (st.session_state["username"], score))
#             conn.commit()
#             conn.close()
            
#             # Display all login students' details
#             st.subheader("📜 All Student Scores")
#             conn = sqlite3.connect("users.db")
#             c = conn.cursor()
#             c.execute("SELECT username, score FROM scores")
#             records = c.fetchall()
#             conn.close()
#             for record in records:
#                 st.write(f"{record[0]}: {record[1]:.2f}%")



#main 2;;;;


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
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            username TEXT,
            score REAL
        )
    """)
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
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[0] == hash_password(password):
        return True
    return False

# Initialize database
init_db()

# Streamlit UI
st.title("🔐 Login to OMR Scanner")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Login":
    st.subheader("📥 User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success(f"Welcome {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
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

# Check if user is logged in before showing OMR grader
if "logged_in" in st.session_state and st.session_state["logged_in"]:
    st.title("📄 OMR Scanner and Test Grader")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="🖼 Original Image", use_container_width=True)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        st.image(edged, caption="📌 Edge Detected Image", use_container_width=True)
        
        # Find contours
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            warped = four_point_transform(gray, docCnt.reshape(4, 2))
            st.image(paper, caption="📃 Warped Paper (Top-Down View)", use_container_width=True)
            
            thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            st.image(thresh, caption="⚪ Thresholded Image", use_container_width=True)
            
            # Find contours for questions
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            questionCnts = [c for c in cnts if cv2.boundingRect(c)[2] >= 20 and cv2.boundingRect(c)[3] >= 20]
            
            # Sort question contours
            questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
            correct = 0
            
            # Grading the OMR Sheet
            ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 2}  # Example answer key
            for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
                cnts = contours.sort_contours(questionCnts[i: i + 5])[0]
                bubbled = max([(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=cv2.drawContours(np.zeros(thresh.shape, dtype="uint8"), [c], -1, 255, -1)))), j] for j, c in enumerate(cnts))
                color = (0, 0, 255)
                if ANSWER_KEY.get(q, -1) == bubbled[1]:
                    color = (0, 255, 0)
                    correct += 1
                cv2.drawContours(paper, [cnts[ANSWER_KEY.get(q, 0)]], -1, color, 3)
            
            score = (correct / len(ANSWER_KEY)) * 100
            cv2.putText(paper, f"Score: {score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            st.image(paper, caption=f"✅ Graded OMR Sheet - Score: {score:.2f}%")
            
            # Store score in database
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("INSERT INTO scores (username, score) VALUES (?, ?)", (st.session_state["username"], score))
            conn.commit()
            conn.close()
            
            # Display all students' scores
            st.subheader("🏆 Leaderboard - Student Scores")
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT username, score FROM scores ORDER BY score DESC")
            records = c.fetchall()
            conn.close()
            
            df = pd.DataFrame(records, columns=["Username", "Score (%)"])
            def highlight_top(val):
                if val >= 90:
                    return "background-color: #28a745; color: white;"
                elif val >= 75:
                    return "background-color: #ffc107; color: black;"
                return ""

            if not df.empty:
                st.dataframe(df.style.applymap(highlight_top, subset=["Score (%)"]))
            else:
                st.info("No scores available yet.")






#########

