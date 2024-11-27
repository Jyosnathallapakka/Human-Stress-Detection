import os
import mysql.connector
import mysql.connector as sql
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.http import HttpResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout

from django.contrib import messages
from django.conf import settings
import pickle
from .forms import SignupForm, StressForm
from .models import User
from .ml_models.train_ann import model, scaler  # Ensure paths are correct
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import json
from .models import StressModel 
# Load the trained ANN model and scaler

# def StressModel(input_data):

#     input_array = np.array([input_data])
#     input_scaled = scaler.transform(input_array)  # Scale the input
#     prediction = model.predict(input_scaled)[0][0]  # First prediction

#     return "Stressed" if prediction > 0.3 else "Not Stressed"


# Index page
def index(request):
    return render(request, 'stressdetector/index.html')


# Signup page
@csrf_protect
def signup(request):
    if request.method == "POST":
        us = request.POST.get('username')
        em = request.POST.get('email')
        ps = request.POST.get('password')
        cpass = request.POST.get('cpassword')

        errors = {}
        if ps != cpass:
            errors['cpassword_error'] = "Passwords do not match!"

        conn = None
        cursor = None
        try:
            conn = sql.connect(
                host="127.0.0.1",
                user="root",
                password="Jyosna@520316",
                database="humanstress"
            )
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM signup WHERE username = %s OR email = %s"
            cursor.execute(query, (us, em))
            if cursor.fetchone()[0] > 0:
                errors['duplicate_error'] = "Username or email already exists!"
                return render(request, 'stressdetector/signup.html', {'errors': errors})

            if not errors:
                comm = "INSERT INTO signup (username, email, password, cpassword) VALUES (%s, %s, %s, %s)"
                cursor.execute(comm, (us, em, ps, cpass))
                conn.commit()
                # messages.success(request, "Account created successfully!")
                return redirect('login')

        except sql.Error as e:
            messages.error(request, f"Error: {e}")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    return render(request, 'stressdetector/signup.html')


from django.shortcuts import render, redirect
@csrf_protect
def login_view(request):
    if request.user.is_authenticated:  # Redirect logged-in users to predict_stress
        return redirect('predict_stress')
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        conn = None
        cursor = None
        try:
            conn = sql.connect(
                host="127.0.0.1",
                user="root",
                password="Jyosna@520316",
                database="humanstress"
            )
            cursor = conn.cursor()
            query = "SELECT * FROM signup WHERE username = %s AND password = %s"
            cursor.execute(query, (username, password))
            user = cursor.fetchone()
            if user:
                # messages.success(request, "✅ Successfully logged in!")
                return redirect('predict_stress')
            else:
                # messages.error(request, "❌Invalid username or password.")
                return render(request, 'stressdetector/login.html', {'error_message': "Invalid username or password."})

        except sql.Error as e:
            return render(request, 'stressdetector/login.html', {'error_message': f"An error occurred: {e}"})

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    return render(request, 'stressdetector/login.html')




# from django.http import HttpResponse, HttpResponseBadRequest
# from django.shortcuts import render
import mysql.connector
import os
import mysql.connector
from django.shortcuts import render, redirect
from django.conf import settings
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# Database connection function
def connect_to_database():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Jyosna@520316",
        database="humanstress"
    )

# Predict stress function
def predict_stress(request):
    if request.method == "POST":
        try:
            # 1. Extract Form Data
            snoring_rate = float(request.POST.get("snoring_rate", 0))
            respiratory_rate = float(request.POST.get("respiratory_rate", 0))
            body_temperature = float(request.POST.get("body_temperature", 0))
            limb_movement = float(request.POST.get("limb_movement", 0))
            blood_oxygen = float(request.POST.get("blood_oxygen", 0))
            eye_movements = float(request.POST.get("eye_movements", 0))
            sleep_hours = float(request.POST.get("sleep_hours", 0))
            heart_rate = float(request.POST.get("heart_rate", 0))

            # 2. Load Model and Scaler
            model_path = os.path.join(settings.BASE_DIR, 'stressdetector/ml_models/model.h5')
            scaler_path = os.path.join(settings.BASE_DIR, 'stressdetector/ml_models/scaler.pkl')
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            # 3. Prepare Data for Prediction
            input_data = np.array([[snoring_rate, respiratory_rate, body_temperature,
                                    limb_movement, blood_oxygen, eye_movements,
                                    sleep_hours, heart_rate]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            is_stressed = "Stressed" if prediction[0][0] > 0.5 else "Not Stressed"

            # 4. Store Data in MySQL
            conn = connect_to_database()
            cursor = conn.cursor()
            query = """
                INSERT INTO health_data (snoring_rate, respiratory_rate, body_temperature,
                                         limb_movement, blood_oxygen, eye_movements,
                                         sleep_hours, heart_rate, is_stressed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (snoring_rate, respiratory_rate, body_temperature, limb_movement,
                      blood_oxygen, eye_movements, sleep_hours, heart_rate, is_stressed)
            cursor.execute(query, values)
            conn.commit()

            # 5. Close Database Connection
            cursor.close()
            conn.close()

            # 6. Pass Prediction Result to Result Page
            request.session['is_stressed'] = is_stressed
            return redirect("result")

        except ValueError as e:
            return render(request, "stressdetector/predict_stress.html", {'error_message': f"Invalid input: {e}"})
        except mysql.connector.Error as e:
            return render(request, "stressdetector/predict_stress.html", {'error_message': f"Database error: {e}"})
        except Exception as e:
            return render(request, "stressdetector/predict_stress.html", {'error_message': f"Unexpected error: {e}"})

    return render(request, "stressdetector/predict_stress.html")

# @login_required
def result(request):
    # Fetch the prediction from session
    is_stressed = request.session.get('is_stressed', 'Unknown')
    return render(request, 'stressdetector/result.html', {'is_stressed': is_stressed})