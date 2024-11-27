# forms.py
from django import forms
from .models import User 
from django.contrib.auth.models import User
class StressForm(forms.Form):
    snoring_rate = forms.FloatField(label='Snoring Rate', min_value=46, max_value=98)
    respiratory_rate = forms.FloatField(label='Respiratory Rate', min_value=15, max_value=30)
    body_temperature = forms.FloatField(label='Body Temperature', min_value=86, max_value=98)
    limb_movements = forms.FloatField(label='Limb Movements', min_value=5, max_value=18)
    blood_oxygen = forms.FloatField(label='Blood Oxygen', min_value=84, max_value=96)
    eye_movement = forms.FloatField(label='Eye Movement', min_value=60, max_value=105)
    sleep_hours = forms.FloatField(label='Sleep Hours', min_value=0, max_value=9)
    heart_rate = forms.FloatField(label='Heart Rate', min_value=52, max_value=84)


class SignupForm(forms.ModelForm):
    
        cpassword = forms.CharField(widget=forms.PasswordInput)
        class Meta:
            model = User  # Link the form to the User model
            fields = ['username', 'email', 'password', 'cpassword']  # Define the fields that the form will use


        def clean(self):
            cleaned_data = super().clean()
            password = cleaned_data.get('password')
            cpassword = cleaned_data.get('cpassword')

            # Check if passwords match
            if password and cpassword and password != cpassword:
                raise forms.ValidationError("Passwords do not match.")
            
            return cleaned_data


class LoginForm(forms.Form):
    username = forms.CharField(max_length=255)
    password = forms.CharField(widget=forms.PasswordInput)

from django.db import models

class StressResult(models.Model):
    snoring_rate = models.FloatField()
    respiratory_rate = models.FloatField()
    body_temperature = models.FloatField()
    limb_movements = models.FloatField()
    blood_oxygen = models.FloatField()
    eye_movement = models.FloatField()
    sleep_hours = models.FloatField()
    heart_rate = models.FloatField()
    stress_level = models.FloatField()  # You can add a stress level field if you plan to calculate it

    def __str__(self):
        return f"Stress Result for {self.snoring_rate} snoring rate"


        


        
