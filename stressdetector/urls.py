from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login_view, name='login'),
    path('predict_stress/', views.predict_stress, name='predict_stress'),
    path('result/', views.result, name='result'),
]
