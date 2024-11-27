from django.db import models

class User(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=50)
    password = models.CharField(max_length=15)
    cpassword = models.CharField(max_length=100,default='')

    class Meta:
        db_table = 'stressdetector_user'

    def __str__(self):
        return self.username
class StressModel(models.Model):
    snoring_rate = models.FloatField()
    respiratory_rate = models.FloatField()
    body_temperature = models.FloatField()
    limb_movement = models.FloatField()
    blood_oxygen = models.FloatField()
    eye_movements = models.FloatField()
    sleep_hours = models.FloatField()
    heart_rate = models.FloatField()

    def __str__(self):
        return f"Stress Model {self.id}"
