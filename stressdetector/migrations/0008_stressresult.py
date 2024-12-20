# Generated by Django 5.1.3 on 2024-11-22 18:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stressdetector', '0007_user_cpassword'),
    ]

    operations = [
        migrations.CreateModel(
            name='StressResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('snoring_rate', models.FloatField()),
                ('respiratory_rate', models.FloatField()),
                ('body_temperature', models.FloatField()),
                ('limb_movements', models.FloatField()),
                ('blood_oxygen', models.FloatField()),
                ('eye_movement', models.FloatField()),
                ('sleep_hours', models.FloatField()),
                ('heart_rate', models.FloatField()),
                ('stress_level', models.FloatField()),
            ],
        ),
    ]
