# Generated by Django 5.1.3 on 2024-11-16 15:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('stressdetector', '0005_alter_user_table'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='cpassword',
        ),
    ]