import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# For Model Building
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import warnings
# from ml_models.train_ann import predict_stress

# Suppress the Matplotlib GUI warning
warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread will likely fail.")

import matplotlib
matplotlib.use('Agg') 



# Load the dataset
df = pd.read_csv(r"C:\Users\jyosn\OneDrive\Desktop\Book2.csv")
df.head(10)

# Feature columns (X) and target column (y)
X = df.drop(columns=['Stress level']) # Drop 'Stress Levels' as it's the target
y = df['Stress level'] # Target

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape

# Scalling the features


scaler = StandardScaler()

# Fit scaler on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # Use transform, not fit_transform for the test data

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaler for later use (for prediction)
joblib.dump(scaler, 'scaler.pkl')

X_train_scaled.head(5)

# Initialize the ANN model
model = Sequential()

# Adding input layer and the first hidden layer (neurons=64, activation='relu')
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))

# Adding more hidden layers (neurons=32, activation='relu')
model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=1, activation='sigmoid')) # 2 classes for stress levels (0-1)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Training The Model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled,y_test))

model.save(r"C:/Users/jyosn/OneDrive/Desktop/Human Stress Detection/humanstressdetection/ml_models/model.h5")

# Plot training & validation accuracy and loss over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluating the MODEL
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_accuracy*100: .2f}')
print(f'Test Loss: {test_loss*100: .2f}')
loaded_scaler = joblib.load('scaler.pkl')

# Make Prediction new Unseen Data
new_data = np.array([[60,18,70,8,97,60,9,75]]) # Replace With actual new Data
new_data_scaled = loaded_scaler.transform(new_data)
loaded_model = keras.models.load_model(r"C:/Users/jyosn/OneDrive/Desktop/Human Stress Detection/humanstressdetection/ml_models/model.h5") 
prediction = loaded_model.predict(new_data_scaled)


#Output Prediction in Words.
if prediction > 0.5:
    print("Stressed")

else:
    print("Not Stressed")

# Save the model

# # For inference: Load the saved model and scaler
# loaded_model = keras.models.load_model('model.h5')
# loaded_scaler = joblib.load('scaler.pkl')

# # Make Prediction on new unseen data
# new_data = np.array([[60,18,70,8,97,60,9,75]]) # Replace with actual new data
# new_data_scaled = loaded_scaler.transform(new_data)  # Transform using the loaded scaler

# prediction = loaded_model.predict(new_data_scaled)




# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
# print(f'Test Accuracy: {test_accuracy*100:.2f}')
# print(f'Test Loss: {test_loss:.2f}')

# def predict_stress(input_data):
#     """
#     Function to predict stress level based on input data
#     :param input_data: A 2D array of input features to be predicted
#     :return: Stress prediction ('Stressed' or 'Not Stressed')
#     """
#     try:
#         # Load the saved model and scaler
#         model = keras.models.load_model('model.h5')
#         scaler = joblib.load('scaler.pkl')

#         # Scale the input data using the loaded scaler
#         input_scaled = scaler.transform(input_data)  # input_data should be a 2D array

#         # Make the prediction using the model
#         prediction = model.predict(input_scaled)

#         # Return the prediction as a label ('Stressed' or 'Not Stressed')
#         stress_status = 'Stressed' if prediction[0][0] > 0.5 else 'Not Stressed'

#         return stress_status

#     except Exception as e:
#         raise ValueError(f"Error in prediction: {e}")