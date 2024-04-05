from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

# Mount Google Drive and load the dataset (Size: 168MB, Total Columns: 284807)
drive.mount("/content/drive")
file_path = "/content/drive/My Drive/Datasets/creditcard.csv"
df = pd.read_csv(file_path)

# Printing the length and first few rows of the dataset
print(len(df))
print(df.head())

# Removing the 'Class' column because it's the target variable we want to predict
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to build and compile a Keras model for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input',
                                 min_value=32,
                                 max_value=512,
                                 step=32),
                    input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(hp.Float('dropout_input', 0, 0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_hidden',
                                 min_value=32,
                                 max_value=512,
                                 step=32),
                    activation='relu'))
    model.add(Dropout(hp.Float('dropout_hidden', 0, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

# Initialize a Keras Tuner RandomSearch instance to optimize the hyperparameters of the model
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='bipin-pc',
    project_name='creditfraudtuner')

early_stopping = EarlyStopping(monitor='val_loss', patience=5)


# Start the hyperparameter search by fitting the tuner on the training data, with early stopping
tuner.search(X_train, y_train,
             epochs=20,
             validation_data=(X_test, y_test),
             callbacks=[early_stopping])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters
for hp in best_hps.values.keys():
    print(f"{hp}: {best_hps.get(hp)}")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=20, batch_size=1000, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Confusion Matrix and Classification Report
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Saving the trained model
model.save('NN-KerasTuner.h5')