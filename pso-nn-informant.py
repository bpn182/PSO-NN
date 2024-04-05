from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import random

# Mount Google Drive and load the dataset (Size: 168MB, Total Columns: 284807)
drive.mount("/content/drive")
file_path = "/content/drive/My Drive/Datasets/creditcard.csv"
data = pd.read_csv(file_path)

# Printing the length and first few rows of the dataset
print(len(data))
print(data.head())


# Separating the features and target variable
X = data.drop("Class", axis=1)
y = data["Class"]

# Spliting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class NeuralNetwork:
    """
    Class for creating and training the Neural Network.

    Attributes:
    num_neurons (int): The number of neurons in the hidden layer.
    dropout_rate (float): The dropout rate for regularization.
    learning_rate (float): The learning rate for the optimizer.
    activation (str): The activation function to use in the hidden layer.
    model (Sequential): The Keras model representing the neural network.
    """

    def __init__(self, num_neurons, dropout_rate, learning_rate, activation="relu"):
        self.num_neurons = num_neurons
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation = activation
        self.model = self.create_model()

    def create_model(self):
        model = Sequential(
            [
                Dense(
                    self.num_neurons,
                    input_dim=X_train.shape[1],
                    activation=self.activation,
                ),
                Dropout(self.dropout_rate),
                Dense(64, activation=self.activation),
                Dropout(self.dropout_rate),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def train(self):
        # Initialize an EarlyStopping callback to stop training when the validation accuracy stops improving.
        early_stopping = EarlyStopping(
            monitor="val_accuracy", patience=4, verbose=1, restore_best_weights=True
        )

        # Train the model with early stopping
        # Due to large dataset, we will use a batch size of 1024 it will make the training faster
        history = self.model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=1024,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopping],
        )
        return history

    # Evaluate the performance of the model on the test set.
    def evaluate(self):
        y_pred = self.model.predict(X_test)
        y_pred = y_pred > 0.5
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


class Particle:
    """
    Represents a particle in the Particle Swarm Optimization (PSO) algorithm.

    Each particle represents a set of hyperparameters for a neural network model,
    including the number of neurons, dropout rate, learning rate, and activation
    function.

    Attributes:
        num_neurons (int): The number of neurons in the neural network.
        dropout_rate (float): The dropout rate for the neural network.
        learning_rate (float): The learning rate for the neural network.
        activation_index (int): The index of the activation function to use.
        activation (str): The name of the activation function to use.
        position (numpy.ndarray): The current position of the particle in the
            hyperparameter search space.
        velocity (numpy.ndarray): The current velocity of the particle in the
            hyperparameter search space.
        best_position (numpy.ndarray): The best position the particle has found
            so far.
        best_score (float): The best score the particle has achieved so far.
        model (NeuralNetwork): The neural network model associated with the
            particle's hyperparameters.
        informants (list): A list of other particles that this particle can
            learn from (its informants).
    """

    def __init__(self, bounds, activation_functions):
        self.num_neurons = int(np.random.uniform(bounds[0][0], bounds[0][1]))
        self.dropout_rate = np.random.uniform(bounds[1][0], bounds[1][1])
        self.learning_rate = np.random.uniform(bounds[2][0], bounds[2][1])
        self.activation_index = np.random.randint(len(activation_functions))
        self.activation = activation_functions[self.activation_index]
        self.position = np.array(
            [
                self.num_neurons,
                self.dropout_rate,
                self.learning_rate,
                self.activation_index,
            ]
        )
        self.velocity = np.random.uniform(-1, 1, size=4)
        self.best_position = self.position.copy()
        self.best_score = 0
        self.model = NeuralNetwork(
            self.num_neurons, self.dropout_rate, self.learning_rate
        )
        self.informants = []

    # Method to update the velocity of the particle based on the global best position and the informant's best positions.
    def update_velocity(self, global_best_position, w=0.7, c1=1.4, c2=1.4):

        # Generating random numbers for the cognitive, social, and informant components
        r1 = np.random.uniform(0, 1, size=4)
        r2 = np.random.uniform(0, 1, size=4)
        r3 = np.random.uniform(0, 1, size=4)

        # Calculate the cognitive and social velocities
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)

        informant_best_position = self.position
        best_score = 0

        # Iterate over all informants
        for informant in self.informants:
            activation_function = activation_functions[int(informant.best_position[3])]
            # Create a new model with the informant's best position
            model = NeuralNetwork(
                int(informant.best_position[0]),
                informant.best_position[1],
                informant.best_position[2],
                activation_function,
            )

            # Train the model with the informant's best position
            model.train()

            # Evaluate the model at the informant's best position
            score = model.evaluate()

            # If the score is better than the best score, update the best position and score
            if score > best_score:
                informant_best_position = informant.best_position
                best_score = score
                print(f"Best informant position found: {informant_best_position}")

        informant_velocity = c2 * r3 * (informant_best_position - self.position)
        self.velocity = (
            w * self.velocity
            + cognitive_velocity
            + social_velocity
            + informant_velocity
        )

    # Update the position of the particle based on the velocity
    def update_position(self, bounds):
        self.position += self.velocity

        # Making sure that the values of the hyperparameters are within the specified bounds
        self.num_neurons = int(np.clip(self.position[0], bounds[0][0], bounds[0][1]))
        self.dropout_rate = np.clip(self.position[1], bounds[1][0], bounds[1][1])
        self.learning_rate = np.clip(self.position[2], bounds[2][0], bounds[2][1])
        self.activation_index = int(
            np.clip(self.position[3], 0, len(activation_functions) - 1)
        )
        self.activation = activation_functions[self.activation_index]
        self.position = np.array(
            [
                self.num_neurons,
                self.dropout_rate,
                self.learning_rate,
                self.activation_index,
            ]
        )
        self.model = NeuralNetwork(
            self.num_neurons, self.dropout_rate, self.learning_rate, self.activation
        )

    # Update the best position and score of the particle
    def update_score(self):
        self.model.train()
        self.score = self.model.evaluate()
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_position = self.position.copy()


class Swarm:
    """
    Swarm class represents the group of particle in PSO.

    Attributes:
    particles (list): All the particles in the swarm.
    global_best_score (float): The best score of any particle in the swarm.
    global_best_position (numpy array): The position of the particle achieving the best score.
    activation_functions (list): The activation functions available for the neural network.
    """

    def __init__(self, num_particles, bounds, activation_functions, num_informants):
        self.particles = [
            Particle(bounds, activation_functions) for _ in range(num_particles)
        ]
        self.global_best_score = 0
        self.global_best_position = self.particles[0].position
        self.activation_functions = activation_functions

        # Assign informants to each particle
        for particle in self.particles:
            particle.informants = random.sample(
                [p for p in self.particles if p != particle], num_informants
            )

    # Update the global best score and position based on the best score of any particle in the swarm
    def update_global_best(self):
        for particle in self.particles:
            print("particle best", particle.best_score, self.global_best_score)
            if particle.best_score > self.global_best_score:
                self.global_best_score = particle.best_score
                self.global_best_position = particle.best_position

    # Move the particles towards the best position in the swarm
    def move_particles(self, bounds):
        for particle in self.particles:
            particle.update_velocity(self.global_best_position)
            particle.update_position(bounds)
            particle.update_score()


# Defining the hyperparameters and activation functions
iteration = 2
num_particles = 4
num_informants = 1
neurons_range = [10, 200]
dropout_range = [0.1, 0.5]
learning_rate_range = [1e-4, 1e-2]
bounds = [neurons_range, dropout_range, learning_rate_range]
activation_functions = ["relu", "tanh"]

# Creating the swarm
swarm = Swarm(
    num_particles=num_particles,
    bounds=bounds,
    activation_functions=activation_functions,
    num_informants=num_informants,
)


# Running the PSO algorithm for a specified number of iterations
for i in range(iteration):
    swarm.update_global_best()
    swarm.move_particles(bounds)
    print(f"Iteration {i+1}/{2}")
    print("Current best score: ", swarm.global_best_score)
    print("Current best position: ", swarm.global_best_position)

print("Best hyperparameters found:")
print("Number of neurons: ", int(swarm.global_best_position[0]))
print("Dropout rate: ", swarm.global_best_position[1])
print("Learning rate: ", swarm.global_best_position[2])
print(
    "Best activation function: ",
    activation_functions[int(swarm.global_best_position[3])],
)


# Train the model with the best hyperparameters found by PSO
nn = NeuralNetwork(
    int(swarm.global_best_position[0]),
    swarm.global_best_position[1],
    swarm.global_best_position[2],
    activation_functions[int(swarm.global_best_position[3])],
)


# Train the model
history = nn.train()

# Evaluate the model
accuracy = nn.evaluate()
print(f"Test Accuracy: {accuracy:.2f}")


# Get the predictions on the test set
y_pred = nn.model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Print the confusion matrix for model prediction
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Saving the trained model
nn.save("PSO-NN.h5")