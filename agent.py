import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam, SGD
from random import random, randint, sample
import csv
from data import LossPlot

pd.set_option('display.max_columns', None)

loss_plot = LossPlot()

# Hyperparameters
UNITS = 100
ALPHA = 0.005
GAMMA = 0.995
TAU = 0.3
LAMBDA = 5
SAMPLE_PROPORTION = 0.4


# Z-score normalisation function
def normalise(vector):
    scaler = StandardScaler()
    normalized_vector = scaler.fit_transform(vector)
    return normalized_vector


# Calculate the distance between the snake's head and nearest body segment
def distance_to_body(snake):
    min_distance = float('inf')
    for segment in snake.segments[1:]:
        distance = snake.head.distance(segment)
        if distance < min_distance:
            min_distance = distance
    return min_distance


# Create the experience vector
def experience_vector(state, action, reward, next_state, done):
    experience = np.empty(0)
    for i in range(10):
        experience = np.append(experience, state[i])
    experience = np.append(experience, action)
    experience = np.append(experience, reward)
    for i in range(10):
        experience = np.append(experience, next_state[i])
    experience = np.append(experience, done)
    return experience


# Sample a mini batch from the total memory buffer
def sample_mini_batch():

    memory_buffer = r'C:\Users\arvin\PycharmProjects\Snake\experience_history.csv'

    with open(memory_buffer, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        num_observations = len(rows)
        sample_size = SAMPLE_PROPORTION*num_observations

    mini_batch = sample(rows, int(sample_size))
    mini_batch = pd.DataFrame(mini_batch)

    return mini_batch


class Agent:
    def __init__(self):
        self.action_set = [0, 1, 2, 3]  # 0 = up, 1 = left, 2 = down, 3 = right
        self.current_state = np.zeros(15)

        self.current_reward = 0  # Reward of current run
        self.next_action = -1  # Initial empty action

        self.LAMBDA = 0.01

        # Creating a sequential model (As opposed to a Functional API for simplicity)
        self.q_network = Sequential([
            Input(shape=10),
            Dense(units=UNITS, activation="relu"),
            Dense(units=UNITS, activation="relu"),
            Dense(units=UNITS, activation="relu"),
            Dense(units=UNITS, activation="relu"),
            Dense(units=4, activation="linear")
        ])

        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=ALPHA)

        # Create the target network
        self.target_q_network = Sequential([
            Input(shape=10),
            Dense(units=UNITS, activation="relu"),
            Dense(units=UNITS, activation="relu"),
            Dense(units=UNITS, activation="relu"),
            Dense(units=UNITS, activation="relu"),
            Dense(units=4, activation="linear")
        ])

        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=ALPHA)

    # Calculate the current state
    def state(self, snake, food):

        position_x, position_y = snake.head.pos()

        distance_right_wall = abs(7 - position_x)
        distance_left_wall = abs(7 - position_x)
        distance_top_wall = abs(7 - position_y)
        distance_bottom_wall = abs(7 - position_y)

        position_food_x, position_food_y = food.pos()
        distance_to_food = snake.head.distance(food)

        distance_to_own_body = distance_to_body(snake)

        self.current_state = [
            position_x,
            position_y,
            distance_right_wall,
            distance_left_wall,
            distance_top_wall,
            distance_bottom_wall,
            position_food_x,
            position_food_y,
            distance_to_food,
            distance_to_own_body
        ]

        return self.current_state

    # Compute the current loss
    def compute_loss(self, mini_batch, gamma, experience_count):
        states = mini_batch['state']
        actions = mini_batch['action']
        rewards = mini_batch['reward']
        next_states = mini_batch['next_state']
        done_vals = mini_batch['done']

        batch_experiences = [states, actions, rewards, next_states, done_vals]

        states = [eval(row) for row in states]
        actions = [eval(row) for row in actions]
        next_states = [eval(row) for row in next_states]
        rewards = [eval(row) for row in rewards]
        done_vals = [eval(row) for row in done_vals]

        # batch_experiences = (np.array(batch_experiences) for experience in batch_experiences)
        for i in range(len(batch_experiences)):
            batch_experiences[i] = np.array(batch_experiences[i])

        states = normalise(states)
        next_states = normalise(next_states)

        max_q_sa = tf.reduce_max(self.target_q_network(next_states), axis=-1)
        y_targets = rewards + (gamma * max_q_sa * done_vals)

        q_values = self.q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))

        loss = MSE(y_targets, q_values)
        loss_plot.append_loss(float(loss), experience_count)

        return loss

    # Perform soft updates to the target network for stability
    def soft_update(self, tau):
        updated_variables = []
        for q_variable, target_variable in zip(self.q_network.trainable_variables,
                                               self.target_q_network.trainable_variables):
            updated_variable = tau * q_variable + (1 - tau) * target_variable
            updated_variables.append(updated_variable)

        self.q_network.set_weights(updated_variables)

    # Perform a step of gradient descent
    def learn(self, mini_batch, gamma, experience_count):
        with tf.GradientTape() as tape:
            cost = self.compute_loss(mini_batch, gamma, experience_count)
        gradients_q = tape.gradient(cost, self.q_network.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients_q, self.q_network.trainable_variables))
        self.soft_update(TAU)

    # Perform epsilon-greedy exploration
    def epsilon_greedy(self, epsilon):
        p = random()
        if p < epsilon:
            self.next_action = randint(0, 3)
        else:
            current_state_batch = np.expand_dims(self.current_state, axis=0)
            prediction = self.q_network.predict(current_state_batch, verbose=0)
            q_max_index = np.argmax(prediction)
            self.next_action = int(q_max_index)

        return self.next_action

    # Save the model
    def save_model(self):
        tf.keras.Model.save(self.q_network, r'C:\Users\arvin\PycharmProjects\Snake\q_network.keras')
        tf.keras.Model.save(self.target_q_network, r'C:\Users\arvin\PycharmProjects\Snake\target_q_network.keras')

