import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

memory_buffer = 'experience_history.csv'  # In the same working directory as 'game', so relative file path can be used
loss_history = 'loss_history.csv'
reward_history = 'reward_history.csv'
fieldnames_experiences = ['state', 'action', 'reward', 'next_state', 'done']
fieldnames_loss = ['Loss', 'Experiences']
fieldnames_reward = ['Reward', 'Epoch']


class MemoryBuffer:
    def __init__(self):
        with open(memory_buffer, 'w', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=fieldnames_experiences)
            self.writer.writeheader()

    def append_experience(self, experience):
        with open(memory_buffer, 'a', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=fieldnames_experiences)
            self.writer.writerow(experience)


class LossPlot:
    def __init__(self):
        self.loss_history = []
        self.experience_count = []
        with open(loss_history, 'w', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=fieldnames_loss)
            self.writer.writeheader()
        self.loss_data = pd.read_csv(loss_history)
        self.figure_loss = plt.figure(1)
        self.plot = FuncAnimation(self.figure_loss, self.plot_loss, 1000)
        plt.tight_layout()
        #plt.show(block=False)

    def append_loss(self, loss, experience_count):
        data = {'Loss': loss, 'Experiences': experience_count}
        with open(loss_history, 'a', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=fieldnames_loss)
            self.writer.writerow(data)

    def plot_loss(self, *args):
        self.loss_data = pd.read_csv(loss_history)
        x_experiences = self.loss_data['Experiences']
        y_loss = self.loss_data['Loss']
        plt.cla()

        plt.title('Loss History', fontdict={"fontname": "Times New Roman", "fontsize": 16})
        plt.xlabel('Experiences', fontdict={"fontname": "Times New Roman", "fontsize": 12})
        plt.ylabel('Loss', fontdict={"fontname": "Times New Roman", "fontsize": 12})

        plt.plot(x_experiences, y_loss, label="Loss History")

    def save_plot(self):
        plt.savefig(fname=r'C:\Users\arvin\PycharmProjects\Snake\loss_plot.png', dpi=100)


class RewardPlot:
    def __init__(self):
        self.reward_history = []
        self.epoch_count = []
        with open(reward_history, 'w', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=fieldnames_reward)
            self.writer.writeheader()
        self.reward_data = pd.read_csv(reward_history)
        self.figure_reward = plt.figure(2)
        self.plot = FuncAnimation(self.figure_reward, self.plot_reward, 30000)
        plt.tight_layout()
        plt.show(block=False)

    def append_reward(self, reward, epoch_count):
        data = {'Reward': reward, 'Epoch': epoch_count}
        with open(reward_history, 'a', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=fieldnames_reward)
            self.writer.writerow(data)

    def plot_reward(self, *args):
        self.reward_data = pd.read_csv(reward_history)
        x_epoch = self.reward_data['Epoch']
        y_reward = self.reward_data['Reward']
        plt.cla()

        plt.title('Reward History', fontdict={"fontname": "Times New Roman", "fontsize": 16})
        plt.xlabel('Epochs', fontdict={"fontname": "Times New Roman", "fontsize": 12})
        plt.ylabel('Reward', fontdict={"fontname": "Times New Roman", "fontsize": 12})

        plt.plot(x_epoch, y_reward, label="Reward History")

    def save_plot(self):
        plt.savefig(fname=r'C:\Users\arvin\PycharmProjects\Snake\loss_plot.png', dpi=100)