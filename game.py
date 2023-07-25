from turtle import Screen, setworldcoordinates
from snake import Snake
from food import Food
from scoreboard import Scoreboard
from agent import Agent
from epoch import Epoch
from data import MemoryBuffer, LossPlot, RewardPlot
from agent import sample_mini_batch
import time

screen = Screen()
screen.setup(width=600, height=600)
screen.bgcolor("black")
screen.title("Snake")
screen.tracer(0)
snake = Snake()
food = Food()
scoreboard = Scoreboard()
epoch = Epoch()
memory_buffer = MemoryBuffer()
loss_plot = LossPlot()
reward_plot = RewardPlot()

MAX_EPISODES = 1000
GAMMA = 0.80

agent = Agent()


# Reset game function
def reset_game():
    scoreboard.reset()
    scoreboard.__init__()

    food.reset()
    food.__init__()

    snake.reset()
    snake.clear_segments()
    snake.__init__()

    agent.current_reward = 0
    epoch.increase_epoch()


# Storing the current experience
def experience_store(current_state, epsilon, current_reward):

    next_action_index = agent.epsilon_greedy(epsilon)
    perform = snake.action_list[next_action_index]

    if scoreboard.game_is_on != 0:
        perform()

    current_state_plus_1 = agent.state(snake, food)

    experience = {
        'state': current_state,
        'action': next_action_index,
        'reward': current_reward,
        'next_state': current_state_plus_1,
        'done': scoreboard.game_is_on
    }

    memory_buffer.append_experience(experience)
    reward = 0

    return current_state_plus_1, reward


# Save the total reward to file
def reward_store(total_reward):
    reward_plot.append_reward(total_reward, epoch.current_epoch)
    total_reward = 0
    return total_reward


# Perform one step of gradient descent
def update(experience_count):
    mini_batch = sample_mini_batch()
    agent.learn(mini_batch, GAMMA, experience_count)
    steps_since_update_check = 0
    return steps_since_update_check


# Function for our game
def game():

    # Initialize the screen
    screen.listen()
    # screen.onkey(snake.turn_upward, "Up")
    # screen.onkey(snake.turn_left, "Left")
    # screen.onkey(snake.turn_downward, "Down")
    # screen.onkey(snake.turn_right, "Right")

    # Initialize constants
    epsilon = 0.95

    # Initialize variables
    steps_since_update_check = 0
    experience_count = 0
    current_reward = 0
    total_reward = 0

    game_is_on = True
    while game_is_on:
        state = agent.state(snake, food)  # initialize
        for t in range(100):

            # Begin snake's continuous movement
            screen.update()
            time.sleep(0.1)
            snake.move()

            # Reduce the grid to a 15x15 coordinate system
            setworldcoordinates(-7.5, -7.5, 7.5, 7.5)

            # Establish reward system and game mechanics
            if snake.head.distance(food) < 0.5:
                current_reward = 0.05
                total_reward += 0.05
                scoreboard.increase_score()
                snake.extend()
                food.refresh()

            if 1 > snake.head.distance(food) >= 0.05:
                current_reward = 0.002
                total_reward += 0.005

            if 3 > snake.head.distance(food) >= 1:
                current_reward = 0.001
                total_reward += 0.002

            if snake.head.xcor() < -7 or snake.head.xcor() > 7 \
                    or snake.head.ycor() < -7 or snake.head.ycor() > 7:
                current_reward = -1
                total_reward += -1
                total_reward = reward_store(total_reward)
                state_plus_1, current_reward = experience_store(state, epsilon, current_reward)
                state = state_plus_1
                print(f"Epoch {epoch.current_epoch}: Final Score = {scoreboard.current_score}, "
                      f"Epsilon = {round(epsilon + 0.005, 2)}")
                scoreboard.game_over()
                time.sleep(2)
                reset_game()

            for segment in snake.segments[1:]:  # You can return a list in reverse order using [::-1]
                if snake.head.distance(segment) < 0.25:
                    current_reward = -1
                    total_reward += -1
                    total_reward = reward_store(total_reward)
                    state_plus_1, current_reward = experience_store(state, epsilon, current_reward)
                    state = state_plus_1
                    print(f"Epoch {epoch.current_epoch}: Final Score = {scoreboard.current_score}, Epsilon = {epsilon}")
                    scoreboard.game_over()
                    time.sleep(2)
                    reset_game()

            # Store the current experience
            state_plus_1, current_reward = experience_store(state, epsilon, current_reward)
            state = state_plus_1
            steps_since_update_check += 1
            experience_count += 1

            # Perform a gradient descent step
            if steps_since_update_check >= 4:
                steps_since_update_check = update(experience_count)

        # Decrease the epsilon-greedy exploration probability
        if epsilon >= 0:
            epsilon -= 0.005

        if epoch.current_epoch >= 50000:
            break

    # Save model and metrics after learning period
    agent.save_model()
    loss_plot.save_plot()
    reward_plot.save_plot()

    screen.exitonclick()
    return agent.q_network


network = game()

for var in network.trainable_variables:
    print(var.name, var.shape)
    print(var.numpy())







