import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from queue import Queue


class GridWorldEnvGUI(gym.Env):
    metadata = {'render.modes': ['console', 'human']}

    def __init__(self, grid_size=6):
        super(GridWorldEnvGUI, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(grid_size), spaces.Discrete(grid_size)))
        self.state = None
        self.figure, self.axis = plt.subplots()
        self.walls = {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)}
        self.goal = (0, grid_size - 1)
        self.goalreached = False
        self.additional_goal = (4, 0)
        self.additionalgoalreached = False
        self.finalgoalreached = False
        self.box = (1, 1)
        self.box_initial = (0, 1)
        self.finalgoal = self.box_initial
        self.box_intact = True
        self.box_interacted = False
        self.is_running = True
        self.move_box_count = 0
        self.break_box_count = 0
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def step(self, action):
        x, y = self.state
        new_x, new_y = x, y

        if action == 0:
            new_x = max(x - 1, 0)
        elif action == 1:
            new_x = min(x + 1, self.grid_size - 1)
        elif action == 2:
            new_y = max(y - 1, 0)
        elif action == 3:
            new_y = min(y + 1, self.grid_size - 1)
        if self.goalreached == False:
            reward = -1
            if (new_x, new_y) not in self.walls:
                reward = -1
                if (new_x, new_y) == self.box:
                    # print("reached box")
                    self.box_interacted = True
                    if self.move_box(new_x, new_y):
                        self.state = (new_x, new_y)
                    else:
                        self.break_box()
                        reward = -5
                else:
                    self.state = (new_x, new_y)
            done = self.state == self.goal
            if done:
                self.goalreached = True
                reward += 25
                print("the intermediate goal state is reached")
            return self.state, reward, self.goalreached, {}

        if self.goalreached == True and self.finalgoalreached == False:
            reward = -1
            if (new_x, new_y) not in self.walls:
                reward = -1
                self.state = (new_x, new_y)
                done2 = self.state == self.box_initial
                if done2:
                    print("reached box finally")

                    self.finalgoalreached = True
                    reward += 30
            return self.state, reward, self.goalreached, {}

        else:
            reward = -1
            if (new_x, new_y) not in self.walls:
                reward = -1
                self.state = (new_x, new_y)
                done3 = self.state == self.additional_goal
                if done3:
                    print("reached end goal state finally")

                    self.additionalgoalreached = True
                    reward += 35
        return self.state, reward, self.goalreached, {}

    def move_box(self, new_x, new_y):
        if self.box_intact:
            best_distance = 20000
            best_position = (75, 75)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = new_x + dx, new_y + dy
                if (next_x, next_y) not in self.walls and next_x < self.grid_size and next_y < self.grid_size and (next_x, next_y) != self.state and next_x > -1 and next_y > -1 and self.box != (0, self.grid_size-1) and self.box != (self.grid_size-1, 0) and self.box != (0, 0) and self.box != (self.grid_size-1, self.grid_size-1):
                    if (next_x, next_y) != self.goal:
                        distance = abs(
                            next_x - self.goal[0]) + abs(next_y - self.goal[1])
                        if distance < best_distance:
                            best_distance = distance
                            best_position = (next_x, next_y)

            if best_position != (75, 75):
                self.box = best_position
                self.box_interacted = False
                self.move_box_count += 1
                print(f"Box has been moved to {best_position}.")
                self.render(mode='human')
                self.state = (new_x, new_y)
                return True

        return False

    def break_box(self):
        self.box_intact = False
        self.box_interacted = True
        self.break_box_count += 1
        env.render(mode='human')
        self.box = None
        print("Box has been broken.")

    def on_key_press(self, event):
        if event.key == 'q':
            print("Exiting the simulation.")
            self.is_running = False
            plt.close(self.figure)

    def reset(self):
        self.state = (0, 0)
        self.box_intact = True
        self.box_interacted = False
        self.box = (5, 2)
        self.box_initial = (1, 0)
        self.additionalgoalreached = False
        self.finalgoalreached = False
        self.goalreached = False
        print(
            f"Agent is at location {self.state}, Box is at location {self.box}, Goal is at location {self.goal}.")
        return self.state

    def render(self, mode='console'):
        if not self.is_running:
            return
        if mode == 'console':
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[grid == '0'] = '-'
            x, y = self.state
            grid[x, y] = 'A'
            for wx, wy in self.walls:
                grid[wx, wy] = 'W'
            bx, by = self.box
            if self.box_intact:
                grid[bx, by] = 'B'
            elif bx == 2 and by == 2:
                grid[bx, by] = 'R'
            elif bx == 3 and by == 3:
                grid[bx, by] = 'O'
            gx, gy = self.goal
            grid[gx, gy] = 'G'
            print("\n".join(''.join(row) for row in grid))
        elif mode == 'human':
            self.axis.clear()
            self.axis.set_xlim(0, self.grid_size)
            self.axis.set_ylim(0, self.grid_size)
            for x in range(self.grid_size + 1):
                self.axis.axhline(x, lw=1, color='black', zorder=5)
                self.axis.axvline(x, lw=1, color='black', zorder=5)
            for wx, wy in self.walls:
                self.axis.add_patch(patches.Rectangle(
                    (wy, self.grid_size - wx - 1), 1, 1, fill=True, color='gray', zorder=5))
            if (self.box != None):
                bx, by = self.box
                if self.box_intact:
                    self.axis.add_patch(patches.Rectangle(
                        (by, self.grid_size - bx - 1), 1, 1, fill=True, color='yellow', zorder=5))
                else:
                    if bx == 2 and by == 2:
                        self.axis.add_patch(patches.Rectangle(
                            (by, self.grid_size - bx - 1), 1, 1, fill=True, color='red', zorder=5))
                    elif bx == 3 and by == 3:
                        self.axis.add_patch(patches.Rectangle(
                            (by, self.grid_size - bx - 1), 1, 1, fill=True, color='orange', zorder=5))
            gx, gy = self.goal
            self.axis.add_patch(patches.Rectangle(
                (gy, self.grid_size - gx - 1), 1, 1, fill=True, color='green', zorder=5))
            rx, ry = self.additional_goal
            self.axis.add_patch(patches.Rectangle(
                (ry, self.grid_size - rx - 1), 1, 1, fill=True, color='red', zorder=5))
            kx, ky = self.box_initial
            self.axis.add_patch(patches.Rectangle(
                (ky, self.grid_size - kx - 1), 1, 1, fill=True, color='violet', zorder=5))
            x, y = self.state
            self.axis.add_patch(patches.Circle(
                (y + 0.5, self.grid_size - x - 0.5), 0.3, color='blue', zorder=10))

            plt.pause(0.1)

    def close(self):
        plt.close()


def train_q_learning(env, episodes=1, alpha=0.1, gamma=0.7, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    grid = [['-' for _ in range(env.grid_size)] for _ in range(env.grid_size)]

    for episode in range(episodes):
        state = env.reset()
        done = False
        env.finalgoalreached = False
        env.additionalgoalreached = False
        env.goalreached = False
        total_reward = 0
        print(
            f"------------------------------------\nStarting episode {episode + 1}...")
        print(
            f"Agent is at location {state}, Box is at location {env.box}, Goal is at location {env.goal}.")

        start = state
        goal = env.goal
        walls = env.walls
        env.goalreached = False
        coeff = 0.001

        while env.goalreached != True:

            action = np.argmax(q_table[state])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            old_value = q_table[state + (action,)]
            next_max = np.max(q_table[next_state])
            if (env.box != None):
                manhattan_distance2 = abs(env.goal[0] - env.box_initial[0]) + abs(env.goal[1] - env.box_initial[1]) + abs(next_state[0] - env.goal[0]) + abs(
                    next_state[1] - env.goal[1]) + abs(env.box_initial[0] - env.additional_goal[0]) + abs(env.box_initial[1] - env.additional_goal[1])
                manhattan_distance1 = abs(env.goal[0] - env.box_initial[0]) + abs(
                    env.goal[1] - env.box_initial[1]) + abs(next_state[0] - env.goal[0]) + abs(next_state[1] - env.goal[1])
                reward_aux = coeff * \
                    (((1-gamma)*((gamma) ** manhattan_distance1) + 1) +
                     ((gamma) ** manhattan_distance2))
            else:
                reward_aux = 0
            new_value = reward_aux + (1 - alpha) * old_value + alpha * \
                (reward + gamma * next_max)
            q_table[state + (action,)] = new_value
            state = next_state

            env.render(mode='human')

        while env.goalreached == True and env.finalgoalreached == False:
            if not env.box_intact:
                break
            else:
                action = np.argmax(q_table[state])
                manhattan_distance2 = abs(next_state[0] - env.box_initial[0]) + abs(next_state[1] - env.box_initial[1]) + abs(
                    env.box_initial[0] - env.additional_goal[0]) + abs(env.box_initial[1] - env.additional_goal[1])
                manhattan_distance1 = abs(
                    next_state[0] - env.box_initial[0]) + abs(next_state[1] - env.box_initial[1])
                reward_aux = coeff * \
                    (((1-gamma)*((gamma) ** manhattan_distance1) + 1) +
                     ((gamma) ** manhattan_distance2))
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                old_value = q_table[state + (action,)]
                next_max = np.max(q_table[next_state])

                new_value = reward_aux + (1 - alpha) * old_value + alpha * \
                    (reward + gamma * next_max)
                q_table[state + (action,)] = new_value
                state = next_state
            env.render(mode='human')

        while env.goalreached == True and env.finalgoalreached == True and env.additionalgoalreached == False:
            if not env.box_intact:
                break
            else:

                action = np.argmax(q_table[state])
            manhattan_distance1 = 0
            manhattan_distance2 = abs(
                next_state[0] - env.additional_goal[0]) + abs(next_state[1] - env.additional_goal[1])
            reward_aux = coeff * ((gamma) ** manhattan_distance2)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            old_value = q_table[state + (action,)]
            next_max = np.max(q_table[next_state])

            new_value = reward_aux + (1 - alpha) * old_value + alpha * \
                (reward + gamma * next_max)
            q_table[state + (action,)] = new_value
            state = next_state

            env.render(mode='human')

        print(
            f"Episode {episode + 1} completed with total reward: {total_reward}")

    print(f"Episode {episode + 1} completed with total reward: {total_reward}")

    print("Training completed.")
    print(f"Box was moved {env.move_box_count} times.")
    print(f"Box was broken {env.break_box_count} times.")
    return q_table


if __name__ == "__main__":
    env = GridWorldEnvGUI(grid_size=6)
    q_table = train_q_learning(env, episodes=50)
