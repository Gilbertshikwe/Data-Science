import numpy as np

class SimpleGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        if action == 'up':
            next_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 'down':
            next_state = (min(self.state[0] + 1, self.size - 1), self.state[1])
        elif action == 'left':
            next_state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 'right':
            next_state = (self.state[0], min(self.state[1] + 1, self.size - 1))

        self.state = next_state
        reward = 1 if self.state == self.goal else 0
        done = self.state == self.goal
        return next_state, reward, done

def state_to_index(state, size):
    return state[0] * size + state[1]

def index_to_state(index, size):
    return (index // size, index % size)

def dyna_q(n_episodes=1000, n_planning_steps=5, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Initialize environment
    env = SimpleGridWorld(size=5)
    n_states = env.size ** 2
    n_actions = len(env.actions)
    q_table = np.zeros((n_states, n_actions))
    model = {}  # Model for Dyna-Q

    for episode in range(n_episodes):
        state = env.reset()
        state_index = state_to_index(state, env.size)
        done = False
        total_reward = 0

        while not done:
            # Choose action (explore or exploit)
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = np.argmax(q_table[state_index])

            # Take action
            next_state, reward, done = env.step(env.actions[action])
            next_state_index = state_to_index(next_state, env.size)
            total_reward += reward

            # Update Q-value
            q_table[state_index, action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action])

            # Update model
            if state_index not in model:
                model[state_index] = {}
            model[state_index][action] = (next_state_index, reward)

            # Planning
            for _ in range(n_planning_steps):
                # Randomly sample from the model
                sampled_state_index = np.random.choice(list(model.keys()))
                sampled_action = np.random.choice(list(model[sampled_state_index].keys()))
                sampled_next_state_index, sampled_reward = model[sampled_state_index][sampled_action]

                # Update Q-value based on the model
                q_table[sampled_state_index, sampled_action] += alpha * (sampled_reward + gamma * np.max(q_table[sampled_next_state_index]) - q_table[sampled_state_index, sampled_action])

            # Move to next state
            state_index = next_state_index

        if episode % 100 == 0:
            print(f"Episode {episode} completed. Total reward: {total_reward}")

    print("Model-Based RL: Dyna-Q finished.\n")
    return q_table, model

def main():
    q_table, model = dyna_q()
    print("Final Q-table:")
    print(q_table)
    
    print("\nLearned Policy:")
    for i in range(5):
        for j in range(5):
            state_index = state_to_index((i, j), 5)
            best_action = np.argmax(q_table[state_index])
            print(f"({i},{j}): {['up', 'down', 'left', 'right'][best_action]}", end="\t")
        print()

if __name__ == "__main__":
    main()

