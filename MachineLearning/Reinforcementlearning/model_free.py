import numpy as np
import gym

def q_learning(n_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    env = gym.make('FrozenLake-v1', is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    for episode in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0

        while not done:
            # Choose action (explore or exploit)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
            else:
                next_state, reward, done, _ = result
                terminated, truncated = done, False

            done = terminated or truncated
            total_reward += reward

            # Update Q-value
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            # Move to next state
            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode} completed. Total reward: {total_reward}")

    print("Model-Free RL: Q-Learning finished.\n")
    return q_table

def print_policy(q_table):
    actions = ['Left', 'Down', 'Right', 'Up']
    policy = np.argmax(q_table, axis=1)
    print("Learned Policy:")
    for i in range(4):
        for j in range(4):
            print(f"{actions[policy[i*4 + j]]}", end="\t")
        print()

def main():
    q_table = q_learning()
    print("Final Q-table:")
    print(q_table)
    print()
    print_policy(q_table)

    # Test the learned policy
    env = gym.make('FrozenLake-v1', is_slippery=False)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    done = False
    total_reward = 0

    print("\nTesting the learned policy:")
    while not done:
        action = np.argmax(q_table[state])
        result = env.step(action)
        if len(result) == 5:
            state, reward, terminated, truncated, _ = result
        else:
            state, reward, done, _ = result
            terminated, truncated = done, False

        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()
