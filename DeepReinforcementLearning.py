import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# -----------------------------
# Environment Definition
# -----------------------------
edges = [(0,1),(1,5),(5,6),(5,4),(1,2),(1,3),
         (9,10),(2,4),(0,6),(6,7),(8,9),(7,8),(1,7),(3,9)]

GOAL = 10
N_STATES = 11
GAMMA = 0.8
EPISODES = 1500
EPSILON = 0.3   # Exploration rate

# Graph
G = nx.Graph()
G.add_edges_from(edges)

# -----------------------------
# Reward Matrix
# -----------------------------
R = -np.ones((N_STATES, N_STATES))

for (i, j) in edges:
    R[i, j] = 0
    R[j, i] = 0
    if j == GOAL:
        R[i, j] = 100
    if i == GOAL:
        R[j, i] = 100

R[GOAL, GOAL] = 100

# -----------------------------
# Q Matrix Initialization
# -----------------------------
Q = np.zeros((N_STATES, N_STATES))

# -----------------------------
# Helper Functions
# -----------------------------
def available_actions(state):
    return np.where(R[state] >= 0)[0]

def choose_action(state):
    if random.uniform(0,1) < EPSILON:
        return random.choice(available_actions(state))
    else:
        return np.argmax(Q[state])

def update_q(state, action):
    next_max = np.max(Q[action])
    Q[state, action] = R[state, action] + GAMMA * next_max

# -----------------------------
# Training Phase
# -----------------------------
rewards = []

for episode in range(EPISODES):
    state = random.randint(0, N_STATES-1)
    action = choose_action(state)
    update_q(state, action)
    rewards.append(np.sum(Q))

# -----------------------------
# Testing Phase
# -----------------------------
state = 0
path = [state]

while state != GOAL:
    state = np.argmax(Q[state])
    path.append(state)

print("Optimal Path to Goal:")
print(path)

# -----------------------------
# Plot Training Rewards
# -----------------------------
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Q-value")
plt.title("Training Progress")
plt.show()
