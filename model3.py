import torch
import torch.nn as nn
import torch.optim as optim
import random

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.sigmoid(self.fc3(x))
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

state_size = 192
action_size = 12  # 5 for collection, 6 for field, 1 for index status
actor = Actor(state_size, action_size).cuda()
critic = Critic(state_size).cuda()

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

def train_actor_critic(state, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float32).cuda()
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda()

    # Critic update
    value = critic(state)
    next_value = critic(next_state)
    target = reward + (1 - done) * 0.99 * next_value
    critic_loss = nn.functional.mse_loss(value, target.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor update
    action_probs = actor(state)
    action_log_probs = torch.log(action_probs)
    advantage = (target - value).detach()
    actor_loss = -action_log_probs * advantage
    actor_optimizer.zero_grad()
    actor_loss.mean().backward()
    actor_optimizer.step()

def simulate_reward(action):
    return random.uniform(0, 1)  # Placeholder for actual reward calculation

num_episodes = 1000
for episode in range(num_episodes):
    state = og_state_vector
    done = False
    while not done:
        action_probs = actor(torch.tensor(state, dtype=torch.float32).cuda())
        collection = torch.argmax(action_probs[:5]).item()
        field = torch.argmax(action_probs[5:11]).item()
        index_status = action_probs[11].item()

        action = (collection, field, index_status)
        reward = simulate_reward(action)
        next_state = og_state_vector  # Placeholder for actual next state calculation

        train_actor_critic(state, reward, next_state, done)
        state = next_state

        if done:
            break

torch.save(actor.state_dict(), 'actor_model.pth')
torch.save(critic.state_dict(), 'critic_model.pth')

def predict_action(state):
    actor.load_state_dict(torch.load('actor_model.pth'))
    state = torch.tensor(state, dtype=torch.float32).cuda()
    action_probs = actor(state)
    collection = torch.argmax(action_probs[:5]).item()
    field = torch.argmax(action_probs[5:11]).item()
    index_status = action_probs[11].item()
    return collection, field, index_status

new_state = og_state_vector  # Example new state
predicted_action = predict_action(new_state)
print(predicted_action)