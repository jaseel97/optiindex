import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_values = torch.sigmoid(self.fc3(x))  # Action values between 0 and 1
        return action_values

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)  # State value
        return value

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
    
    def forward(self, state):
        action_values = self.actor(state)
        state_value = self.critic(state)
        return action_values, state_value


def reward_function(state, action):
    return torch.sum(action)

def train_step(state):
    state = torch.FloatTensor(state)
    action_values, state_value = model(state)
    reward = reward_function(state, action_values)

    advantage = reward - state_value
    actor_loss = -torch.log(action_values.mean()) * advantage.detach()
    critic_loss = advantage.pow(2)
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# hyperparameters
state_size = 180
action_size = 30
hidden_size = 256
learning_rate = 0.0001
num_epochs = 2000

model = ActorCritic(state_size, action_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    state = [[random.random() for _ in range(state_size)]]
    loss = train_step(state)
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}')


model_path = "actor_critic_model.pth"
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

loaded_model = ActorCritic(state_size, action_size, hidden_size)
loaded_model.load_state_dict(torch.load(model_path))

random_state = torch.FloatTensor([[random.random() for _ in range(state_size)]])
action_values, _ = loaded_model(random_state)

print(f'Predicted action values:\n{action_values}')