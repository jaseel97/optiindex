import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Define the actor network
def build_actor(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(100, activation='sigmoid')  # Output layer with sigmoid activation
    ])
    return model

# Define the critic network
def build_critic(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)  # Output a single value
    ])
    return model

# Placeholder reward function (to be replaced with actual implementation)
def reward_function(state, value):
    # Example: reward based on the mean of state values and the value from critic
    state_mean = np.mean(state)
    reward = state_mean + value
    return reward

# Define input shape and build the models
input_shape = (800,)
actor = build_actor(input_shape)
critic = build_critic(input_shape)

# Optimizers
actor_optimizer = optimizers.Adam(learning_rate=0.001)
critic_optimizer = optimizers.Adam(learning_rate=0.001)

# Training step
def train_step(state):
    state = np.reshape(state, (1, -1))  # Reshape the input to (1, 800)
    with tf.GradientTape(persistent=True) as tape:
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probs = actor(state)
        value = critic(state)
        reward = reward_function(state, value)
        
        # Critic loss
        critic_loss = tf.math.square(reward - value)
        
        # Actor loss (policy gradient loss)
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.reduce_mean(action_log_probs * (reward - value))

    # Compute gradients
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    
    # Apply gradients
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    return actor_loss, critic_loss

# Dummy state for testing
dummy_state = np.random.random((800,))

# Training loop
for episode in range(1000):  # Number of episodes can be adjusted
    actor_loss, critic_loss = train_step(dummy_state)
    if episode % 100 == 0:
        print(f"Episode {episode}: Actor Loss: {actor_loss.numpy()}, Critic Loss: {critic_loss.numpy()}")

# Get the final output from the actor
dummy_state = np.reshape(dummy_state, (1, -1))  # Reshape the input to (1, 800)
final_output = actor(tf.convert_to_tensor(dummy_state, dtype=tf.float32)).numpy().flatten()
print("Final Output:", final_output)