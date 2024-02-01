import tensorflow as tf
import numpy as np

# class PolicyNetwork(tf.keras.Model):
#     def __init__(self, input_dim, action_dim):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = tf.keras.layers.Dense(64, activation='relu', input_shape=input_dim)
#         self.fc2 = tf.keras.layers.Dense(64, activation='relu')
#         self.fc3 = tf.keras.layers.Dense(action_dim, activation='softmax')

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         action_probs = self.fc3(x)
#         return action_probs


# Proximal Policy Optimization class
class PPO:
    def __init__(self, model, lr, gamma, epsilon, batch_size=64, epochs=1):
        self.policy = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
    
    def compute_advantage(self, rewards, values):
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_rewards = 0

        for reward in reversed(rewards):
            cumulative_rewards = reward + self.gamma * cumulative_rewards
            discounted_rewards.insert(0, cumulative_rewards)
        
        # Calculate advantages
        advantages = np.array(discounted_rewards) - np.asarray(reversed(values))
        
        return discounted_rewards, advantages

    def _surrogate_loss(self, old_probs, new_probs, advantages):
        ratio = new_probs / (old_probs + 1e-8)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        return loss

    def train_step(self, old_probs, new_probs, advantages):
        with tf.GradientTape() as tape:
            loss = self._surrogate_loss(old_probs, new_probs, advantages)

        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    def train(self, old_probs, new_probs, advantages):
        for _ in range(self.epochs):
            self.train_step(old_probs, new_probs, advantages)

    # def compute_advantage(self, states, rewards, next_value):
    #     advantages = np.zeros_like(rewards, dtype=np.float32)
    #     returns = np.zeros_like(rewards, dtype=np.float32)

    #     advantage = 0
    #     next_value = next_value.numpy()

    #     for i in reversed(range(len(rewards))):
    #         returns[i] = rewards[i] + self.gamma * next_value
    #         next_value = returns[i]

    #         td_error = rewards[i] + self.gamma * next_value - values[i].numpy()
    #         advantage = advantage * self.gamma * self.epsilon + td_error
    #         advantages[i] = advantage

    #     return returns, advantages