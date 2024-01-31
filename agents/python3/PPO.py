# class PolicyNetwork(tf.keras.Model):
#     def __init__(self, input_dim, action_dim):
#         super(Policy, self).__init__()
#         self.fc1 = tf.keras.layers.Dense(64, activation='relu')
#         self.fc2 = tf.keras.layers.Dense(64, activation='relu')
#         self.fc3 = tf.keras.layers.Dense(action_dim, activation='softmax')

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         action_probs = self.fc3(x)
#         return action_probs


# # Proximal Policy Optimization class
# class PPO:
#     def __init__(self, state_dim, action_dim, lr, gamma, clip_ratio, epochs):
#         self.policy = Policy(state_dim, action_dim)
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#         self.gamma = gamma
#         self.clip_ratio = clip_ratio
#         self.epochs = epochs

#     def compute_advantage(self, rewards, values, next_value):
#         advantages = np.zeros_like(rewards, dtype=np.float32)
#         returns = np.zeros_like(rewards, dtype=np.float32)

#         advantage = 0
#         next_value = next_value.numpy()

#         for i in reversed(range(len(rewards))):
#             returns[i] = rewards[i] + self.gamma * next_value
#             next_value = returns[i]

#             td_error = rewards[i] + self.gamma * next_value - values[i].numpy()
#             advantage = advantage * self.gamma * self.clip_ratio + td_error
#             advantages[i] = advantage

#         return returns, advantages

#     def surrogate_loss(self, old_probs, actions, advantages):
#         action_probs = self.policy(actions)
#         ratio = tf.reduce_sum(action_probs * actions, axis=-1) / (old_probs + 1e-8)
#         surr1 = ratio * advantages
#         surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
#         loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
#         return loss

#     def train_step(self, states, actions, old_probs, returns, advantages):
#         with tf.GradientTape() as tape:
#             values = self.policy(states)
#             loss = self.surrogate_loss(old_probs, actions, advantages)

#         gradients = tape.gradient(loss, self.policy.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

#     def train(self, states, actions, old_probs, returns, advantages):
#         for _ in range(self.epochs):
#             self.train_step(states, actions, old_probs, returns, advantages)