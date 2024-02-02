import tensorflow as tf
import numpy as np

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
        advantages = np.array(discounted_rewards) - np.array(list(reversed(values)))
        
        return discounted_rewards, advantages

    def _surrogate_loss(self, old_probs, new_probs, advantages):
        ratio = new_probs / (old_probs + 1e-8)
        surr1 = ratio * advantages[:, np.newaxis]
        temp = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
        surr2 = temp * advantages[:, np.newaxis]
        loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        return loss

    # def train_step(self, states, old_probs, new_probs, advantages):
    #     with tf.GradientTape() as tape:
    #         # for var in self.policy.trainable_variables:
    #         #     print(f'VAR: {var.name}, TRAIN: {var.trainable}\n')
    #         # tape.watch(self.policy.trainable_variables)

    #         # new_probs = []
    #         # for state in states:
    #         #     prediction = self.policy(state) # , training=False
                
    #         #     action_probabilities = prediction[0][0]
    #         #     estimated_baseline = prediction[1][0][0]
    #         #     new_probs.append(action_probabilities)
            
    #         # loss = self._surrogate_loss(old_probs, new_probs, advantages)
    #         print(f'STATES: {len(states)}')

    #         predictions = [self.policy(state, training=True) for state in states] # , training=False
    #         # predictions = tf.stack([self.policy(state, training=True) for state in states])
    #         print(f'PREDICTIONS: {predictions}')
    #         # new_probs = [pred[0][0] for pred in predictions]
    #         new_probs = predictions[:, 0, 0]
    #         loss= self._surrogate_loss(old_probs, new_probs, advantages)

    #         if loss is None:
    #             print('LOSS FAILED')

    #         gradients = tape.gradient(loss, self.policy.trainable_variables)
    #         print(f'GRADIENTS: {gradients}')

    #         if any(g is None for g in gradients):
    #             print("Gradient is None. Check model architecture or loss function.")
    #         else:
    #             print(f'GRADIENTS: {gradients}')

    #     self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    def train_step(self, states, old_probs, new_probs, advantages):
        with tf.GradientTape() as tape:
            tape.watch(self.policy.get_layer("output_layer_actions").trainable_variables)
            new_probs = []
            action_values = []  # If you're using action values for loss calculation
            for state in states:
                prediction = self.policy(state, training=True)  # Use training=True
                action_probabilities, estimated_baseline = prediction
                new_probs.append(action_probabilities)
                action_values.append(estimated_baseline)

            # Convert lists to tensors if necessary
            new_probs = tf.convert_to_tensor(new_probs, dtype=tf.float32)
            advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
            old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

            # Calculate loss
            loss = self._surrogate_loss(old_probs, new_probs, advantages)

        gradients = tape.gradient(loss, self.policy.get_layer("output_layer_actions").trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.get_layer("output_layer_actions").trainable_variables))


    def train(self, states, old_probs, new_probs, advantages):
        for _ in range(self.epochs):
            self.train_step(states, old_probs, new_probs, advantages)

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