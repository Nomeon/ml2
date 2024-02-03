import tensorflow as tf
import numpy as np

# Proximal Policy Optimization class
class PPO:
    def __init__(self, model, lr, gamma, epsilon, batch_size=64):
        self.policy = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
    
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

    def train_step(self, states, old_probs, advantages):
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
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.5)
        self.optimizer.apply_gradients(zip(gradients, self.policy.get_layer("output_layer_actions").trainable_variables))

        return new_probs

    def train(self, states, old_probs, advantages):
        return self.train_step(states, old_probs, advantages)