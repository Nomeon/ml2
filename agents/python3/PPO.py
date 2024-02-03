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
        # Calculate advantages
        advantages = np.array(rewards) - np.array(values[:-3]) + self.gamma * np.array(values[3:])

        mean_advantages = np.mean(advantages)
        std_advantages = np.std(advantages)
        normalized_advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        return normalized_advantages

    def _surrogate_loss(self, old_probs, new_probs, advantages):
        ratio = new_probs / (old_probs + 1e-8)
        surr1 = ratio * advantages[:, np.newaxis]
        temp = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
        surr2 = temp * advantages[:, np.newaxis]
        loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        return loss

    def train_step(self, states, old_probs, advantages, actions):
        with tf.GradientTape() as tape:
            tape.watch(self.policy.get_layer("output_layer_actions").trainable_variables)
            new_probs = []
            action_values = []  # If you're using action values for loss calculation
            for state in states:
                prediction = self.policy(state, training=True)  # Use training=True
                action_probabilities, estimated_baseline = prediction
                new_probs.append(action_probabilities[0])
                action_values.append(estimated_baseline[0][0])

            # Convert lists to tensors if necessary
            # new_probs = tf.convert_to_tensor(new_probs, dtype=tf.float32)
            # old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
            # print(f"ACTIONS {actions}")
            # print(f"OLDPROBS {old_probs}")
            tmep = tf.convert_to_tensor(new_probs, dtype=tf.float32)
            new_probs = tf.cast(tf.gather(new_probs, actions, batch_dims=1), dtype=tf.float32)
            old_probs = tf.cast(tf.gather(old_probs, actions, batch_dims=1), dtype=tf.float32)
            # print(f"OLDPROBS TENSOR {old_probs}")
            advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

            # Calculate loss
            loss = self._surrogate_loss(old_probs, new_probs, advantages)

        gradients = tape.gradient(loss, self.policy.get_layer("output_layer_actions").trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.5)
        self.optimizer.apply_gradients(zip(gradients, self.policy.get_layer("output_layer_actions").trainable_variables))

        return tmep

    def train(self, states, old_probs, advantages, actions):
        return self.train_step(states, old_probs, advantages, actions)