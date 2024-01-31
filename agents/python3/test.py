import numpy as np
import tensorflow as tf
import os
import time
from game_state import GameState
import asyncio

uri = os.environ.get('GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"

# Define hyperparameters
gamma = 0.99  # Discount factor
epsilon = 0.2  # Clip parameter
lr = 0.001  # Learning rate
epochs = 10  # Number of optimization epochs

# Define neural network architecture
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        # Modify the architecture as per your requirements
        # Example architecture:
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(num_actions, activation=None)
        self.value = tf.keras.layers.Dense(1, activation=None)

    def call(self, spatial_input):
        x = self.conv1(spatial_input)
        x = self.flatten(x)
        x = self.fc1(x)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value

# Define PPO algorithm
class PPO:
    def __init__(self, num_actions):
        self.policy_network = PolicyNetwork(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    # Other methods such as action selection, training step, etc. can be added here

# Agent class incorporating PPO
class Agent():
    def __init__(self):
        self._client = GameState(uri)
        self._actions = ["up", "down", "left", "right", "bomb", "detonate"]

        # Initialize PPO agent
        self.ppo_agent = PPO(len(self._actions))

        # Set game tick callback
        self._client.set_game_tick_callback(self._on_game_tick)

        # Connect to the game
        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [asyncio.ensure_future(self._client._handle_messages(connection))]
        loop.run_until_complete(asyncio.wait(tasks))

async def _on_game_tick(self, tick_number, game_state):
    my_agent_id = game_state.get("connection").get("agent_id")
    my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")

    # Construct spatial input
    spatial_input = construct_spatial_input(game_state, my_agent_id)  # Implement your spatial input construction

    actions = []
    old_probs = []
    rewards = []

    # Interaction loop for each unit
    for unit_id in my_units:
        cur_unit = ord(unit_id)
        cur_hp = game_state.get("unit_state")[unit_id].get("hp")

        # Get non-spatial data (current tick, unit ID, HP)
        non_spatial_data = np.array([[tick_number, cur_unit, cur_hp]])

        # Perform inference with the PPO agent
        output_probabilities, value = self.ppo_agent.policy_network(spatial_input, non_spatial_data)

        # Select action
        action = np.argmax(output_probabilities[0])

        # Execute action and obtain reward
        reward = execute_action_and_get_reward(action, game_state, unit_id)  # Implement action execution and reward calculation

        # Store data for training
        actions.append(action)
        old_probs.append(output_probabilities[0, action])  # Log probability of selected action
        rewards.append(reward)

    # Train the PPO agent
    self.ppo_agent.train(spatial_input, actions, old_probs, rewards)

def construct_spatial_input(self, game_state, my_agent_id):
        tile_dict = {'x' : 0, 'm' : 1, 'o' : 2, 'w' : 3, 'bp' : 4, 'fp' : 5}
        tiles = game_state.get("entities")

        # CNN Spatial Input
        cnn_spatial_input = np.zeros(self._input_shape)

        for tile in tiles:
            tile_type = tile.get("type")

            # If type is not a bomb:
            if tile_type in tile_dict:
                cnn_spatial_input[tile.get("x"), tile.get("y"), tile_dict[tile_type]] = 1

            # If type is bomb from player
            elif tile_type == 'b' and tile['agent_id'] == my_agent_id:
                cnn_spatial_input[tile.get("x"), tile.get("y"), 6] = 1

            # If type is bomb from enemy 
            elif tile_type == 'b' and tile['agent_id'] != my_agent_id:
                cnn_spatial_input[tile.get("x"), tile.get("y"), 7] = 1
            
            # Unknown entity
            else:
                print(f'Tile type {tile_type} not in tile_dict or bomb')

        # Add information from units:
        for unit in game_state.get("unit_state"):

            # Player units
            if game_state["unit_state"][unit]["agent_id"] == my_agent_id:
                cnn_spatial_input[game_state["unit_state"][unit]["coordinates"][0], game_state["unit_state"][unit]["coordinates"][1], 8] = 1
            
            # Enemy units
            else:
                cnn_spatial_input[game_state["unit_state"][unit]["coordinates"][0], game_state["unit_state"][unit]["coordinates"][1], 9] = 1

        # Expand dimension of array (representing number of samples)
        cnn_spatial_input = np.expand_dims(cnn_spatial_input, axis=0)
        return cnn_spatial_input

# todo: implement reward
async def execute_action_and_get_reward(self, action, game_state, unit_id):
    reward = 0
    if action == 0:
        await self._client.send_move("up", unit_id)
    elif action == 1:
        await self._client.send_move("down", unit_id)
    elif action == 2:
        await self._client.send_move("left", unit_id)
    elif action == 3:
        await self._client.send_move("right", unit_id)
    elif action == 4:
        await self._client.send_bomb(unit_id)
    elif action == 5:
        bomb_coordinates = self._get_bomb_to_detonate(unit_id)
        if bomb_coordinates != None:
            x, y = bomb_coordinates
            await self._client.send_detonate(x, y, unit_id)

        else:
            print(f"Unhandled action: {action} for unit {unit_id}")

    return reward


def main():
    for i in range(0, 10):
        while True:
            try:
                Agent()
            except:
                time.sleep(5)
                continue
            break

if __name__ == "__main__":
    main()
