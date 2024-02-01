from typing import Union
from game_state import GameState
from PPO import PPO
import asyncio
import random
import os
import time
import create_cnn
import numpy as np
from copy import copy

uri = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"



class Agent():
    def __init__(self):

        self._client = GameState(uri)

        # Init settings for cnn
        self._actions = ["up", "down", "left", "right", "bomb", "detonate"]
        self._non_spatial_shape = 3
        self._input_shape = (15, 15, 10)
        self._num_actions = 6
        self._hidden_units = 64

        # Create cnn
        self.cnn = create_cnn.create_cnn(self._input_shape, self._non_spatial_shape, self._num_actions, self._hidden_units)

        # PPO Hyperparameters
        self._gamma = 0.99
        self._lr = 0.001
        self._epsilon = 0.2
        self._batch_size = 3

        # Init settings for training
        self._states = []
        #self._actions = []
        self._rewards = []
        self._values = []
        self._old_probs = [[1/self._num_actions for _ in range(self._num_actions)] for _ in range(self._batch_size)]
        self._old_probs = [[1/self._num_actions for _ in range(self._num_actions)] for _ in range(self._batch_size)]
        self._new_probs = []
        
        # Create PPO
        self.ppo = PPO(self.cnn, self._lr, self._gamma, self._epsilon, self._batch_size)

        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

        self._prev_state = None

    # returns coordinates of the first bomb placed by a unit
    def _get_bomb_to_detonate(self, unit) -> Union[int, int] or None:
        entities = self._client._state.get("entities")
        bombs = list(filter(lambda entity: entity.get(
            "unit_id") == unit and entity.get("type") == "b", entities))
        bomb = next(iter(bombs or []), None)
        if bomb != None:
            return [bomb.get("x"), bomb.get("y")]
        else:
            return None

    async def _on_game_tick(self, tick_number, game_state):
        print("TICKING TICKING")
        if len(self._rewards) >= self._batch_size:
            print(f"{tick_number}========================", self.cnn.get_weights()[0][0][0], "============================")

            # Update Network
            self._states = self._states[:self._batch_size]
            #self._actions = np.array(self._actions[:self._batch_size])
            self._rewards = np.array(self._rewards[:self._batch_size])
            self._values = np.array(self._values[:self._batch_size])
            self._new_probs = np.array(self._new_probs[:self._batch_size])
            self._old_probs = np.array(self._old_probs[:self._batch_size])    
            self._values = np.array(self._values[:self._batch_size])
            self._new_probs = np.array(self._new_probs[:self._batch_size])
            self._old_probs = np.array(self._old_probs[:self._batch_size])    

            _, advantages = self.ppo.compute_advantage(self._rewards, self._values)

            self.ppo.train(self._states, self._old_probs, self._new_probs, advantages)

            # Reset settings for training new game
            self._states = []
            #self._actions = []
            self._rewards = []
            self._values = []
            self._old_probs = copy(self._new_probs)
            self._new_probs = []

        if tick_number == 1000:
            self._save_weights()
            return
        elif tick_number == 1:
            print("first========================", self.cnn.get_weights()[0][0][0], "============================")
            #print(self.cnn.get_weights().shape)
            self._prev_state = game_state

        # get my units
        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")
        # TO DO:

        # Run neural network once for all units, instead of calling it multiple times
        # Add code to detonate specific bombs, currently the action "detonate" will detonate the bomb placed first by the unit

        ## Dictionary of (x,y):type
        ## e.g. {(0,1): 'w'}
        ## types include: 
            # a: ammunition
            # b: Bomb
            # x: Blast
            # bp: Blast Powerup
            # fp: Freeze Powerup
            # m: Metal Block
            # o: Ore Block
            # w: Wooden Block
        
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

        # send each unit a random action
        for unit_id in my_units:
            cur_unit = ord(unit_id)
            cur_hp = game_state.get("unit_state")[unit_id].get("hp")

            # Get current game tick, unit, and HP
            non_spatial_data = np.array([[tick_number, cur_unit, cur_hp]])

            # Generate a random test sample for non_spatial data
            # non_spatial_data = np.random.rand(1, 10)

            # Perform inference
            # action_probabilities, estimated_baseline = self.cnn([cnn_spatial_input, non_spatial_data], training=False)
            prediction = self.cnn([cnn_spatial_input, non_spatial_data], training=False)
            action_probabilities = prediction[0][0]
            estimated_baseline = prediction[1][0][0]
            self._new_probs.append(action_probabilities)

            # Select action
            # action = np.argmax(action_probabilities)
            action = np.random.choice(np.arange(self._num_actions), p=action_probabilities.numpy())
            action = np.random.choice(np.arange(self._num_actions), p=action_probabilities.numpy())

            self._update_training_data(state=[cnn_spatial_input, non_spatial_data], action=action, 
                                      game_state=game_state, tick_number=tick_number, unit=unit_id, value=estimated_baseline)

            print(f'OUTPUT PROB: {action_probabilities}')
            # print(f'BASELINE: {action_probabilities[1][0]}')
            print(f'action: {action}')
            # print(f'Sending action: {self._actions[action]} for unit {unit_id}')

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


    def _save_weights(self):
        self.cnn.save_weights(f'/app/data/weights.h5')

    def _update_training_data(self, state, action, game_state, tick_number, unit, value):
        if tick_number != 1:
            reward = self._calculate_reward(game_state, unit)
            self._rewards.append(reward)
        #print("HELLLLLO")
        
        self._states.append(state)
        #self._actions.append(action)
        self._values.append(value)

    def _calculate_reward(self, game_state, current_unit):
        reward = 0  # Placeholder reward, modify as per game objectives

        prev_coordinates = self._prev_state.get("unit_state")[current_unit].get("coordinates")
        coordinates = game_state.get("unit_state")[current_unit].get("coordinates")

        prev_hp =  int(self._prev_state.get("unit_state")[current_unit].get("hp"))
        hp =  int(game_state.get("unit_state")[current_unit].get("hp"))

        prev_bombs = int(self._prev_state.get("unit_state")[current_unit].get("inventory").get("bombs"))
        bombs = int(game_state.get("unit_state")[current_unit].get("inventory").get("bombs"))

        if prev_coordinates != coordinates:
            # Unit moved
            reward += (-1)
        else:
            # Unit not moved
            reward += (-10)  # Invalid moves included
        if prev_hp > hp:
            # Lost HP
            reward += (-100)
        if prev_bombs > bombs:
            # Bomb placed
            reward += (5)

        # TODO:
            # Add reward for being the last unit alive
            # Add penalty for being close to a bomb

        self._prev_states = game_state
        return reward

def main():
    for i in range(0,10):
        while True:
            try:
                Agent()
            except:
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()