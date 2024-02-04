from typing import Union
from game_state import GameState
from PPO import PPO
import asyncio
import random
import os
import time
import create_cnn
import numpy as np
from copy import deepcopy

uri = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"


class Agent():
    def __init__(self):
        self._client = GameState(uri)

        self._prev_state = None
        self._game_count = 0
        self._agent_id = None
        self._my_id = "default"

        self._is_training = False  # Set False for not updating the weights
        self._is_save_weights = False
        self._is_load_weights = True
        self._input_path = 'training_agent1/b_weights_agent1.h5'

        # Init settings for cnn
        self._actions = ["up", "down", "left", "right", "bomb"]  # "detonate"
        self._non_spatial_shape = 1
        self._input_shape = (15, 15, 10)
        self._num_actions = len(self._actions)
        self._hidden_units = 64

        # Create cnn
        self.cnn = create_cnn.create_cnn(self._input_shape, self._non_spatial_shape, self._num_actions, self._hidden_units)
        if self._is_load_weights:
            self.cnn.load_weights(self._input_path)

        # PPO Hyperparameters
        self._gamma = 0.99
        self._lr = 0.001       # Learning rate
        self._epsilon = 0.2    # Clippping parameter
        self._batch_size = 30  # (>= 3) One action and reward added for each unit
                               # If each unit is alive it means 3 action per tick
        # Init settings for training
        self._states = []
        self._rewards = []
        self._values = []
        self._chosen_actions = []
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
        print(f'TICKING {tick_number}')

        if tick_number == 1000:
            if self._is_save_weights:
                output_path = f'{self._agent_id}_weights_agent1.h5'

                self.cnn.save_weights(output_path)
            return
        
        elif tick_number == 1:
            # Reset settings for training new game
            self._states = []
            self._rewards = []
            self._values = []
            self._chosen_actions = []

            if self._game_count != 0:
                self._old_probs = deepcopy(self._new_probs)

            self._new_probs = []
            self._prev_state = deepcopy(game_state)

            self._game_count += 1

        if len(self._rewards) >= self._batch_size:
            # Update Network
            n = min(self._batch_size, len(self._rewards))

            self._old_probs = np.array(self._old_probs[:n])

            rewards = deepcopy(np.array(self._rewards[:n]))
            values = deepcopy(np.array(self._values[:n+3]))
            states = deepcopy(self._states[:n])
            actions = deepcopy(self._chosen_actions[:n])

            if self._is_training:
                advantages = self.ppo.compute_advantage(rewards, values)
                self._new_probs = self.ppo.train(states, self._old_probs, advantages, actions)
                self._new_probs = self._new_probs.numpy().reshape(-1, self._num_actions)

            self._states = list(self._states[n:])
            self._rewards = list(self._rewards[n:])
            self._values = list(self._values[n:])
            self._chosen_actions = list(self._chosen_actions[n:])
            self._old_probs = deepcopy(self._new_probs)

        # get my units
        my_agent_id = game_state.get("connection").get("agent_id")
        self._agent_id = my_agent_id
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")
        
        # TO DO:
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

        unit_num = 0
        # send each unit a random action
        for unit_id in my_units:
            cur_unit = ord(unit_id)
            cur_hp = game_state.get("unit_state")[unit_id].get("hp")

            # Get current game tick, unit, and HP
            # non_spatial_data = np.array([[tick_number, cur_unit, cur_hp]])
            non_spatial_data = np.array([[cur_hp]])

            # Perform inference
            prediction = self.cnn([cnn_spatial_input, non_spatial_data], training=True)
            action_probabilities = prediction[0][0]
            estimated_baseline = prediction[1][0][0]


            # Select action
            action = np.random.choice(np.arange(self._num_actions), p=action_probabilities.numpy())

            unit_num += 1
            self._update_training_data(unit_num=unit_num, state=[cnn_spatial_input, non_spatial_data], action=action,
                                      game_state=game_state, tick_number=tick_number, unit=unit_id, value=estimated_baseline)

            print(f'ACTION: {self._actions[action].upper()} for unit {unit_id.upper()}\nPROBABILITY: {action_probabilities}')

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

        if (len(self._rewards) != 0):
            print(f'REWARD: {sum(self._rewards)}')
        else:
            print(f'NO REWARD')

    def _save_weights(self):
        self.cnn.save_weights(f'/app/data/weights.h5')

    def _update_training_data(self, unit_num, state, action, game_state, tick_number, unit, value):
        if tick_number != 1:
            reward = self._calculate_reward(unit_num, game_state, unit)
            self._rewards = np.append(self._rewards, reward)

        self._states.append(state)
        self._values.append(value)
        self._chosen_actions.append(action)

    def _calculate_reward(self, unit_num, game_state, current_unit):
        reward = 0  # Placeholder reward, modify as per game objectives

        prev_coordinates = self._prev_state.get("unit_state")[current_unit].get("coordinates")
        coordinates = game_state.get("unit_state")[current_unit].get("coordinates")

        prev_hp =  int(self._prev_state.get("unit_state")[current_unit].get("hp"))
        hp =  int(game_state.get("unit_state")[current_unit].get("hp"))

        prev_bombs = int(self._prev_state.get("unit_state")[current_unit].get("inventory").get("bombs"))
        bombs = int(game_state.get("unit_state")[current_unit].get("inventory").get("bombs"))
        # print(f"BOMBS: {coordinates}, {bombs}")
        # print(f"HP: {prev_hp}, {hp}", type(game_state))

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
            reward += (60)
        
        is_danger, penalty = self._is_in_danger(game_state, current_unit)
        if is_danger:
            # Add penalty for being close to a bomb
            reward += (-penalty)            

        if unit_num == 3:
            # print("CHANGE")
            self._prev_state = deepcopy(game_state)

        return reward
    
    def _is_in_danger(self, game_state, unit_id):
        unit_position = game_state.get("unit_state")[unit_id].get("coordinates")
        gameboard = self._get_gameboard(game_state)

        rows = len(gameboard)
        cols = len(gameboard[0])


        # Check for bombs in the same row
        for j in range(unit_position[1], max(0, unit_position[1] - 3) -1, -1):
            if gameboard[unit_position[0]][j] == 1:
                return True, 50 - abs(j - unit_position[1])*7
            elif gameboard[unit_position[0]][j] == -1:
                break
        
        for j in range(unit_position[1], min(cols, unit_position[1] + 4)):
            if gameboard[unit_position[0]][j] == 1:
                return True, 50 - abs(j - unit_position[1])*7
            elif gameboard[unit_position[0]][j] == -1:
                break
        
        # Check for bombs in the same column
        for i in range(unit_position[0], max(0, unit_position[0] - 3) -1, -1):
            if gameboard[i][unit_position[1]] == 1:
                return True, 50 - abs(i - unit_position[0])*7
            if gameboard[i][unit_position[1]] == -1:
                break
            
        for i in range(unit_position[0], min(rows, unit_position[0] + 4)):
            if gameboard[i][unit_position[1]] == 1:
                return True, 50 - abs(i - unit_position[0])*7
            if gameboard[i][unit_position[1]] == -1:
                break
        
        return False, -1
    
    def _get_gameboard(self, game_state):
        game_board = np.zeros((15,15), dtype=np.int32)
        # tile_dict = {'x' : 0, 'm' : 1, 'o' : 2, 'w' : 3, 'bp' : 4, 'fp' : 5}

        tiles = game_state.get("entities")

        for tile in tiles:
            tile_type = tile.get("type")

            # If type is not a bomb:
            if tile_type in ["m", "o", "w"]:
                game_board[tile.get("x"), tile.get("y")] = -1

            # If type is bomb
            elif tile_type == 'b':
                game_board[tile.get("x"), tile.get("y")] = 1
        
        return game_board            


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