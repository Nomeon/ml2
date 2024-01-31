from typing import Union
from game_state import GameState
import asyncio
import random
import os
import time
import create_cnn
import numpy as np

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
            output_probabilities = self.cnn([cnn_spatial_input, non_spatial_data], training=False)

            # Select action
            action = np.argmax(output_probabilities[0][0])

            print(f'OUTPUT PROB: {output_probabilities[0]}')
            print(f'Sending action: {self._actions[action]} for unit {unit_id}')

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