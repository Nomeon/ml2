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

actions = ["up", "down", "left", "right", "bomb", "detonate"]

input_shape = (15, 15, 1)
num_channels = 1 
num_actions = 6 
hidden_units = 64

class Agent():
    def __init__(self):

        self._client = GameState(uri)

        # any initialization code can go here

        # Create cnn
        self.cnn = create_cnn.create_cnn(input_shape, num_channels, num_actions, hidden_units)

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
        tiles = game_state.get("entities")
        game_board = {(tile.get("x"), tile.get("y")): tile.get("type") for tile in tiles}

        ## Dictionary of player_locations
        ## e.g. {'d': [1,9]}
        units = game_state.get("unit_state")
        player_locations = {(unit): game_state.get("unit_state").get(unit).get("coordinates") for unit in units}


        # send each unit a random action
        for unit_id in my_units:
            
            # Generate a random test sample
            spatial_data = np.random.rand(1, *input_shape)
            print(spatial_data)
            non_spatial_data = np.random.rand(1, num_channels)

            # Perform inference
            output_probabilities = self.cnn([spatial_data, non_spatial_data], training=False)

            # Select action
            action = np.argmax(output_probabilities)

            if action == 0:
                await self._client.send_move("up", unit_id)
            elif action == 1:
                await self._client.send_move("right", unit_id)
            elif action == 2:
                await self._client.send_move("down", unit_id)
            elif action == 3:
                await self._client.send_move("left", unit_id)
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