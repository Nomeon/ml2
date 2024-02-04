# Neural Network Performance in Bomberman: A Proximal Policy Optimization Study

*This is a repository for a course project at University of Twente.*

## Overview

In this project, we aim to develop an AI agent capable of mastering Bomberman through the application of the PPO algorithm using CNN. Bomberman is a strategic video game where players navigate a grid-based environment, placing bombs to eliminate opponents and obstacles.

The AI agent is trained using reinforcement learning techniques, specifically PPO, to learn optimal strategies for navigating the Bomberman environment, placing bombs strategically, and ultimately outperforming opponents.

This project is a fork from the Coder One engine for the Bomberland challenge. The original repository can be found [here](https://github.com/CoderOneHQ/bomberland).

![Bomberland multi-agent environment](./engine/bomberland-ui/src/source-filesystem/docs/2-environment-overview/bomberland-preview.gif "Bomberland")

## Repository Structure

- `agents/python3/agent[1-5].py`: Contains implementations of the AI agent class with different hyparparameter settings, including functions for interacting with the game environment and updating the policy based on rewards.
- `agents/python3/ppo.py`: Implements the Proximal Policy Optimization algorithm for training the AI agent.
- `agents/python3/cnn.py`: Defines the Convolutional Neural Network architecture used to process the game state and make action predictions.
- `agents/python3/admin_state.py`: Implements an admin agent who is responisble for launching games and saving the results.
- `requirements.txt`: Contains the list of Python dependencies required to run the code.

## Usage

### Basic usage

See: [Documentation](https://www.gocoder.one/docs)

1. Clone or download this repo (including both `base-compose.yml` and `docker-compose.yml` files).
2. To connect agents and run a game instance, run from the root directory:

```
docker-compose up --abort-on-container-exit --force-recreate
```

3. While the engine is running, access the client by going to `http://localhost:3000/` in your browser (may be different depending on your settings).
4. From the client, you can connect as a `spectator` or `agent` (to play as a human player)
5. Whenever a change is made to `bace-compose.yml` or to `docker-compose.yml` a build is required. Do so using:
```
docker-compose up --abort-on-container-exit --force-recreate --build
```

---

Happy gaming with your Bomberman AI Agent! If you have any questions or feedback, don't hesitate to reach out.


