# Hindsight Experience Replay for Neural-Guided Monte Carlo Tree Search

A configurable implementation of Hindsight Experience Replay (HER) for generalized AlphaZero.

## Installation

Python version 3.9 is expected. Install project dependencies by running

```pip install -r requirements.txt```

## Supported game types

- Gymnasium environments can be used with the [GymGame](src/game/gym_game.py) interface. Following environments
were included in tests:
  - BitFlip: bit-flipping environment as defined in 
  [the HER paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf)
  - [PointMaze](https://robotics.farama.org/envs/maze/point_maze/): physics-based maze navigation with a ball 
  
  Environment-specific details are mostly kept transparent to the learning setup. Notable exceptions include:
  - Goal distance for HER experience ranking is assumed to be a linear norm, but other metrics should be used in many
  cases
  - A helper function ```reset_env_to_state``` sets environment directly to a given state to avoid unreliable
  ```copy.deepcopy``` and lower memory consumption. 
  However, this requires some knowledge of some internal data

- [FindEquationGame](src/game/find_equation_game.py) interpolates unknown equations formed from a context-free grammar
by a set of datapoints.

- New games can be added by inheriting from a [Game](src/game/game.py) class.

## HER variants

Following hindsight aspects can be customized:

- Trajectory selection: played states or random from the search tree
- Number of trajectories
- Virtual goal selection strategy: k future states or final state only
- Number of goals selected per trajectory
- AlphaZero policy target (move probabilities in HER samples): original, one-hot or noisy
- [Hindsight-Combined Experience Replay](https://www.researchgate.net/publication/346030781) can be enabled to guarantee
that latest episode transitions are included in training without delays
- [Experience Ranking](https://ieeexplore.ieee.org/abstract/document/8850705/) can be enabled to force selected HER
goals to be close to the original goal
- [Aggressive Rewards](http://arxiv.org/abs/1809.02070) can be enabled to reduce HER bias by numerically changing
hindsight rewards


## Tests

The [test script](test/test_a0.py) contains example HER-enabled configurations for different environments.
Alternatively, the program can be run manually with

```python main.py [arguments]```

Full list of available arguments and their description can be found in the [configuration file](src/config.py).

## Credits

- This repository is based on existing AlphaZero/MuZero implementations: https://github.com/suragnair/alpha-zero-general
and https://github.com/kaesve/muzero (MIT licensed, see [here](license/LICENSE-alphazero) and
[here](license/LICENSE-muzero)).
- MCTS and equation discovery code is adapted from https://github.com/wwjbrugger/AmEx-MCTS (MIT licensed,
[here](license/LICENSE-amex)).
- BitFlip environment is based on https://github.com/NervanaSystems/gym-bit-flip/blob/master/gym_bit_flip/bit_flip.py
(Apache, see [here](https://pypi.org/project/gym-bit-flip/)).
