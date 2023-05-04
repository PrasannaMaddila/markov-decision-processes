# Markov Decision Processes

This repository is an implementation of Markov Decision Processes in Gymnasium (used to be Gym) in Python. The code is placed inside the [src/](src/) folder, and the main environment is implemented as a Gymnasium environment in [src/mdp.py](src/mdp.py). The Value Iteration algorithm has also been implemented in [src/value_iterations.py](src/value_iterations.py).

## MDP

The class implements a discounted reward MDP with a strategy profile `f`. For the moment, both files are testing the following MDP: 

| -        | State 1          | State 2         | State 3          |
|----------|------------------|-----------------|------------------|
| Action 1 | -5 | (0.9,0,0.1) | 5 | (0,1,0)     | 20 | (0.9,0.1,0) |
| Action 2 | 10 | (0,1,0)     | 0 | (0.8,0.2,0) | undef            |


with a discount factor $\beta = 0.8$, which should have the optimal values $v_\beta = [28.9, 25, 42.8]$. 

[mdp.py](src/mdp.py) can be run as follows:

```bash
python src/mdp.py <initial-state>
```

**Note** that in the implementation, all states are zero-indexed i.e. State 1 from the MDP defined above is implemented as State 0 in the code.

## Value Iterations

[value_iterations.py](src/value_iterations.py) implements the direct iterative VI algorithm to test the MDP class. This can be run directly via 

```bash
python src/value_iterations.py
```

which will return the optimal values and policy for the same MDP. This can be compared by runnning the `mdp.py` file for different initial states.
