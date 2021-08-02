# Reinforcement learning by example

Work carried out for CSC 8099 MSc Project & Dissertation as part of MSc Computer Science at Newcastle University



## Installation
------------
1. Clone the repo 
    ```sh
    git clone https://github.com/zebexx/reinforcement_learning_by_example
    ```
2. Install packages using pip
    ```sh
    python -m pip -r install requirements.txt
    ```

## Usage
------------
Run `python ddqn/main.py`

Progress will be printed to the console.
Once finished the environment history will be saved in `history`
Graph data will be saved in `graphData`
A graph will be generated displaying the learning curve and epsilon of the agents trained in the main directory.
The graph file name is generated from Hyperparameters used

Hyperparameters can be changed in `main.py`

To generate graphs from saved graph data open `ddqn/plot.py` in a code editor and edit folder and filename variables before running.


## License
------------
Distributed under the MIT License. See `LICENSE` for more information.