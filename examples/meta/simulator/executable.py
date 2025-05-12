from monarch.simulator.simulator import Simulator


# First run "python3 -m examples.simulator.attention" to generate the pkl file.
if __name__ == "__main__":
    Simulator.replay("command_history.pkl")
