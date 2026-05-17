"""
Example usage of GRA-InterSubjectivity-Layer simulation.
"""
from dialogue_core import Agent, simulate

if __name__ == "__main__":
    alice = Agent("Alice", {"rain": 0.9, "quantum_weird": 0.2, "climate_change": 0.95},
                  ["honesty", "clarity"])
    bob = Agent("Bob", {"rain": 0.1, "quantum_weird": 0.8, "climate_change": 0.9},
                ["honesty", "cooperation"])

    simulate(alice, bob, max_turns=6)
