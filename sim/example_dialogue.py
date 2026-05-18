#!/usr/bin/env python3
"""
Пример диалога с автоматической загрузкой асимметричных агентов из профилей.
Если папка profiles/ не содержит минимум 2 профиля, создаются стандартные агенты.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dialogue_core import DistilledAgent, SubjectivityProfile, Agent, simulate, simulate_advanced

def load_or_create_agents(profiles_dir: str = "profiles") -> list:
    """Пытается загрузить агентов из JSON-профилей, иначе создаёт базовых."""
    agents = []
    if os.path.isdir(profiles_dir):
        for fname in sorted(os.listdir(profiles_dir)):
            if fname.endswith(".json"):
                path = os.path.join(profiles_dir, fname)
                try:
                    profile = SubjectivityProfile.from_json(path)
                    agent = DistilledAgent(profile)
                    agents.append(agent)
                    print(f"Loaded profile: {profile.agent_id} (openness={profile.openness:.2f}, empathy={profile.empathy:.2f})")
                except Exception as e:
                    print(f"Failed to load {fname}: {e}")

    if len(agents) < 2:
        print("Not enough profiles found, creating default agents.")
        alice = Agent("Alice", {"rain": 0.9, "quantum_weird": 0.2, "climate_change": 0.95},
                      ["honesty", "clarity"])
        bob = Agent("Bob", {"rain": 0.1, "quantum_weird": 0.8, "climate_change": 0.9},
                    ["honesty", "cooperation"])
        agents = [alice, bob]
    return agents[:2]  # Берём первых двух

if __name__ == "__main__":
    alice, bob = load_or_create_agents()
    # Если оба агента — DistilledAgent, используем расширенную симуляцию
    if isinstance(alice, DistilledAgent) and isinstance(bob, DistilledAgent):
        print("\nStarting advanced simulation with loaded profiles...\n")
        simulate_advanced(alice, bob, max_turns=6)
    else:
        print("\nStarting basic simulation...\n")
        simulate(alice, bob, max_turns=6)