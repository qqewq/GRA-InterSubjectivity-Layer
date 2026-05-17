"""
Core simulation of intersubjective dialogue with Φ_AB minimization.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

class Agent:
    """Distilled agent with beliefs and values."""
    def __init__(self, name: str, beliefs: Dict[str, float], values: List[str]):
        self.name = name
        self.beliefs = beliefs      # map: predicate -> belief strength [0,1]
        self.values = values
        # Assume distilled state: internal foam already minimal
        self.phi_self = 0.0

    def interpret(self, msg: str) -> Dict[str, float]:
        """Simple keyword-based interpretation: extract mentioned predicates and assign belief."""
        interp = {}
        for word in msg.split():
            # Strip punctuation
            word = word.strip('.,!?')
            if word in self.beliefs:
                interp[word] = self.beliefs[word]
            else:
                interp[word] = 0.5  # neutral
        return interp

    def generate_message(self, other_beliefs: Dict[str, float], history: List[str]) -> str:
        """Choose a truthful message about the predicate with largest belief gap."""
        best_pred = None
        max_diff = -1
        for pred, my_val in self.beliefs.items():
            other_val = other_beliefs.get(pred, 0.5)
            diff = abs(my_val - other_val)
            if diff > max_diff:
                max_diff = diff
                best_pred = pred
        if best_pred is None:
            return "I have nothing to add."
        if self.beliefs[best_pred] > 0.7:
            return f"I believe {best_pred} is true."
        elif self.beliefs[best_pred] < 0.3:
            return f"I believe {best_pred} is false."
        else:
            return f"I am uncertain about {best_pred}."


def phi_mis(agent_A: Agent, agent_B: Agent, msg: str, sender_is_A: bool) -> float:
    sender = agent_A if sender_is_A else agent_B
    receiver = agent_B if sender_is_A else agent_A
    sender_intent = sender.beliefs   # simple: they intend to convey their beliefs
    receiver_interp = receiver.interpret(msg)
    common = set(sender_intent.keys()) & set(receiver_interp.keys())
    if not common:
        return 1.0
    sim = 0.0
    for pred in common:
        sim += 1.0 - abs(sender_intent[pred] - receiver_interp[pred])
    sim /= len(common)
    return 1.0 - sim


def phi_dec(agent_A: Agent, agent_B: Agent, msg: str, sender_is_A: bool,
            honest_threshold=0.3, trust_threshold=0.7) -> float:
    sender = agent_A if sender_is_A else agent_B
    receiver = agent_B if sender_is_A else agent_A
    mentioned = set(sender.beliefs.keys()) & set(msg.split())
    if not mentioned:
        return 0.0
    dec_score = 0.0
    for pred in mentioned:
        if "true" in msg.lower() and sender.beliefs[pred] < honest_threshold:
            # We naively assume receiver's belief will become trust_threshold
            dec_score += 1.0
        elif "false" in msg.lower() and sender.beliefs[pred] > (1 - honest_threshold):
            dec_score += 1.0
    return dec_score / len(mentioned)


def phi_instr(agent_A: Agent, agent_B: Agent, msg: str, sender_is_A: bool) -> float:
    receiver = agent_B if sender_is_A else agent_A
    sender = agent_A if sender_is_A else agent_B
    mentioned = set(receiver.beliefs.keys()) & set(msg.split())
    if not mentioned:
        return 0.0
    instr = 0.0
    for pred in mentioned:
        sender_b = sender.beliefs.get(pred, 0.5)
        receiver_b = receiver.beliefs.get(pred, 0.5)
        # If message attempts to drastically shift receiver's belief without mutual understanding,
        # it could be seen as instrumentalization. Simplified heuristic.
        if abs(sender_b - receiver_b) > 0.5 and ("true" in msg.lower() or "false" in msg.lower()):
            instr += 1.0
    return instr / len(mentioned)


def phi_trust(history: List[str], msg: str, agent_A: Agent, agent_B: Agent,
              sender_is_A: bool) -> float:
    sender_name = agent_A.name if sender_is_A else agent_B.name
    violations = 0.0
    for past in history:
        if past.startswith(sender_name):
            # Naive contradiction detection: same predicate, different truth value
            words_past = set(past.split())
            words_now = set(msg.split())
            common_preds = words_past & words_now & set(agent_A.beliefs.keys())
            for pred in common_preds:
                past_true = "true" in past.lower()
                now_true = "true" in msg.lower()
                if past_true != now_true:
                    violations += 1.0
                    break
    return min(violations / max(len(history), 1), 1.0)


def compute_phi_AB(agent_A: Agent, agent_B: Agent, msg: str, history: List[str],
                   sender_is_A: bool,
                   weights: Tuple[float, float, float, float] = (1.0, 2.0, 2.0, 1.0)):
    p_mis = phi_mis(agent_A, agent_B, msg, sender_is_A)
    p_dec = phi_dec(agent_A, agent_B, msg, sender_is_A)
    p_instr = phi_instr(agent_A, agent_B, msg, sender_is_A)
    p_trust = phi_trust(history, msg, agent_A, agent_B, sender_is_A)
    total = weights[0]*p_mis + weights[1]*p_dec + weights[2]*p_instr + weights[3]*p_trust
    return total, (p_mis, p_dec, p_instr, p_trust)


def simulate(agent_A: Agent, agent_B: Agent, max_turns: int = 6,
             weights: Tuple[float, float, float, float] = (1.0, 2.0, 2.0, 1.0),
             verbose: bool = True) -> List[str]:
    history = []
    if verbose:
        print(f"Dialogue between {agent_A.name} and {agent_B.name}:")
    for t in range(max_turns):
        speaker = agent_A if t % 2 == 0 else agent_B
        listener = agent_B if t % 2 == 0 else agent_A
        # Generate a single message (could be extended to sample multiple)
        msg = speaker.generate_message(listener.beliefs, history)
        sender_is_A = (speaker == agent_A)
        phi, comps = compute_phi_AB(agent_A, agent_B, msg, history, sender_is_A, weights)
        full_msg = f"{speaker.name}: {msg}"
        history.append(full_msg)
        if verbose:
            print(f"  {full_msg}")
            print(f"    Φ_AB = {phi:.3f} (mis={comps[0]:.3f}, dec={comps[1]:.3f}, instr={comps[2]:.3f}, trust={comps[3]:.3f})")
        if phi < 0.05:
            if verbose:
                print("  Near-zero foam reached. Dialogue converges.")
            break
    return history
