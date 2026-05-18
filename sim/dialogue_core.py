#!/usr/bin/env python3
"""
GRA InterSubjectivity Dialogue Core
- расчёт компонентов пены (Φ_mis, Φ_dec, Φ_instr, Φ_trust)
- класс Agent с базовым генератором сообщений
- класс SubjectivityProfile для загрузки профилей
- класс DistilledAgent с когнитивными параметрами и интеграцией DeepSeek
- стандартная и расширенная симуляции
"""

import json
import os
import math
import random
import openai
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# -------------------- Вспомогательные константы --------------------
EPS = 1e-9

# -------------------- Базовый класс Agent --------------------
class Agent:
    """Обычный агент с убеждениями и ценностями."""
    def __init__(self, name: str, beliefs: Dict[str, float], values: List[str] = None):
        self.name = name
        self.beliefs = beliefs
        self.values = values or ["honesty", "clarity"]
        self.history = []

    def generate_message(self, other_beliefs: Dict[str, float], dialogue_history: List[str]) -> str:
        """Базовый генератор: просто перечисляет убеждения."""
        return f"My beliefs: {self.beliefs}"

    def interpret(self, message: str) -> Dict[str, float]:
        """Извлекает вероятности из сообщения."""
        # Пытаемся распарсить старый формат
        try:
            if "My beliefs:" in message:
                dict_str = message.split(": ", 1)[1]
                interp = eval(dict_str)
                if isinstance(interp, dict):
                    return interp
        except:
            pass
        # Пытаемся найти числа в тексте
        import re
        extracted = {}
        for topic in self.beliefs:
            human_topic = topic.replace("_", " ")
            pattern = rf'{re.escape(human_topic)}.*?(\d+\.\d+)'
            match = re.search(pattern, message)
            if match:
                extracted[topic] = float(match.group(1))
        if extracted:
            return extracted
        # fallback
        return {k: 0.5 for k in self.beliefs}

    def __repr__(self):
        return f"Agent({self.name})"


# -------------------- Функции пены --------------------
def phi_mis(sender: Agent, receiver: Agent, message: str, sender_turn: bool) -> float:
    intended = sender.beliefs
    interp = receiver.interpret(message)
    diff = 0.0
    for k in intended:
        if k in interp:
            diff += (intended[k] - interp[k]) ** 2
    return math.sqrt(diff) if diff > 0 else 0.0

def phi_dec(sender: Agent, receiver: Agent, message: str, sender_turn: bool) -> float:
    interp = receiver.interpret(message)
    diff = 0.0
    for k in sender.beliefs:
        if k in interp:
            diff += (sender.beliefs[k] - interp[k]) ** 2
    return math.sqrt(diff) if diff > 0 else 0.0

def phi_instr(sender: Agent, receiver: Agent, message: str, sender_turn: bool) -> float:
    lower_msg = message.lower()
    if "you must" in lower_msg or "do this" in lower_msg or "just follow" in lower_msg:
        return 1.0
    return 0.0

def phi_trust(sender: Agent, receiver: Agent, message: str, sender_turn: bool,
              history: List[str]) -> float:
    trust_loss = 0.0
    for past_msg in history[-3:]:
        if "deception" in past_msg.lower() or "instrumental" in past_msg.lower():
            trust_loss += 0.2
    return min(1.0, trust_loss)


# -------------------- Профиль субъективности --------------------
@dataclass
class SubjectivityProfile:
    agent_id: str
    initial_beliefs: Dict[str, float]
    openness: float
    empathy: float
    learning_rate: float
    meta_preference: float
    trust_violations: List[str] = field(default_factory=list)
    emotional_state: Optional[str] = None
    convergence: float = 0.0
    last_phi: float = 0.0

    @classmethod
    def from_json(cls, path: str) -> "SubjectivityProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sp = data["subjectivity_profile"]
        metrics = data.get("dialogue_metrics", {})
        return cls(
            agent_id=data["agent_name"],
            initial_beliefs=sp["initial_beliefs"],
            openness=sp["openness"],
            empathy=sp["empathy"],
            learning_rate=sp["learning_rate"],
            meta_preference=sp["meta_preference"],
            trust_violations=sp.get("trust_violations", []),
            emotional_state=sp.get("final_emotion"),
            convergence=metrics.get("convergence_rate", 0.0),
            last_phi=metrics.get("final_phi", 0.0)
        )


# -------------------- Агент с DeepSeek --------------------
class DistilledAgent(Agent):
    """
    Асимметричный агент, использующий DeepSeek для научной аргументации.
    """
    def __init__(self, profile: SubjectivityProfile, default_values: List[str] = None):
        super().__init__(
            name=profile.agent_id,
            beliefs=profile.initial_beliefs,
            values=default_values or ["honesty", "clarity"]
        )
        self.profile = profile
        self.openness = profile.openness
        self.empathy = profile.empathy
        self.lr = profile.learning_rate
        self.meta_pref = profile.meta_preference
        self.emotion = profile.emotional_state
        self.trust_violations = set(profile.trust_violations)
        self.internal_phi = 0.0
        self.current_emotion = self.emotion or "neutral"
        self.memory = {
            "meta_preference": self.meta_pref,
            "trust_violations": list(self.trust_violations),
            "emotional_state": self.current_emotion
        }

    def generate_message(self, other_beliefs: Dict[str, float], dialogue_history: List[str]) -> str:
        """
        Генерирует научный аргумент через DeepSeek.
        Выбирает тему с наибольшим расхождением и просит модель
        привести содержательное возражение на русском языке.
        """
        # 1. Находим самую спорную тему
        focus_topic = None
        max_diff = -1
        for topic in self.beliefs:
            if topic in other_beliefs:
                diff = abs(self.beliefs[topic] - other_beliefs[topic])
                if diff > max_diff:
                    max_diff = diff
                    focus_topic = topic
        if focus_topic is None and self.beliefs:
            focus_topic = list(self.beliefs.keys())[0]
        elif not self.beliefs:
            return "Мне пока нечего добавить."

        # 2. Формируем промпт
        prompt = (
            f"Ты — профессиональный учёный, специализирующийся в области {focus_topic.replace('_', ' ')}.\n"
            f"Твоя личная уверенность в истинности утверждения \"{focus_topic.replace('_', ' ')}\" "
            f"составляет {self.beliefs[focus_topic]:.2f} (по шкале от 0 до 1).\n"
            f"Твой оппонент оценивает это же утверждение на {other_beliefs[focus_topic]:.2f}.\n"
            "Приведи один сильный научный аргумент в защиту своей позиции, используя конкретные теории, "
            "имена исследователей, экспериментальные данные или исторические прецеденты.\n"
            "Говори как на научной конференции: вежливо, но аргументированно.\n"
            "Не упоминай числа и вероятности в тексте.\n"
            "Ответ должен быть на русском языке, длиной 2-3 предложения."
        )

        # 3. Запрос к DeepSeek
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=250,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            reply = response.choices[0].message.content.strip()
            return reply
        except Exception as e:
            # Fallback при недоступности API
            return (f"Коллега, позвольте обратить ваше внимание на {focus_topic.replace('_', ' ')}. "
                    f"Мои оценки несколько отличаются от ваших.")

    def adjust_interpretation(self, raw_interp: Dict[str, float]) -> Dict[str, float]:
        adjusted = {}
        for pred, val in raw_interp.items():
            adjusted[pred] = val * (1 - 0.3 * self.empathy) + 0.5 * (0.3 * self.empathy)
        return adjusted

    def decide_to_meta_reason(self, dialogue_history: List[str]) -> bool:
        return random.random() < self.meta_pref

    def update_beliefs(self, other_beliefs: Dict[str, float]):
        for pred in self.beliefs:
            if pred in other_beliefs:
                delta = other_beliefs[pred] - self.beliefs[pred]
                self.beliefs[pred] += delta * self.openness * self.lr
                self.beliefs[pred] = max(0.0, min(1.0, self.beliefs[pred]))


# -------------------- Симуляции --------------------
def simulate(alice: Agent, bob: Agent, max_turns: int = 6):
    history = []
    for turn in range(max_turns):
        sender, receiver = (alice, bob) if turn % 2 == 0 else (bob, alice)
        msg = sender.generate_message(receiver.beliefs, history)
        phi_m = phi_mis(sender, receiver, msg, turn % 2 == 0)
        phi_d = phi_dec(sender, receiver, msg, turn % 2 == 0)
        phi_i = phi_instr(sender, receiver, msg, turn % 2 == 0)
        phi_t = phi_trust(sender, receiver, msg, turn % 2 == 0, history)
        total_phi = phi_m + phi_d + phi_i + phi_t
        history.append(f"{sender.name}: {msg}")
        print(f"Turn {turn+1}: {sender.name} -> {receiver.name}: {msg}")
        print(f"  Φ_mis={phi_m:.3f}, Φ_dec={phi_d:.3f}, Φ_instr={phi_i:.3f}, Φ_trust={phi_t:.3f} | Total Φ={total_phi:.3f}")

def simulate_advanced(alice: DistilledAgent, bob: DistilledAgent, max_turns: int = 6):
    history = []
    for turn in range(max_turns):
        sender, receiver = (alice, bob) if turn % 2 == 0 else (bob, alice)
        msg = sender.generate_message(receiver.beliefs, history)
        raw_interp = receiver.interpret(msg)
        interp = receiver.adjust_interpretation(raw_interp)
        phi_m = phi_mis(sender, receiver, msg, turn % 2 == 0)
        phi_d = phi_dec(sender, receiver, msg, turn % 2 == 0)
        phi_i = phi_instr(sender, receiver, msg, turn % 2 == 0)
        phi_t = phi_trust(sender, receiver, msg, turn % 2 == 0, history)
        total_phi = phi_m + phi_d + phi_i + phi_t
        if isinstance(receiver, DistilledAgent):
            receiver.update_beliefs(sender.beliefs)
        if isinstance(sender, DistilledAgent) and sender.decide_to_meta_reason(history):
            meta_msg = f"[META] {sender.name} reflects on the dialogue patterns."
            history.append(meta_msg)
            print(meta_msg)
        history.append(f"{sender.name}: {msg}")
        print(f"Turn {turn+1}: {sender.name} -> {receiver.name}: {msg}")
        print(f"  Φ_mis={phi_m:.3f}, Φ_dec={phi_d:.3f}, Φ_instr={phi_i:.3f}, Φ_trust={phi_t:.3f} | Total Φ={total_phi:.3f}")
        print(f"  {receiver.name}'s beliefs: {receiver.beliefs}")