#!/usr/bin/env python3
"""
GRA InterSubjectivity Dialogue Core
- базовые функции расчёта пены (Φ_mis, Φ_dec, Φ_instr, Φ_trust)
- класс Agent с экспертным генератором сообщений
- класс SubjectivityProfile для загрузки асимметричных профилей
- класс DistilledAgent с когнитивными параметрами
- стандартная и расширенная симуляции
"""

import json
import os
import math
import random
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy

# -------------------- Вспомогательные константы --------------------
EPS = 1e-9

# -------------------- Базовый класс Agent --------------------
class Agent:
    """Обычный агент с убеждениями и ценностями (из оригинального dialogue_core.py)."""
    def __init__(self, name: str, beliefs: Dict[str, float], values: List[str] = None):
        self.name = name
        self.beliefs = beliefs  # {predicate: probability}
        self.values = values or ["honesty", "clarity"]
        self.history = []

    def generate_message(self, other_beliefs: Dict[str, float], dialogue_history: List[str]) -> str:
        """
        Генерирует экспертное высказывание на основе своих убеждений.
        Преобразует вероятности в словесные оценки и подчёркивает расхождения.
        """
        # Если убеждений нет, возвращаем нейтральное сообщение
        if not self.beliefs:
            return "У меня пока нет сформированных убеждений."

        statements = []
        for topic, prob in self.beliefs.items():
            # Определяем словесную оценку уверенности
            if prob > 0.9:
                confidence = "абсолютно уверен"
            elif prob > 0.7:
                confidence = "практически уверен"
            elif prob > 0.5:
                confidence = "склоняюсь к тому, что"
            elif prob > 0.3:
                confidence = "сомневаюсь, но допускаю"
            else:
                confidence = "считаю маловероятным"

            # Преобразуем ключ в читаемый вид (заменяем подчёркивания на пробелы)
            human_topic = topic.replace("_", " ")
            statement = f"Я {confidence}, что {human_topic} (вероятность {prob:.2f})"
            statements.append(statement)

        # Находим самые сильные расхождения с собеседником (если есть данные)
        if other_beliefs:
            differences = []
            for topic in self.beliefs:
                if topic in other_beliefs:
                    diff = abs(self.beliefs[topic] - other_beliefs[topic])
                    differences.append((topic, diff))
            differences.sort(key=lambda x: x[1], reverse=True)
            # Выбираем до двух тем с расхождением > 0.15
            focus_topics = [t for t, d in differences[:2] if d > 0.15]
            if focus_topics:
                topic_name = focus_topics[0].replace("_", " ")
                my_val = self.beliefs[focus_topics[0]]
                other_val = other_beliefs[focus_topics[0]]
                msg = (f"Коллега, хочу обратить внимание на {topic_name}. "
                       f"Моя оценка: {my_val:.2f}, у вас – {other_val:.2f}. "
                       f"Мои аргументы основаны на последних публикациях.")
                return msg

        # Если сильных расхождений нет, делимся несколькими своими оценками
        return "Позвольте поделиться моими текущими оценками: " + "; ".join(statements[:3])

    def interpret(self, message: str) -> Dict[str, float]:
        """Парсит сообщение, пытаясь извлечь убеждения собеседника."""
        # Пытаемся извлечь словарь из старого формата "My beliefs: ..."
        try:
            if "My beliefs:" in message:
                dict_str = message.split(": ", 1)[1]
                interp = eval(dict_str)
                if isinstance(interp, dict):
                    return interp
        except:
            pass

        # Новый экспертный формат: ищем числа после фразы "вероятность X.XX"
        import re
        extracted = {}
        for topic in self.beliefs:
            human_topic = topic.replace("_", " ")
            # Ищем упоминание темы и число рядом
            pattern = rf'{re.escape(human_topic)}.*?(\d+\.\d+)'
            match = re.search(pattern, message)
            if match:
                extracted[topic] = float(match.group(1))
        if extracted:
            return extracted
        # Если ничего не нашли, возвращаем нейтральное значение 0.5
        return {k: 0.5 for k in self.beliefs}

    def __repr__(self):
        return f"Agent({self.name})"


# -------------------- Функции пены (оригинальные) --------------------
def phi_mis(sender: Agent, receiver: Agent, message: str, sender_turn: bool) -> float:
    """
    Недопонимание: разница между тем, что сказал sender,
    и тем, как receiver интерпретировал.
    """
    intended = sender.beliefs
    interp = receiver.interpret(message)
    diff = 0.0
    for k in intended:
        if k in interp:
            diff += (intended[k] - interp[k]) ** 2
    return math.sqrt(diff) if diff > 0 else 0.0

def phi_dec(sender: Agent, receiver: Agent, message: str, sender_turn: bool) -> float:
    """
    Обман: расхождение между убеждениями sender'а и сказанным.
    (Если sender намеренно искажает правду, Φ_dec растёт).
    """
    interp = receiver.interpret(message)
    diff = 0.0
    for k in sender.beliefs:
        if k in interp:
            diff += (sender.beliefs[k] - interp[k]) ** 2
    return math.sqrt(diff) if diff > 0 else 0.0

def phi_instr(sender: Agent, receiver: Agent, message: str, sender_turn: bool) -> float:
    """
    Инструментализация (нарушение Закона Алана): если один субъект
    низводит другого до средства. Упрощённо: если сообщение содержит приказ
    или манипулятивный паттерн.
    """
    lower_msg = message.lower()
    if "you must" in lower_msg or "do this" in lower_msg or "just follow" in lower_msg:
        return 1.0
    return 0.0

def phi_trust(sender: Agent, receiver: Agent, message: str, sender_turn: bool,
              history: List[str]) -> float:
    """
    Разрушение доверия: сумма предыдущих актов обмана/инструментализации
    и грубых ошибок интерпретации.
    """
    trust_loss = 0.0
    for past_msg in history[-3:]:  # анализируем последние 3 сообщения
        if "deception" in past_msg.lower() or "instrumental" in past_msg.lower():
            trust_loss += 0.2
    return min(1.0, trust_loss)


# -------------------- Загрузка профилей --------------------
@dataclass
class SubjectivityProfile:
    """
    Хранилище всех когнитивных и эмоциональных параметров агента,
    загружаемое из JSON-профиля.
    """
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
        """Загрузка из JSON-файла, сгенерированного example_dialogue.py (расширенного)."""
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


# -------------------- Расширенный агент с когнитивными параметрами --------------------
class DistilledAgent(Agent):
    """
    Асимметричный агент, «дистиллированный» из загруженного профиля.
    Наследует базовые поля Agent (name, beliefs, values) и добавляет
    индивидуальные когнитивные параметры, влияющие на ход диалога.
    """
    def __init__(self, profile: SubjectivityProfile, default_values: List[str] = None):
        super().__init__(
            name=profile.agent_id,
            beliefs=profile.initial_beliefs,
            values=default_values or ["honesty", "clarity"]
        )
        self.profile = profile
        self.openness = profile.openness          # готовность менять убеждения
        self.empathy = profile.empathy             # способность к пониманию
        self.lr = profile.learning_rate            # скорость обучения
        self.meta_pref = profile.meta_preference   # склонность к мета-рассуждениям
        self.emotion = profile.emotional_state
        self.trust_violations = set(profile.trust_violations)

        # Внутреннее состояние для формул пены
        self.internal_phi = 0.0
        self.current_emotion = self.emotion or "neutral"

        # Слоты для совместимости с GRA-Subjectivity-Layer
        self.memory = {
            "meta_preference": self.meta_pref,
            "trust_violations": list(self.trust_violations),
            "emotional_state": self.current_emotion
        }

    def adjust_interpretation(self, raw_interp: Dict[str, float]) -> Dict[str, float]:
        """
        Корректирует интерпретацию сообщения с учётом эмпатии и открытости.
        Эмпатичный агент лучше «слышит» собеседника, смещая интерпретацию
        к его предполагаемой позиции.
        """
        adjusted = {}
        for pred, val in raw_interp.items():
            # Эмпатия сдвигает интерпретацию к 0.5 (нейтраль), снижая крайности
            adjusted[pred] = val * (1 - 0.3 * self.empathy) + 0.5 * (0.3 * self.empathy)
        return adjusted

    def decide_to_meta_reason(self, dialogue_history: List[str]) -> bool:
        """Вероятность мета-рассуждения зависит от meta_pref."""
        return random.random() < self.meta_pref

    def update_beliefs(self, other_beliefs: Dict[str, float]):
        """
        Обновление убеждений с учётом открытости и скорости обучения.
        """
        for pred in self.beliefs:
            if pred in other_beliefs:
                delta = other_beliefs[pred] - self.beliefs[pred]
                # Открытость определяет, насколько агент готов принять чужое мнение
                self.beliefs[pred] += delta * self.openness * self.lr
                # Ограничиваем диапазоном [0, 1]
                self.beliefs[pred] = max(0.0, min(1.0, self.beliefs[pred]))


# -------------------- Симуляции --------------------
def simulate(alice: Agent, bob: Agent, max_turns: int = 6):
    """
    Базовая симуляция диалога (из оригинального dialogue_core.py).
    """
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
    """
    Расширенная симуляция, использующая асимметричные параметры агентов.
    Агенты применяют adjust_interpretation, update_beliefs и мета-рассуждения.
    """
    history = []
    for turn in range(max_turns):
        sender, receiver = (alice, bob) if turn % 2 == 0 else (bob, alice)

        # Генерация сообщения
        msg = sender.generate_message(receiver.beliefs, history)

        # Интерпретация с учётом эмпатии
        raw_interp = receiver.interpret(msg)
        interp = receiver.adjust_interpretation(raw_interp)

        # Расчёт компонентов пены (используем базовые функции)
        phi_m = phi_mis(sender, receiver, msg, turn % 2 == 0)
        phi_d = phi_dec(sender, receiver, msg, turn % 2 == 0)
        phi_i = phi_instr(sender, receiver, msg, turn % 2 == 0)
        phi_t = phi_trust(sender, receiver, msg, turn % 2 == 0, history)
        total_phi = phi_m + phi_d + phi_i + phi_t

        # Обновление убеждений получателя (если DistilledAgent)
        if isinstance(receiver, DistilledAgent):
            receiver.update_beliefs(sender.beliefs)

        # Мета-рассуждение (если DistilledAgent)
        if isinstance(sender, DistilledAgent) and sender.decide_to_meta_reason(history):
            meta_msg = f"[META] {sender.name} reflects on the dialogue patterns."
            history.append(meta_msg)
            print(meta_msg)

        history.append(f"{sender.name}: {msg}")
        print(f"Turn {turn+1}: {sender.name} -> {receiver.name}: {msg}")
        print(f"  Φ_mis={phi_m:.3f}, Φ_dec={phi_d:.3f}, Φ_instr={phi_i:.3f}, Φ_trust={phi_t:.3f} | Total Φ={total_phi:.3f}")
        print(f"  {receiver.name}'s beliefs: {receiver.beliefs}")