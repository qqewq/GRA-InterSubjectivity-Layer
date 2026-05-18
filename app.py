import streamlit as st
import os
import sys

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))

from dialogue_core import SubjectivityProfile, DistilledAgent, simulate_advanced, Agent

st.set_page_config(page_title="GRA InterSubjectivity Layer", layout="wide")
st.title("🌀 GRA InterSubjectivity – Веб-симуляция диалога")

# --- Загрузка профилей из папки profiles/ ---
profiles_dir = "profiles"
available_profiles = []
if os.path.isdir(profiles_dir):
    for fname in sorted(os.listdir(profiles_dir)):
        if fname.endswith(".json"):
            try:
                profile = SubjectivityProfile.from_json(os.path.join(profiles_dir, fname))
                available_profiles.append(profile)
            except Exception as e:
                st.warning(f"Не удалось загрузить {fname}: {e}")

if len(available_profiles) < 2:
    st.info("Профилей недостаточно, используются стандартные агенты.")
    agents_dict = {
        "Alice": Agent("Alice", {"rain": 0.9, "quantum_weird": 0.2, "climate_change": 0.95},
                       ["honesty", "clarity"]),
        "Bob": Agent("Bob", {"rain": 0.1, "quantum_weird": 0.8, "climate_change": 0.9},
                     ["honesty", "cooperation"]),
    }
else:
    agents_dict = {p.agent_id: DistilledAgent(p) for p in available_profiles}

# --- Выбор агентов ---
agent_names = list(agents_dict.keys())
col1, col2 = st.columns(2)
with col1:
    agent1_name = st.selectbox("Первый агент", agent_names, index=0)
with col2:
    # Второй агент по умолчанию – следующий в списке (или первый, если он один)
    idx2 = 1 if len(agent_names) > 1 else 0
    agent2_name = st.selectbox("Второй агент", agent_names, index=idx2)

max_turns = st.slider("Количество шагов диалога", min_value=2, max_value=20, value=6)

# --- Запуск симуляции ---
if st.button("▶ Запустить диалог"):
    agent1 = agents_dict[agent1_name]
    agent2 = agents_dict[agent2_name]

    # Перехватываем вывод simulate_advanced, чтобы отобразить в интерфейсе
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        if isinstance(agent1, DistilledAgent) and isinstance(agent2, DistilledAgent):
            simulate_advanced(agent1, agent2, max_turns=max_turns)
        else:
            from dialogue_core import simulate
            simulate(agent1, agent2, max_turns=max_turns)
    output = f.getvalue()

    st.subheader("📜 Результаты диалога")
    st.text(output)

    # Визуализация финальных убеждений
    st.subheader("📊 Итоговые убеждения агентов")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{agent1.name}**")
        st.json(agent1.beliefs)
    with col2:
        st.write(f"**{agent2.name}**")
        st.json(agent2.beliefs)

    # Кнопка для скачивания лога
    st.download_button(
        label="Скачать лог диалога",
        data=output,
        file_name="dialogue_log.txt",
        mime="text/plain"
    )