import streamlit as st
import os, sys, io, json, itertools, random
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
import dialogue_core
from dialogue_core import SubjectivityProfile, DistilledAgent, simulate_advanced, Agent

# ---------------- Настройки страницы ----------------
st.set_page_config(page_title="GRA Multiverse Lab", layout="wide")
st.title("🌀 GRA InterSubjectivity Layer – Лаборатория диалогов")

# ---------------- Инициализация сессионных данных ----------------
if 'profiles' not in st.session_state:
    st.session_state.profiles = {}  # {имя: SubjectivityProfile}
if 'custom_agents' not in st.session_state:
    st.session_state.custom_agents = {}  # {имя: DistilledAgent или Agent}
if 'log' not in st.session_state:
    st.session_state.log = ""
if 'phi_history' not in st.session_state:
    st.session_state.phi_history = None

# ---------------- Загрузка существующих профилей из папки profiles/ ----------------
PROFILES_DIR = "profiles"
if os.path.isdir(PROFILES_DIR):
    for fname in sorted(os.listdir(PROFILES_DIR)):
        if fname.endswith(".json"):
            path = os.path.join(PROFILES_DIR, fname)
            try:
                profile = SubjectivityProfile.from_json(path)
                st.session_state.profiles[profile.agent_id] = profile
            except Exception as e:
                st.warning(f"Ошибка загрузки {fname}: {e}")

# ---------------- Функции создания агентов из профилей ----------------
def get_all_agents():
    """Объединяет загруженные профили и вручную созданных агентов."""
    agents = {}
    for name, prof in st.session_state.profiles.items():
        agents[name] = DistilledAgent(prof)
    for name, agent in st.session_state.custom_agents.items():
        agents[name] = agent
    return agents

def run_dialogue(agent1, agent2, turns):
    """Запускает диалог, возвращает лог и историю пены."""
    f = io.StringIO()
    phi_series = []
    # Кастомный цикл, чтобы вытащить пену
    history = []
    for turn in range(turns):
        sender, receiver = (agent1, agent2) if turn % 2 == 0 else (agent2, agent1)
        msg = sender.generate_message(receiver.beliefs, history)
        phi_m = dialogue_core.phi_mis(sender, receiver, msg, turn % 2 == 0)
        phi_d = dialogue_core.phi_dec(sender, receiver, msg, turn % 2 == 0)
        phi_i = dialogue_core.phi_instr(sender, receiver, msg, turn % 2 == 0)
        phi_t = dialogue_core.phi_trust(sender, receiver, msg, turn % 2 == 0, history)
        total = phi_m + phi_d + phi_i + phi_t
        phi_series.append((turn+1, phi_m, phi_d, phi_i, phi_t, total))
        if isinstance(receiver, DistilledAgent):
            receiver.update_beliefs(sender.beliefs)
        if isinstance(sender, DistilledAgent) and sender.decide_to_meta_reason(history):
            history.append(f"[META] {sender.name} reflects on the dialogue patterns.")
        history.append(f"{sender.name}: {msg}")
        # Вывод в лог
        f.write(f"Turn {turn+1}: {sender.name} -> {receiver.name}: {msg}\n")
        f.write(f"  Φ_mis={phi_m:.3f}, Φ_dec={phi_d:.3f}, Φ_instr={phi_i:.3f}, Φ_trust={phi_t:.3f} | Total Φ={total:.3f}\n")
    log = f.getvalue()
    return log, phi_series

# ---------------- Интерфейс: сайдбар --------------------
with st.sidebar:
    st.header("⚙️ Управление агентами")

    # ----- Загрузка профиля из файла -----
    uploaded_file = st.file_uploader("📤 Загрузить профиль (.json)", type="json")
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            sp = data["subjectivity_profile"]
            metrics = data.get("dialogue_metrics", {})
            profile = SubjectivityProfile(
                agent_id=data.get("agent_name", "Uploaded"),
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
            st.session_state.profiles[profile.agent_id] = profile
            st.success(f"Профиль '{profile.agent_id}' загружен!")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

    st.markdown("---")
    st.subheader("🛠 Создать агента вручную")
    with st.form("custom_agent"):
        name = st.text_input("Имя агента", value="Custom")
        beliefs_str = st.text_area("Убеждения (JSON)", 
                                   value='{"rain": 0.5, "quantum_weird": 0.5, "climate_change": 0.5}')
        openness = st.slider("Открытость", 0.0, 1.0, 0.3)
        empathy = st.slider("Эмпатия", 0.0, 1.0, 0.5)
        learning_rate = st.slider("Скорость обучения", 0.0, 1.0, 0.5)
        meta_pref = st.slider("Склонность к мета-рассуждениям", 0.0, 1.0, 0.1)
        submitted = st.form_submit_button("Добавить агента")
        if submitted:
            try:
                beliefs = json.loads(beliefs_str)
                prof = SubjectivityProfile(
                    agent_id=name,
                    initial_beliefs=beliefs,
                    openness=openness,
                    empathy=empathy,
                    learning_rate=learning_rate,
                    meta_preference=meta_pref
                )
                st.session_state.profiles[name] = prof
                st.success(f"Агент '{name}' добавлен!")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    # ----- Редактирование и удаление существующих агентов -----
    st.markdown("---")
    st.subheader("📋 Редактировать агента")
    all_agent_names = list(st.session_state.profiles.keys()) + list(st.session_state.custom_agents.keys())
    if not all_agent_names:
        st.info("Пока нет ни одного агента.")
    else:
        selected_agent = st.selectbox("Выберите агента", all_agent_names)
        if selected_agent:
            # Определяем источник данных
            if selected_agent in st.session_state.profiles:
                prof = st.session_state.profiles[selected_agent]
                is_profile = True
                agent_type = "DistilledAgent (из профиля)"
                current_beliefs = json.dumps(prof.initial_beliefs, indent=2, ensure_ascii=False)
                current_openness = prof.openness
                current_empathy = prof.empathy
                current_lr = prof.learning_rate
                current_meta = prof.meta_preference
            else:
                agent = st.session_state.custom_agents[selected_agent]
                is_profile = False
                agent_type = "Обычный Agent (вручную)"
                current_beliefs = json.dumps(agent.beliefs, indent=2, ensure_ascii=False)
                # У обычного агента нет таких параметров, ставим значения по умолчанию
                current_openness = 0.3
                current_empathy = 0.5
                current_lr = 0.5
                current_meta = 0.1

            st.write(f"**Тип:** {agent_type}")
            # Редактирование убеждений
            new_beliefs_str = st.text_area("Убеждения (JSON)", value=current_beliefs, height=150, key=f"edit_beliefs_{selected_agent}")
            # Ползунки для когнитивных параметров (только если профиль)
            if is_profile:
                new_openness = st.slider("Открытость", 0.0, 1.0, current_openness, key=f"edit_openness_{selected_agent}")
                new_empathy = st.slider("Эмпатия", 0.0, 1.0, current_empathy, key=f"edit_empathy_{selected_agent}")
                new_lr = st.slider("Скорость обучения", 0.0, 1.0, current_lr, key=f"edit_lr_{selected_agent}")
                new_meta = st.slider("Мета-рассуждения", 0.0, 1.0, current_meta, key=f"edit_meta_{selected_agent}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Сохранить изменения", key=f"save_{selected_agent}"):
                    try:
                        new_beliefs = json.loads(new_beliefs_str)
                        if is_profile:
                            # Обновляем профиль
                            prof.initial_beliefs = new_beliefs
                            prof.openness = new_openness
                            prof.empathy = new_empathy
                            prof.learning_rate = new_lr
                            prof.meta_preference = new_meta
                            # Пересоздаём DistilledAgent в custom_agents не хранится, он генерируется на лету
                            st.success(f"Профиль '{selected_agent}' обновлён!")
                        else:
                            # Обновляем обычного агента
                            agent = st.session_state.custom_agents[selected_agent]
                            agent.beliefs = new_beliefs
                            st.success(f"Агент '{selected_agent}' обновлён!")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
            with col2:
                if st.button("🗑 Удалить агента", key=f"delete_{selected_agent}"):
                    if is_profile:
                        del st.session_state.profiles[selected_agent]
                    else:
                        del st.session_state.custom_agents[selected_agent]
                    st.success(f"Агент '{selected_agent}' удалён!")
                    st.experimental_rerun()

# ---------------- Основная рабочая область: три вкладки ----------------
tab1, tab2, tab3 = st.tabs(["🎭 Диалог", "📊 Наблюдатель", "📈 Аналитика"])

# ===== Вкладка 1: Прямой диалог =====
with tab1:
    st.header("Прямой диалог двух агентов")
    all_agents = get_all_agents()
    if len(all_agents) < 2:
        st.warning("Нужно минимум два агента. Загрузите профили или создайте вручную.")
    else:
        names = list(all_agents.keys())
        col1, col2 = st.columns(2)
        with col1:
            agent1_name = st.selectbox("Первый агент", names, key="a1")
        with col2:
            agent2_name = st.selectbox("Второй агент", names, index=min(1, len(names)-1), key="a2")
        turns = st.slider("Раундов диалога", 2, 20, 6, key="turns")
        if st.button("▶ Запустить диалог", key="run_dialogue"):
            agent1 = all_agents[agent1_name]
            agent2 = all_agents[agent2_name]
            log, phi_hist = run_dialogue(agent1, agent2, turns)
            st.session_state.log = log
            st.session_state.phi_history = phi_hist
            st.subheader("📜 Лог диалога")
            st.text(log)
            st.subheader("📊 Финальные убеждения")
            colb1, colb2 = st.columns(2)
            with colb1:
                st.write(f"**{agent1.name}**")
                st.json(agent1.beliefs)
            with colb2:
                st.write(f"**{agent2.name}**")
                st.json(agent2.beliefs)
            # График пены
            if phi_hist:
                df = pd.DataFrame(phi_hist, columns=["Ход", "Φ_mis", "Φ_dec", "Φ_instr", "Φ_trust", "Total Φ"])
                st.subheader("📈 Динамика пены")
                fig, ax = plt.subplots(figsize=(8,4))
                for col in ["Φ_mis", "Φ_dec", "Φ_instr", "Φ_trust"]:
                    ax.plot(df["Ход"], df[col], marker='o', label=col)
                ax.set_xlabel("Ход")
                ax.set_ylabel("Значение Φ")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                st.download_button("💾 Скачать лог", log, "dialogue_log.txt")

# ===== Вкладка 2: Наблюдатель (все пары) =====
with tab2:
    st.header("👁 Режим наблюдателя: все возможные пары")
    all_agents = get_all_agents()
    if len(all_agents) < 2:
        st.info("Добавьте хотя бы двух агентов для наблюдения.")
    else:
        turns_obs = st.slider("Раундов на диалог", 2, 10, 4, key="obs_turns")
        if st.button("🔍 Запустить наблюдение", key="run_observer"):
            names = list(all_agents.keys())
            pairs = list(itertools.permutations(names, 2))
            results = []
            progress = st.progress(0)
            for i, (name_a, name_b) in enumerate(pairs):
                # Создаём свежих агентов из профилей/кастомных
                agent_a = all_agents[name_a]  # get_all_agents уже возвращает новых
                agent_b = all_agents[name_b]
                # Сбрасываем историю и убеждения до исходных, если нужно
                if name_a in st.session_state.profiles:
                    agent_a = DistilledAgent(st.session_state.profiles[name_a])
                else:
                    agent_a = st.session_state.custom_agents[name_a]
                if name_b in st.session_state.profiles:
                    agent_b = DistilledAgent(st.session_state.profiles[name_b])
                else:
                    agent_b = st.session_state.custom_agents[name_b]
                log, phi_hist = run_dialogue(agent_a, agent_b, turns_obs)
                final_phi = phi_hist[-1][-1] if phi_hist else 0.0
                results.append({
                    "Пара": f"{name_a} → {name_b}",
                    "Итоговая Φ": final_phi,
                    "Убеждения A": agent_a.beliefs,
                    "Убеждения B": agent_b.beliefs
                })
                progress.progress((i+1)/len(pairs))
            st.session_state.observer_results = results
            st.success(f"Завершено {len(results)} диалогов.")
        if 'observer_results' in st.session_state and st.session_state.observer_results:
            df = pd.DataFrame(st.session_state.observer_results)
            st.subheader("📋 Таблица результатов")
            st.dataframe(df[["Пара", "Итоговая Φ"]])
            st.subheader("📈 Распределение итоговой Φ")
            fig2, ax2 = plt.subplots()
            ax2.hist(df["Итоговая Φ"], bins=10, alpha=0.7)
            ax2.set_xlabel("Φ")
            ax2.set_ylabel("Число пар")
            st.pyplot(fig2)
            selected_pair = st.selectbox("Выберите пару для деталей", df["Пара"].tolist())
            if selected_pair:
                row = df[df["Пара"] == selected_pair].iloc[0]
                st.write("**Финальные убеждения A:**", row["Убеждения A"])
                st.write("**Финальные убеждения B:**", row["Убеждения B"])

# ===== Вкладка 3: Сравнительная аналитика =====
with tab3:
    st.header("📈 Сравнительная аналитика")
    if 'observer_results' in st.session_state and st.session_state.observer_results:
        df = pd.DataFrame(st.session_state.observer_results)
        st.write("Тепловая карта взаимной пены (матрица агентов)")
        agents_list = sorted(list(get_all_agents().keys()))
        matrix = pd.DataFrame(index=agents_list, columns=agents_list)
        for _, r in df.iterrows():
            a, b = r["Пара"].split(" → ")
            matrix.loc[a, b] = r["Итоговая Φ"]
        st.dataframe(matrix.style.background_gradient(cmap='Reds'))
    else:
        st.info("Сначала запустите наблюдение во вкладке 'Наблюдатель'.")