import streamlit as st
import os, sys, io, json, itertools, random
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
import dialogue_core
from dialogue_core import SubjectivityProfile, DistilledAgent, simulate_advanced, Agent

st.set_page_config(page_title="GRA Multiverse Lab", layout="wide")
st.title("🌀 GRA InterSubjectivity Layer – Лаборатория диалогов")

# ---------------- Инициализация сессионных данных ----------------
if 'profiles' not in st.session_state:
    st.session_state.profiles = {}
if 'custom_agents' not in st.session_state:
    st.session_state.custom_agents = {}
if 'log' not in st.session_state:
    st.session_state.log = ""
if 'phi_history' not in st.session_state:
    st.session_state.phi_history = None
if 'last_agents' not in st.session_state:
    st.session_state.last_agents = None  # для сохранения после диалога

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

def get_all_agents():
    agents = {}
    for name, prof in st.session_state.profiles.items():
        agents[name] = DistilledAgent(prof)
    for name, agent in st.session_state.custom_agents.items():
        agents[name] = agent
    return agents

def adapt_agent_parameters(agent, phi_series):
    """
    Адаптирует когнитивные параметры DistilledAgent на основе истории пены диалога.
    phi_series: список кортежей (turn, phi_mis, phi_dec, phi_instr, phi_trust, total)
    """
    if not isinstance(agent, DistilledAgent) or not phi_series:
        return
    # Средние значения компонент пены за диалог
    avg = {k: 0.0 for k in ["mis", "dec", "instr", "trust"]}
    n = len(phi_series)
    for _, mis, dec, instr, trust, _ in phi_series:
        avg["mis"] += mis
        avg["dec"] += dec
        avg["instr"] += instr
        avg["trust"] += trust
    for k in avg:
        avg[k] /= n

    # Правило адаптации (можно менять коэффициенты)
    # Открытость: снижается при высоком недопонимании и инструментализации, растёт при доверии
    delta_openness = 0.02 * (avg["trust"] - avg["mis"] - avg["instr"])
    agent.openness = max(0.0, min(1.0, agent.openness + delta_openness))

    # Эмпатия: растёт при недопонимании (чтобы лучше слушать), падает при обмане
    delta_empathy = 0.02 * (avg["mis"] - avg["dec"])
    agent.empathy = max(0.0, min(1.0, agent.empathy + delta_empathy))

    # Скорость обучения: снижается при высоком обмане (недоверие к информации)
    delta_lr = -0.01 * avg["dec"]
    agent.lr = max(0.0, min(1.0, agent.lr + delta_lr))

    # Мета-рассуждения: растут при накоплении недопонимания и инструментализации
    delta_meta = 0.01 * (avg["mis"] + avg["instr"])
    agent.meta_pref = max(0.0, min(1.0, agent.meta_pref + delta_meta))

    # Обновляем эмоциональный фон (просто метку)
    if avg["trust"] < 0.3:
        agent.current_emotion = "разочарован"
    elif avg["mis"] < 0.2 and avg["dec"] < 0.2:
        agent.current_emotion = "удовлетворён"
    else:
        agent.current_emotion = "задумчив"

def run_dialogue(agent1, agent2, turns):
    """Запускает диалог, возвращает лог, историю пены и обновлённых агентов."""
    f = io.StringIO()
    phi_series = []
    history = []
    # Копируем убеждения, чтобы не портить исходные (для наблюдателя)
    agent1 = deepcopy(agent1)
    agent2 = deepcopy(agent2)

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
        f.write(f"Turn {turn+1}: {sender.name} -> {receiver.name}: {msg}\n")
        f.write(f"  Φ_mis={phi_m:.3f}, Φ_dec={phi_d:.3f}, Φ_instr={phi_i:.3f}, Φ_trust={phi_t:.3f} | Total Φ={total:.3f}\n")
    log = f.getvalue()

    # Адаптируем параметры агентов на основе диалога
    adapt_agent_parameters(agent1, phi_series)
    adapt_agent_parameters(agent2, phi_series)
    return log, phi_series, agent1, agent2

def save_agent_to_profile(agent, directory="profiles"):
    """Сохраняет агента (DistilledAgent) в JSON-файл в указанную папку."""
    if not isinstance(agent, DistilledAgent):
        raise ValueError("Только DistilledAgent можно сохранить в профиль.")
    os.makedirs(directory, exist_ok=True)
    profile = {
        "agent_name": agent.name,
        "subjectivity_profile": {
            "initial_beliefs": agent.beliefs,  # текущие убеждения становятся новыми начальными
            "openness": agent.openness,
            "empathy": agent.empathy,
            "learning_rate": agent.lr,
            "meta_preference": agent.meta_pref,
            "trust_violations": list(agent.trust_violations),
            "final_emotion": agent.current_emotion
        },
        "dialogue_metrics": {
            "convergence_rate": 0.0,
            "final_phi": agent.internal_phi
        }
    }
    filename = f"{agent.name}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    return filepath

# ---------------- Сайдбар: управление агентами ----------------
with st.sidebar:
    st.header("⚙️ Управление агентами")

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
        meta_pref = st.slider("Мета-рассуждения", 0.0, 1.0, 0.1)
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

    st.markdown("---")
    st.subheader("📋 Редактировать агента")
    all_agent_names = list(st.session_state.profiles.keys()) + list(st.session_state.custom_agents.keys())
    if not all_agent_names:
        st.info("Пока нет ни одного агента.")
    else:
        selected_agent = st.selectbox("Выберите агента", all_agent_names)
        if selected_agent:
            if selected_agent in st.session_state.profiles:
                prof = st.session_state.profiles[selected_agent]
                is_profile = True
                current_beliefs = json.dumps(prof.initial_beliefs, indent=2, ensure_ascii=False)
                cur_openness = prof.openness
                cur_empathy = prof.empathy
                cur_lr = prof.learning_rate
                cur_meta = prof.meta_preference
            else:
                agent = st.session_state.custom_agents[selected_agent]
                is_profile = False
                current_beliefs = json.dumps(agent.beliefs, indent=2, ensure_ascii=False)
                cur_openness = 0.3
                cur_empathy = 0.5
                cur_lr = 0.5
                cur_meta = 0.1

            new_beliefs_str = st.text_area("Убеждения", value=current_beliefs, height=150, key=f"edit_{selected_agent}")
            if is_profile:
                new_openness = st.slider("Открытость", 0.0, 1.0, cur_openness, key=f"o_{selected_agent}")
                new_empathy = st.slider("Эмпатия", 0.0, 1.0, cur_empathy, key=f"e_{selected_agent}")
                new_lr = st.slider("Скорость обучения", 0.0, 1.0, cur_lr, key=f"l_{selected_agent}")
                new_meta = st.slider("Мета-рассуждения", 0.0, 1.0, cur_meta, key=f"m_{selected_agent}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Сохранить", key=f"save_{selected_agent}"):
                    try:
                        new_beliefs = json.loads(new_beliefs_str)
                        if is_profile:
                            prof.initial_beliefs = new_beliefs
                            prof.openness = new_openness
                            prof.empathy = new_empathy
                            prof.learning_rate = new_lr
                            prof.meta_preference = new_meta
                        else:
                            agent.beliefs = new_beliefs
                        st.success("Изменения сохранены!")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
            with col2:
                if st.button("🗑 Удалить", key=f"del_{selected_agent}"):
                    if is_profile:
                        del st.session_state.profiles[selected_agent]
                    else:
                        del st.session_state.custom_agents[selected_agent]
                    st.success(f"'{selected_agent}' удалён.")
                    st.experimental_rerun()

# ---------------- Вкладки ----------------
tab1, tab2, tab3 = st.tabs(["🎭 Диалог", "📊 Наблюдатель", "📈 Аналитика"])

with tab1:
    st.header("Прямой диалог двух агентов")
    all_agents = get_all_agents()
    if len(all_agents) < 2:
        st.warning("Нужно минимум два агента.")
    else:
        names = list(all_agents.keys())
        col1, col2 = st.columns(2)
        with col1:
            a1_name = st.selectbox("Первый агент", names, key="a1")
        with col2:
            a2_name = st.selectbox("Второй агент", names, index=min(1, len(names)-1), key="a2")
        turns = st.slider("Раундов", 2, 20, 6)
        if st.button("▶ Запустить диалог"):
            agent1 = all_agents[a1_name]
            agent2 = all_agents[a2_name]
            log, phi_hist, updated1, updated2 = run_dialogue(agent1, agent2, turns)
            st.session_state.log = log
            st.session_state.phi_history = phi_hist
            st.session_state.last_agents = (updated1, updated2)
            st.subheader("📜 Лог диалога")
            st.text(log)
            st.subheader("📊 Финальные убеждения")
            colb1, colb2 = st.columns(2)
            with colb1:
                st.write(f"**{updated1.name}**")
                st.json(updated1.beliefs)
                st.write(f"Открытость: {updated1.openness:.2f}, Эмпатия: {updated1.empathy:.2f}, "
                         f"Скорость обучения: {updated1.lr:.2f}, Мета: {updated1.meta_pref:.2f}")
            with colb2:
                st.write(f"**{updated2.name}**")
                st.json(updated2.beliefs)
                st.write(f"Открытость: {updated2.openness:.2f}, Эмпатия: {updated2.empathy:.2f}, "
                         f"Скорость обучения: {updated2.lr:.2f}, Мета: {updated2.meta_pref:.2f}")
            # График пены
            if phi_hist:
                df = pd.DataFrame(phi_hist, columns=["Ход", "Φ_mis", "Φ_dec", "Φ_instr", "Φ_trust", "Total Φ"])
                st.subheader("📈 Динамика пены")
                fig, ax = plt.subplots(figsize=(8,4))
                for col in ["Φ_mis", "Φ_dec", "Φ_instr", "Φ_trust"]:
                    ax.plot(df["Ход"], df[col], marker='o', label=col)
                ax.set_xlabel("Ход")
                ax.set_ylabel("Φ")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            # Кнопки сохранения
            st.subheader("💾 Сохранить эволюционировавших агентов")
            col_save1, col_save2 = st.columns(2)
            with col_save1:
                if st.button(f"💾 Сохранить {updated1.name}"):
                    if isinstance(updated1, DistilledAgent):
                        try:
                            path = save_agent_to_profile(updated1, PROFILES_DIR)
                            st.success(f"Сохранён в {path}")
                            # Обновляем профиль в сессии, чтобы сразу подхватился
                            st.session_state.profiles[updated1.name] = SubjectivityProfile(
                                agent_id=updated1.name,
                                initial_beliefs=updated1.beliefs,
                                openness=updated1.openness,
                                empathy=updated1.empathy,
                                learning_rate=updated1.lr,
                                meta_preference=updated1.meta_pref,
                                trust_violations=list(updated1.trust_violations),
                                emotional_state=updated1.current_emotion
                            )
                        except Exception as e:
                            st.error(f"Ошибка сохранения: {e}")
                    else:
                        st.warning("Можно сохранить только агента с DeepSeek.")
                # Скачать JSON
                if isinstance(updated1, DistilledAgent):
                    json_str = json.dumps({
                        "agent_name": updated1.name,
                        "subjectivity_profile": {
                            "initial_beliefs": updated1.beliefs,
                            "openness": updated1.openness,
                            "empathy": updated1.empathy,
                            "learning_rate": updated1.lr,
                            "meta_preference": updated1.meta_pref,
                            "trust_violations": list(updated1.trust_violations),
                            "final_emotion": updated1.current_emotion
                        }
                    }, indent=2, ensure_ascii=False)
                    st.download_button("📥 Скачать JSON", json_str, f"{updated1.name}.json", key=f"dl_{updated1.name}")

            with col_save2:
                if st.button(f"💾 Сохранить {updated2.name}"):
                    if isinstance(updated2, DistilledAgent):
                        try:
                            path = save_agent_to_profile(updated2, PROFILES_DIR)
                            st.success(f"Сохранён в {path}")
                            st.session_state.profiles[updated2.name] = SubjectivityProfile(
                                agent_id=updated2.name,
                                initial_beliefs=updated2.beliefs,
                                openness=updated2.openness,
                                empathy=updated2.empathy,
                                learning_rate=updated2.lr,
                                meta_preference=updated2.meta_pref,
                                trust_violations=list(updated2.trust_violations),
                                emotional_state=updated2.current_emotion
                            )
                        except Exception as e:
                            st.error(f"Ошибка сохранения: {e}")
                    else:
                        st.warning("Можно сохранить только агента с DeepSeek.")
                if isinstance(updated2, DistilledAgent):
                    json_str = json.dumps({
                        "agent_name": updated2.name,
                        "subjectivity_profile": {
                            "initial_beliefs": updated2.beliefs,
                            "openness": updated2.openness,
                            "empathy": updated2.empathy,
                            "learning_rate": updated2.lr,
                            "meta_preference": updated2.meta_pref,
                            "trust_violations": list(updated2.trust_violations),
                            "final_emotion": updated2.current_emotion
                        }
                    }, indent=2, ensure_ascii=False)
                    st.download_button("📥 Скачать JSON", json_str, f"{updated2.name}.json", key=f"dl_{updated2.name}")

with tab2:
    st.header("👁 Режим наблюдателя: все пары")
    all_agents = get_all_agents()
    if len(all_agents) < 2:
        st.info("Добавьте хотя бы двух агентов.")
    else:
        turns_obs = st.slider("Раундов на диалог", 2, 10, 4, key="obs_turns")
        if st.button("🔍 Запустить наблюдение"):
            names = list(all_agents.keys())
            pairs = list(itertools.permutations(names, 2))
            results = []
            progress = st.progress(0)
            for i, (a, b) in enumerate(pairs):
                agent_a = all_agents[a]
                agent_b = all_agents[b]
                log, phi_hist, _, _ = run_dialogue(agent_a, agent_b, turns_obs)
                final_phi = phi_hist[-1][-1] if phi_hist else 0.0
                results.append({
                    "Пара": f"{a} → {b}",
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

with tab3:
    st.header("📈 Сравнительная аналитика")
    if 'observer_results' in st.session_state and st.session_state.observer_results:
        df = pd.DataFrame(st.session_state.observer_results)
        st.write("Тепловая карта взаимной пены")
        agents_list = sorted(list(get_all_agents().keys()))
        matrix = pd.DataFrame(index=agents_list, columns=agents_list)
        for _, r in df.iterrows():
            a, b = r["Пара"].split(" → ")
            matrix.loc[a, b] = r["Итоговая Φ"]
        st.dataframe(matrix.style.background_gradient(cmap='Reds'))
    else:
        st.info("Сначала запустите наблюдение.")