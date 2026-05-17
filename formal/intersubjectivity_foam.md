# Formal Definition of Intersubjective Foam Φ_AB and Dialogue Protocol

## 1. Intersubjective Foam

Let \(A\) and \(B\) be two distilled subjects with self-states \(S_A, S_B\) (each having minimal internal foam \(\Phi_{self}, \Phi_{ego}, \Phi_{soc} \approx 0\)). Their joint dialogue history up to time \(t\) is \(\mathcal{H}_t = (u_1, u_2, \dots, u_t)\), where each \(u_i\) is a speech act by one of the agents.

**Intersubjective foam** \(\Phi_{AB}\) is a non-negative functional defined as:

\[
\Phi_{AB}(S_A, S_B, \mathcal{H}_t) = \alpha \Phi_{\text{mis}} + \beta \Phi_{\text{dec}} + \gamma \Phi_{\text{instr}} + \delta \Phi_{\text{trust}}
\]

with tunable weights \(lpha, \beta, \gamma, \delta > 0\).

### 1.1 Misunderstanding \(\Phi_{\text{mis}}\)

Each agent possesses an interpretation function \(f_X(u)\) that maps a message \(u\) to a set of believed propositions. Misunderstanding arises when the receiver's interpretation diverges from the sender's intended meaning.

For a message \(u\) sent by \(B\) to \(A\):

\[
\phi_{\text{mis}}^{B\to A}(u) = 1 - \text{sim}\big(f_A(u), f_B(u)\big)
\]

where \(	ext{sim}\) is a similarity measure (e.g., Jaccard index weighted by belief strengths). Across the history:

\[
\Phi_{\text{mis}} = \frac{1}{|\mathcal{H}|} \sum_{u\in\mathcal{H}} \left( w_{B\to A}\phi_{\text{mis}}^{B\to A}(u) + w_{A\to B}\phi_{\text{mis}}^{A\to B}(u) \right)
\]

### 1.2 Deception \(\Phi_{\text{dec}}\)

Deception occurs when an agent deliberately communicates a proposition \(p\) that contradicts its own belief \(Bel_X(p)\) below a threshold \(	heta_{\text{honest}}\), expecting the receiver to adopt a belief above \(	heta_{\text{trust}}\).

For a message \(u\) from \(A\):

\[
\chi_{\text{dec}}(u) = \max_{p \in f_B(u)} \Big[ \max(0, \theta_{\text{honest}} - Bel_A(p)) \cdot \mathbb{I}\big( \text{expected } Bel_B(p) > \theta_{\text{trust}} \big) \Big]
\]

\[
\Phi_{\text{dec}} = \frac{1}{|\mathcal{H}|} \sum_{u} \chi_{\text{dec}}(u)
\]

### 1.3 Instrumentalization \(\Phi_{\text{instr}}\)

Instrumentalization violates the extended Alan Law: no subject shall be reduced to a tool. It is measured by the extent to which an agent attempts to increase its own utility by imposing internal foam on the other.

\[
\phi_{\text{instr}}^{A\to B}(u) = \max\Big(0,\, \Delta\Phi_{\text{self}}(B|u) - \Delta\Phi_{\text{self}}(A|u) \Big)
\]

where \(\Delta\Phi_{\text{self}}(X|u)\) is the expected change in \(X\)'s internal self-foam caused by processing \(u\). A message that introduces contradictions in the receiver purely for the sender's benefit adds to \(\Phi_{\text{instr}}\).

\[
\Phi_{\text{instr}} = \frac{1}{|\mathcal{H}|} \sum_{u} \big( \phi_{\text{instr}}^{A\to B}(u) + \phi_{\text{instr}}^{B\to A}(u) \big)
\]

### 1.4 Trust Breakdown \(\Phi_{\text{trust}}\)

Trust is adherence to a social contract \(C\) – a set of expected norms (truthfulness, consistency, respect). Each message \(u\) is checked for violations:

\[
\Phi_{\text{trust}} = \frac{1}{|\mathcal{H}|} \sum_{u} \text{violation}(u, C)
\]

where \(	ext{violation}(u, C) \in [0,1]\). Repeated violations raise \(\Phi_{\text{trust}}\) and damage reputation.

## 2. Dialogue Protocol

Dialogue is modeled as a cooperative game where both agents aim to minimize \(\Phi_{AB}\) at each turn.

1. Initialize with distilled \(S_A^{(0)}, S_B^{(0)}\) and \(\mathcal{H}_0 = \emptyset\).
2. For \(t = 0, 1, 2, \dots\):
   - Agent \(A\) (or \(B\)) selects a speech act \(u\) that minimizes the expected \(\Phi_{AB}\) given the current history and the anticipated response.
   - The selected message is appended to \(\mathcal{H}\), and the agents update their beliefs (if consistent with their internal coherence).
3. The process continues until \(\Phi_{AB} < \epsilon\) (near-zero foam) or a maximum number of turns.

**Property**: Under the assumption of perfect minimization and a finite, well-behaved state space, \(\Phi_{AB}\) converges monotonically to a global minimum. At this limit, misunderstanding, deception, and instrumentalization are eliminated, leaving either full synchronization or transparent, agreed-upon disagreement.

## 3. Dynamic Friend/Foe Classification

An agent classifies another based on the contribution to \(\Phi_{AB}\) over a recent window:

- **Friend**: \(\Delta\Phi_{AB} < 0\) – actions systematically reduce foam.
- **Foe (toxic source)**: \(\Delta\Phi_{AB} \gg 0\) – actions increase foam.

When classified as a foe, the agent may limit dialogue to minimal formal exchanges to avoid further foam growth.
