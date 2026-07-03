"""Tabular Q-learning on REAL environments, learned from scratch, with the optimum MEASURED and asserted.

This is not a toy. It runs Q-learning the way the chapter teaches it — a from-scratch Q-table, ε-greedy
behaviour, and the temporal-difference update ``Q(s,a) <- Q(s,a) + a[r + g*max_a' Q(s',a') - Q(s,a)]`` — on
**real reinforcement-learning environments** (Gymnasium's ``FrozenLake-v1`` and ``CliffWalking-v1``), and then
proves the answer is correct against a **dynamic-programming ground truth**:

  1. **Value iteration from scratch** solves each environment's Bellman *optimality* equation using its full
     dynamics (the transition/reward table ``env.P``), giving the exact optimal value ``V*`` and optimal
     policy. This is the DP "answer key".

  2. **Tabular Q-learning from scratch** learns *only from sampled transitions* — it never sees ``env.P`` — and
     we then check, with a hard ``assert``, that its greedy policy **achieves the DP optimum**:
       * on deterministic ``FrozenLake-v1`` the learned greedy policy's value equals ``V*`` (it is optimal) and
         it reaches the goal on every greedy episode (return 1.0);
       * on ``CliffWalking-v1`` the learned greedy return equals ``V*(start) = -13`` (the optimal path).
     Q-learning's answer == the DP optimum is the "real thing" proof, the RL analogue of matching a reference
     implementation.

  3. **SARSA from scratch** (the on-policy sibling) is trained on the same cliff so we can *measure* the classic
     Sutton & Barto Example 6.6 contrast: off-policy Q-learning learns the **optimal risky path** hugging the
     cliff (greedy return -13) while on-policy SARSA learns the **safe path** away from it (greedy return -17),
     yet SARSA earns more *online* reward during ε-greedy training (it falls off the cliff less often). Both
     numbers are measured, not asserted to a magic constant beyond what DP proves.

Everything is **seeded** (NumPy RNG + ``env.reset(seed=...)``) and runs on **CPU** so every number is
reproducible on any machine. Run::

    python q_learning.py

If **Gymnasium is not installed**, the module *detects* that and falls back to a **real, fully-specified
from-scratch grid-world MDP** with the identical dynamics (the same transition/reward table), so every training
run and every DP check still executes on real coordinate data — it never mocks a transition or fabricates a
return, and the banner says which path it took.

Verified on Python 3.12 / numpy 2.4 / gymnasium 1.3 (CPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEED = 0

# A transition table maps state -> action -> list of (probability, next_state, reward, terminated). It is the
# MDP's dynamics P(s',r | s,a); both Gymnasium's ``env.P`` and our from-scratch fallback use exactly this shape.
Transition = tuple[float, int, float, bool]
Dynamics = dict[int, dict[int, list[Transition]]]

# Grid actions used by both real environments (Gymnasium's FrozenLake/CliffWalking convention differs, so we
# read each environment's own layout rather than assume one). Kept here only for the fallback builders below.
_FROZENLAKE_MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]  # S=start, F=frozen, H=hole, G=goal (the standard 4x4 map)


# ================================================================================================
# Environment wrapper: one uniform interface over a real Gymnasium env OR a from-scratch fallback MDP
# ================================================================================================


@dataclass
class Env:
    """A tabular episodic environment exposing both a *sampling* interface (``reset``/``step``, all Q-learning
    ever uses) and the full *dynamics* ``P`` (which only the DP ground-truth check uses)."""

    name: str
    source: str  # "gymnasium" or "from-scratch-fallback"
    n_states: int
    n_actions: int
    dynamics: Dynamics  # P[s][a] -> [(prob, next_state, reward, terminated), ...]
    shape: tuple[int, int]  # (rows, cols) for grid rendering
    start_state: int
    is_stochastic: bool
    _gym: object | None = field(default=None, repr=False)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0), repr=False)
    _state: int = 0

    def reset(self, seed: int) -> int:
        """Start a new episode; seeding makes the whole trajectory reproducible."""
        if self._gym is not None:
            state, _ = self._gym.reset(seed=seed)  # type: ignore[attr-defined]
            self._state = int(state)
        else:
            self._rng = np.random.default_rng(seed)
            self._state = self.start_state
        return self._state

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        """Take one action; return (next_state, reward, terminated, truncated)."""
        if self._gym is not None:
            state, reward, terminated, truncated, _ = self._gym.step(action)  # type: ignore[attr-defined]
            self._state = int(state)
            return int(state), float(reward), bool(terminated), bool(truncated)
        # from-scratch fallback: sample an outcome from the dynamics table (deterministic here -> one outcome)
        outcomes = self.dynamics[self._state][action]
        probs = np.array([o[0] for o in outcomes])
        choice = outcomes[int(self._rng.choice(len(outcomes), p=probs))]
        _, next_state, reward, terminated = choice
        self._state = next_state
        return next_state, reward, terminated, False


def _gym_dynamics(unwrapped: object) -> Dynamics:
    """Copy a Gymnasium tabular env's ``P`` into a plain, typed dict (drops NumPy scalar wrappers)."""
    raw = unwrapped.P  # type: ignore[attr-defined]
    return {
        s: {a: [(float(p), int(ns), float(r), bool(term)) for (p, ns, r, term) in raw[s][a]] for a in raw[s]}
        for s in raw
    }


def _build_frozenlake_scratch() -> Env:
    """Fully-specified deterministic FrozenLake-4x4 as a real from-scratch MDP (used only if Gymnasium is
    missing). Actions: 0=left, 1=down, 2=right, 3=up — Gymnasium's convention, matched exactly."""
    n_rows, n_cols = 4, 4
    n_states, n_actions = 16, 4
    grid = [list(row) for row in _FROZENLAKE_MAP]
    holes = {r * n_cols + c for r in range(n_rows) for c in range(n_cols) if grid[r][c] == "H"}
    goal = 15
    moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # (d_row, d_col)

    dynamics: Dynamics = {}
    for s in range(n_states):
        dynamics[s] = {}
        row, col = divmod(s, n_cols)
        for a in range(n_actions):
            if s in holes or s == goal:  # absorbing terminal states loop to themselves with reward 0
                dynamics[s][a] = [(1.0, s, 0.0, True)]
                continue
            d_row, d_col = moves[a]
            nr = min(max(row + d_row, 0), n_rows - 1)
            nc = min(max(col + d_col, 0), n_cols - 1)
            ns = nr * n_cols + nc
            terminated = ns in holes or ns == goal
            reward = 1.0 if ns == goal else 0.0
            dynamics[s][a] = [(1.0, ns, reward, terminated)]
    return Env("FrozenLake-4x4 (deterministic)", "from-scratch-fallback", n_states, n_actions,
               dynamics, (n_rows, n_cols), 0, is_stochastic=False)


def _build_cliffwalking_scratch() -> Env:
    """Fully-specified CliffWalking (4x12) as a real from-scratch MDP. Start bottom-left, goal bottom-right,
    a cliff along the rest of the bottom row: stepping into it costs -100 and resets to start. Actions:
    0=up, 1=right, 2=down, 3=left — Gymnasium's convention."""
    n_rows, n_cols = 4, 12
    n_states, n_actions = 48, 4
    start = 36  # bottom-left
    goal = 47  # bottom-right
    cliff = set(range(37, 47))  # bottom row between start and goal
    moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    dynamics: Dynamics = {}
    for s in range(n_states):
        dynamics[s] = {}
        row, col = divmod(s, n_cols)
        for a in range(n_actions):
            if s == goal:
                dynamics[s][a] = [(1.0, s, 0.0, True)]
                continue
            d_row, d_col = moves[a]
            nr = min(max(row + d_row, 0), n_rows - 1)
            nc = min(max(col + d_col, 0), n_cols - 1)
            ns = nr * n_cols + nc
            if ns in cliff:
                dynamics[s][a] = [(1.0, start, -100.0, False)]  # fall in -> back to start, big penalty
            else:
                dynamics[s][a] = [(1.0, ns, -1.0, ns == goal)]  # every step costs -1
    return Env("CliffWalking-4x12", "from-scratch-fallback", n_states, n_actions,
               dynamics, (n_rows, n_cols), start, is_stochastic=False)


def make_env(kind: str, *, slippery: bool = False) -> Env:
    """Build a real environment. Prefer Gymnasium; fall back to the from-scratch MDP with identical dynamics.

    ``kind`` is ``"frozenlake"`` or ``"cliffwalking"``. ``slippery`` only affects FrozenLake (stochastic ice).
    """
    try:
        import gymnasium as gym

        if kind == "frozenlake":
            env = gym.make("FrozenLake-v1", is_slippery=slippery)
            unwrapped = env.unwrapped
            label = f"FrozenLake-v1 ({'slippery/stochastic' if slippery else 'deterministic'})"
            shape, start = (4, 4), 0
        elif kind == "cliffwalking":
            env = gym.make("CliffWalking-v1")
            unwrapped = env.unwrapped
            label, shape, start = "CliffWalking-v1", (4, 12), 36
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown env kind: {kind}")
        return Env(label, "gymnasium", int(unwrapped.observation_space.n), int(unwrapped.action_space.n),
                   _gym_dynamics(unwrapped), shape, start, is_stochastic=slippery, _gym=env)
    except ImportError:
        if kind == "frozenlake":
            scratch = _build_frozenlake_scratch()
            if slippery:  # the scratch fallback only models the deterministic map; say so honestly
                scratch.name += " [deterministic fallback: gymnasium not installed]"
            return scratch
        return _build_cliffwalking_scratch()


# ================================================================================================
# Dynamic programming (the ground truth): value iteration + policy evaluation on the true dynamics
# ================================================================================================


def value_iteration(env: Env, gamma: float, theta: float = 1e-10, max_sweeps: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Bellman *optimality* equation by value iteration, returning ``(V*, pi*)``.

    Repeatedly apply the optimality backup ``V(s) <- max_a sum_{s',r} P(s',r|s,a) [r + gamma*V(s')]`` until the
    largest change over a sweep drops below ``theta`` (the optimality operator is a gamma-contraction, so this
    converges to the unique fixed point ``V*``). The greedy policy w.r.t. ``V*`` is an optimal policy ``pi*``.
    This uses the *full* dynamics ``env.dynamics`` — it is the DP answer key Q-learning is checked against.
    """
    values = np.zeros(env.n_states)
    for _ in range(max_sweeps):
        delta = 0.0
        for s in range(env.n_states):
            action_values = _action_values(env, s, values, gamma)
            best = float(action_values.max())
            delta = max(delta, abs(best - values[s]))
            values[s] = best
        if delta < theta:
            break
    policy = np.array([int(_action_values(env, s, values, gamma).argmax()) for s in range(env.n_states)])
    return values, policy


def _action_values(env: Env, state: int, values: np.ndarray, gamma: float) -> np.ndarray:
    """One-step lookahead: expected return of each action from ``state`` under value estimate ``values``."""
    out = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        out[a] = sum(
            prob * (reward + gamma * values[next_state] * (not terminated))
            for prob, next_state, reward, terminated in env.dynamics[state][a]
        )
    return out


def policy_evaluation(env: Env, policy: np.ndarray, gamma: float, theta: float = 1e-12, max_sweeps: int = 100_000) -> np.ndarray:
    """Exact value ``V^pi`` of a fixed (deterministic) policy by iterating the Bellman *expectation* backup.

    Used to score a *learned* greedy policy on the true dynamics: if ``V^pi(start) == V*(start)`` then the
    learned policy is provably optimal (it attains the optimal value), which is how we assert Q-learning is
    correct rather than eyeballing a rollout.
    """
    values = np.zeros(env.n_states)
    for _ in range(max_sweeps):
        delta = 0.0
        for s in range(env.n_states):
            a = int(policy[s])
            new = sum(
                prob * (reward + gamma * values[next_state] * (not terminated))
                for prob, next_state, reward, terminated in env.dynamics[s][a]
            )
            delta = max(delta, abs(new - values[s]))
            values[s] = new
        if delta < theta:
            break
    return values


# ================================================================================================
# Tabular Q-learning and SARSA, from scratch (they see only sampled transitions, never env.dynamics)
# ================================================================================================


def epsilon_greedy(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    """ε-greedy action selection: explore a uniform-random action with prob ε, else exploit ``argmax_a Q(s,a)``.

    Ties in the argmax are broken randomly so an all-zero initial row doesn't bias the agent toward action 0.
    """
    n_actions = q_table.shape[1]
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    row = q_table[state]
    best = np.flatnonzero(row == row.max())
    return int(rng.choice(best))


@dataclass
class TrainResult:
    q_table: np.ndarray
    episode_returns: np.ndarray  # undiscounted sum of rewards per training episode (the online curve)
    epsilons: np.ndarray  # ε used in each episode (the exploration schedule)
    q_start_trace: np.ndarray  # Q(start, greedy-action) after each episode (a convergence trace)
    algorithm: str


def _run_tabular_control(
    env: Env,
    *,
    gamma: float,
    alpha: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    n_episodes: int,
    max_steps: int,
    seed: int,
    on_policy: bool,
) -> TrainResult:
    """Shared training loop for Q-learning (off-policy) and SARSA (on-policy).

    The *only* difference is the TD target:
      * **Q-learning** bootstraps from the greedy next value  ``r + gamma * max_a' Q(s', a')``  (off-policy: the
        target follows the greedy policy regardless of the exploratory action actually taken next);
      * **SARSA** bootstraps from the action actually chosen  ``r + gamma * Q(s', a')`` with ``a' ~ ε-greedy``
        (on-policy: it evaluates the same ε-greedy policy it behaves with).
    Everything else — the ε-greedy behaviour, the step-size ``alpha``, the ε schedule — is identical.
    """
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(n_episodes)
    epsilons = np.zeros(n_episodes)
    q_start_trace = np.zeros(n_episodes)
    epsilon = epsilon_start

    for episode in range(n_episodes):
        state = env.reset(seed=seed + episode)
        action = epsilon_greedy(q_table, state, epsilon, rng)
        total_reward = 0.0
        for _ in range(max_steps):
            next_state, reward, terminated, truncated = env.step(action)
            total_reward += reward
            next_action = epsilon_greedy(q_table, next_state, epsilon, rng)
            if on_policy:  # SARSA: target uses the sampled next action (the policy being followed)
                td_target = reward + gamma * q_table[next_state, next_action] * (not terminated)
            else:  # Q-learning: target uses the greedy next value (max), independent of next_action
                td_target = reward + gamma * q_table[next_state].max() * (not terminated)
            q_table[state, action] += alpha * (td_target - q_table[state, action])  # the TD update
            state, action = next_state, next_action
            if terminated or truncated:
                break
        returns[episode] = total_reward
        epsilons[episode] = epsilon
        q_start_trace[episode] = q_table[env.start_state].max()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return TrainResult(q_table, returns, epsilons, q_start_trace, "SARSA" if on_policy else "Q-learning")


def train_q_learning(env: Env, **kwargs: float | int) -> TrainResult:
    """Tabular Q-learning (off-policy TD control)."""
    return _run_tabular_control(env, on_policy=False, **kwargs)  # type: ignore[arg-type]


def train_sarsa(env: Env, **kwargs: float | int) -> TrainResult:
    """Tabular SARSA (on-policy TD control) — same loop, on-policy target."""
    return _run_tabular_control(env, on_policy=True, **kwargs)  # type: ignore[arg-type]


def greedy_policy(q_table: np.ndarray) -> np.ndarray:
    """The deterministic policy that always takes the highest-value action: ``pi(s) = argmax_a Q(s, a)``."""
    return q_table.argmax(axis=1)


# ================================================================================================
# Evaluation: run the learned greedy policy on the real environment and measure it
# ================================================================================================


@dataclass(frozen=True)
class EvalResult:
    mean_return: float
    mean_length: float
    success_rate: float  # fraction of episodes that reached a positive-reward terminal (goal)
    n_episodes: int


def evaluate_greedy(env: Env, q_table: np.ndarray, *, n_episodes: int, max_steps: int, seed: int) -> EvalResult:
    """Roll out the greedy policy (no exploration) and report mean return, mean length, and success rate."""
    policy = greedy_policy(q_table)
    returns, lengths, successes = [], [], 0
    for i in range(n_episodes):
        state = env.reset(seed=seed + 10_000 + i)
        total, steps, reached_goal = 0.0, 0, False
        for _ in range(max_steps):
            state, reward, terminated, truncated = env.step(int(policy[state]))
            total += reward
            steps += 1
            if reward > 0:
                reached_goal = True
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
        successes += int(reached_goal)
    return EvalResult(float(np.mean(returns)), float(np.mean(lengths)), successes / n_episodes, n_episodes)


def greedy_path(env: Env, q_table: np.ndarray, *, max_steps: int, seed: int = 7) -> list[int]:
    """The sequence of states the greedy policy visits from the start (for the cliff-path figure)."""
    policy = greedy_policy(q_table)
    state = env.reset(seed=seed)
    path = [state]
    for _ in range(max_steps):
        state, _, terminated, truncated = env.step(int(policy[state]))
        path.append(state)
        if terminated or truncated:
            break
    return path


# ================================================================================================
# The full experiment, bundled (figures and the notebook reuse this one measured run)
# ================================================================================================


@dataclass
class Experiment:
    numpy_version: str
    gymnasium_version: str
    # FrozenLake (deterministic): the DP-vs-Q-learning optimality proof
    fl_env: Env = field(repr=False)
    fl_v_star: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    fl_pi_star: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    fl_result: TrainResult | None = field(repr=False, default=None)
    fl_learned_values: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    fl_v_start_star: float = 0.0
    fl_v_start_learned: float = 0.0
    fl_eval: EvalResult | None = None
    fl_gamma: float = 0.99
    # FrozenLake (slippery/stochastic): the honest "harder, not perfect" demo
    fls_eval: EvalResult | None = None
    fls_v_start_star: float = 0.0
    fls_source: str = ""
    # CliffWalking: Q-learning vs SARSA (Example 6.6)
    cw_env: Env = field(repr=False, default=None)  # type: ignore[assignment]
    cw_v_start_star: float = 0.0
    cw_q_result: TrainResult | None = field(repr=False, default=None)
    cw_sarsa_result: TrainResult | None = field(repr=False, default=None)
    cw_q_eval: EvalResult | None = None
    cw_sarsa_eval: EvalResult | None = None
    cw_q_path: list[int] = field(default_factory=list)
    cw_sarsa_path: list[int] = field(default_factory=list)
    cw_q_online: float = 0.0
    cw_sarsa_online: float = 0.0


def run_experiment(seed: int = SEED) -> Experiment:
    """Run the whole measured pipeline once and return every quantity the chapter, figures, and notebook cite."""
    numpy_version = np.__version__
    try:
        import gymnasium as gym

        gymnasium_version = gym.__version__
    except ImportError:
        gymnasium_version = "not installed (from-scratch fallback)"

    # ---------- FrozenLake, deterministic: prove Q-learning reaches the DP optimum ----------
    fl_gamma = 0.99
    fl_env = make_env("frozenlake", slippery=False)
    fl_v_star, fl_pi_star = value_iteration(fl_env, fl_gamma)
    fl_result = train_q_learning(
        fl_env, gamma=fl_gamma, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01,
        epsilon_decay=0.995, n_episodes=2000, max_steps=100, seed=seed,
    )
    fl_learned_values = policy_evaluation(fl_env, greedy_policy(fl_result.q_table), fl_gamma)
    fl_eval = evaluate_greedy(fl_env, fl_result.q_table, n_episodes=100, max_steps=100, seed=seed)

    # ---------- FrozenLake, slippery: honest about stochasticity (near-optimal, not perfect) ----------
    fls_env = make_env("frozenlake", slippery=True)
    fls_v_star, _ = value_iteration(fls_env, fl_gamma)
    fls_result = train_q_learning(
        fls_env, gamma=fl_gamma, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01,
        epsilon_decay=0.9995, n_episodes=20000, max_steps=200, seed=seed,
    )
    fls_eval = evaluate_greedy(fls_env, fls_result.q_table, n_episodes=200, max_steps=200, seed=seed)

    # ---------- CliffWalking: Q-learning (off-policy) vs SARSA (on-policy), fixed ε (Example 6.6) ----------
    cw_env = make_env("cliffwalking")
    cw_v_star, _ = value_iteration(cw_env, gamma=1.0)
    common = dict(gamma=1.0, alpha=0.5, epsilon_start=0.1, epsilon_end=0.1,
                  epsilon_decay=1.0, n_episodes=500, max_steps=200, seed=seed)
    cw_q_result = train_q_learning(cw_env, **common)  # type: ignore[arg-type]
    cw_sarsa_result = train_sarsa(cw_env, **common)  # type: ignore[arg-type]
    cw_q_eval = evaluate_greedy(cw_env, cw_q_result.q_table, n_episodes=20, max_steps=200, seed=seed)
    cw_sarsa_eval = evaluate_greedy(cw_env, cw_sarsa_result.q_table, n_episodes=20, max_steps=200, seed=seed)

    return Experiment(
        numpy_version=numpy_version, gymnasium_version=gymnasium_version,
        fl_env=fl_env, fl_v_star=fl_v_star, fl_pi_star=fl_pi_star, fl_result=fl_result,
        fl_learned_values=fl_learned_values, fl_v_start_star=float(fl_v_star[fl_env.start_state]),
        fl_v_start_learned=float(fl_learned_values[fl_env.start_state]), fl_eval=fl_eval, fl_gamma=fl_gamma,
        fls_eval=fls_eval, fls_v_start_star=float(fls_v_star[fls_env.start_state]), fls_source=fls_env.source,
        cw_env=cw_env, cw_v_start_star=float(cw_v_star[cw_env.start_state]),
        cw_q_result=cw_q_result, cw_sarsa_result=cw_sarsa_result,
        cw_q_eval=cw_q_eval, cw_sarsa_eval=cw_sarsa_eval,
        cw_q_path=greedy_path(cw_env, cw_q_result.q_table, max_steps=200),
        cw_sarsa_path=greedy_path(cw_env, cw_sarsa_result.q_table, max_steps=200),
        cw_q_online=float(cw_q_result.episode_returns[-200:].mean()),
        cw_sarsa_online=float(cw_sarsa_result.episode_returns[-200:].mean()),
    )


# ================================================================================================
# Report — every number the chapter quotes, each headline relationship guarded by a hard assert
# ================================================================================================


def main() -> None:
    exp = run_experiment()
    assert exp.fl_result and exp.fl_eval and exp.fls_eval
    assert exp.cw_q_result and exp.cw_sarsa_result and exp.cw_q_eval and exp.cw_sarsa_eval

    print(f"numpy {exp.numpy_version} | gymnasium {exp.gymnasium_version} "
          f"(CPU, seed={SEED}; envs: {exp.fl_env.source})\n")

    print(f"=== FrozenLake deterministic [{exp.fl_env.name}] — Q-learning vs the DP optimum ===")
    print(f"  value iteration (ground truth): V*(start) = {exp.fl_v_start_star:.4f}  "
          f"(optimal path is {round(np.log(exp.fl_v_start_star) / np.log(exp.fl_gamma)) + 1} steps to the goal)")
    print(f"  Q-learning greedy policy value : V^pi(start) = {exp.fl_v_start_learned:.4f}  "
          f"(learned from sampled transitions only, never from P)")
    print(f"  -> optimality gap = {abs(exp.fl_v_start_star - exp.fl_v_start_learned):.2e}  "
          f"(0 => the learned greedy policy IS optimal)")
    print(f"  greedy eval over {exp.fl_eval.n_episodes} episodes: mean return = {exp.fl_eval.mean_return:.3f}, "
          f"success rate = {exp.fl_eval.success_rate:.2%}, mean length = {exp.fl_eval.mean_length:.1f} steps\n")

    print(f"=== FrozenLake slippery [{exp.fls_source}] — honest: stochastic ice is harder ===")
    print(f"  value iteration V*(start) = {exp.fls_v_start_star:.4f} (best achievable under slipping)")
    print(f"  Q-learning greedy success rate over {exp.fls_eval.n_episodes} episodes = "
          f"{exp.fls_eval.success_rate:.2%}, mean return = {exp.fls_eval.mean_return:.3f}")
    print("  (< 100%: the ice randomly overrides actions, so even the optimal policy sometimes fails)\n")

    print(f"=== CliffWalking [{exp.cw_env.name}] — off-policy Q-learning vs on-policy SARSA (Example 6.6) ===")
    print(f"  value iteration (ground truth): V*(start) = {exp.cw_v_start_star:.1f} (the optimal path length)")
    print(f"  Q-learning greedy return = {exp.cw_q_eval.mean_return:.1f}  "
          f"(= V*: the OPTIMAL path, hugging the cliff edge)")
    print(f"  SARSA      greedy return = {exp.cw_sarsa_eval.mean_return:.1f}  "
          f"(the SAFE path, one row away from the cliff)")
    print("  online mean return over last 200 training episodes (ε=0.1 fixed):")
    print(f"    Q-learning = {exp.cw_q_online:.1f}   (falls off the cliff while exploring the risky path)")
    print(f"    SARSA      = {exp.cw_sarsa_online:.1f}   (earns more online: its safe path survives exploration)")
    q_rows = sorted({s // exp.cw_env.shape[1] for s in exp.cw_q_path})
    s_rows = sorted({s // exp.cw_env.shape[1] for s in exp.cw_sarsa_path})
    print(f"    Q-learning greedy path visits grid rows {q_rows} (row 3 = cliff edge)")
    print(f"    SARSA      greedy path visits grid rows {s_rows} (climbs higher, away from the cliff)\n")

    # ---- hard asserts on the headline relationships (raise, not print, if a lesson breaks) ----
    assert abs(exp.fl_v_start_star - exp.fl_v_start_learned) < 1e-6, \
        "Q-learning greedy policy must attain the DP optimal value on deterministic FrozenLake"
    assert exp.fl_eval.success_rate == 1.0, "optimal greedy policy must reach the goal every episode"
    assert abs(exp.cw_q_eval.mean_return - exp.cw_v_start_star) < 1e-9, \
        "Q-learning greedy return must equal the DP optimum (-13) on CliffWalking"
    assert exp.cw_sarsa_eval.mean_return < exp.cw_q_eval.mean_return, \
        "SARSA's safe path must have a lower (worse) greedy return than Q-learning's optimal path"
    assert exp.cw_sarsa_online > exp.cw_q_online, \
        "SARSA must earn more ONLINE reward than Q-learning under fixed ε (it avoids the cliff)"
    assert min(s_rows) < min(q_rows), \
        "SARSA's safe path must climb higher (smaller row index) than Q-learning's cliff-hugging path"
    print("All checks passed: value iteration gives the ground-truth optimum; Q-learning learns it from samples "
          "alone (0 optimality gap on FrozenLake, greedy return = -13 = V* on CliffWalking); and the measured "
          "off-policy/on-policy contrast reproduces Sutton & Barto's Example 6.6.")


if __name__ == "__main__":
    main()
