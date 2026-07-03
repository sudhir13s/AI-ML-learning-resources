"""Generate the step-by-step teaching notebook (06-Q-Learning.ipynb).

The notebook mirrors ``q_learning.py`` one step at a time so a learner can open it, run every cell live, and
*see* Q-learning built and proven on real environments: the Q-table, ε-greedy behaviour, the TD update done by
hand, the training loop, the greedy-policy evaluation, the value-iteration ground truth it is checked against,
the learned policy/value heatmap, the ε schedule and Q(start) convergence, the honest stochastic case, and the
Sutton & Barto Example 6.6 Q-learning-vs-SARSA cliff contrast. Each numbered step has a short markdown lead-in
(the intuition) followed by a focused code cell with real output.

    python build_notebook_06.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../06-Q-Learning/code/06-Q-Learning.ipynb"

This generator lives in the domain-level ``08. Reinforcement_Learning/tools/`` folder; the notebook it writes
(and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "06-Q-Learning" / "code"
NB_PATH = _CHAPTER_CODE / "06-Q-Learning.ipynb"

_CELL_ID = 0


def _next_id() -> str:
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


def add_md(source: str) -> None:
    CELLS.append(md(source))


def add_code(source: str) -> None:
    CELLS.append(code(source))


# ============================ Title ============================================================
add_md(
    "# Q-learning — a runnable, measured, *proven* build\n"
    "\n"
    "An agent is dropped onto a frozen lake. It cannot see a map, it does not know the rules of the ice, and it "
    "gets **no reward at all** until — if — it stumbles onto the goal. From that single delayed signal, and "
    "nothing else, it must learn to walk straight to the goal. That is the reinforcement-learning problem, and "
    "**Q-learning** is the classic algorithm that solves it.\n"
    "\n"
    "This notebook builds tabular Q-learning **from scratch** on **real** Gymnasium environments and then *proves* "
    "it is correct against a dynamic-programming ground truth:\n"
    "\n"
    "- **Value iteration** solves the environment's Bellman *optimality* equation from its full dynamics — the "
    "exact optimal value `V*`. This is the answer key.\n"
    "- **Q-learning** learns from *sampled transitions only* (it never sees the dynamics), and we check with a "
    "hard `assert` that its greedy policy **attains `V*`** — on FrozenLake the optimality gap is `0`; on "
    "CliffWalking the greedy return equals the optimal `-13`.\n"
    "- **SARSA** (the on-policy sibling) is trained on the same cliff to *measure* the famous off-policy/on-policy "
    "contrast (Sutton & Barto Example 6.6).\n"
    "\n"
    "It imports the **exact same functions** as the companion page and its figures (from `q_learning.py`), so the "
    "numbers here are the numbers there. Everything is **seeded and CPU-pinned** for a reproducible trace.\n"
    "\n"
    "> Companion page: **Q-Learning**. Run top-to-bottom (Kernel → Restart & Run All). If Gymnasium is not "
    "installed, the module falls back to a real, fully-specified from-scratch grid-world with identical dynamics — "
    "every run and every check still executes on real data."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup: import the real module and print versions\n"
    "\n"
    "We import the pipeline from the chapter module so this notebook runs the *same code* the page and figures "
    "use, and print the library versions."
)
add_code(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.patches as mpatches\n"
    "\n"
    "import q_learning as ql\n"
    "\n"
    "try:\n"
    "    import gymnasium as gym\n"
    "    gym_ver = gym.__version__\n"
    "except ImportError:\n"
    "    gym_ver = 'not installed (from-scratch fallback)'\n"
    "print(f'numpy {np.__version__} | gymnasium {gym_ver}  (CPU, seed={ql.SEED})')"
)

# ---- Step 1: the environment ----
add_md(
    "## Step 1 — The environment: a real grid-world with a delayed, sparse reward\n"
    "\n"
    "**FrozenLake** is a 4×4 grid: **S**tart (top-left), **F**rozen tiles you can walk on, **H**oles that end the "
    "episode with reward 0, and the **G**oal (bottom-right) worth **+1**. The only reward in the whole task is "
    "that single +1 at the goal — everything else is 0. The agent must learn a route to the goal from that one "
    "delayed signal.\n"
    "\n"
    "We use the deterministic version first (actions do exactly what they say) so the learning is clean; we add "
    "stochastic ice later."
)
add_code(
    "env = ql.make_env('frozenlake', slippery=False)\n"
    "print(f'environment : {env.name}   [{env.source}]')\n"
    "print(f'states      : {env.n_states}   actions: {env.n_actions}  (0=left,1=down,2=right,3=up)')\n"
    "print(f'start state : {env.start_state}   grid shape: {env.shape}')\n"
    "print('map:')\n"
    "for r in range(4):\n"
    "    print('   ' + ' '.join('SFFFFHFHFFFHHFFG'[r*4 + c] for c in range(4)))"
)

# ---- Step 2: DP ground truth ----
add_md(
    "## Step 2 — The ground truth: value iteration (the answer key)\n"
    "\n"
    "Before we let Q-learning loose, we compute the *exact* optimal solution by **value iteration** — repeatedly "
    "applying the Bellman optimality backup\n"
    "\n"
    "$$V(s) \\leftarrow \\max_a \\sum_{s',r} P(s',r\\mid s,a)\\,[\\,r + \\gamma V(s')\\,]$$\n"
    "\n"
    "using the environment's full dynamics `env.dynamics` (`P`). This gives `V*` and the optimal policy `pi*`. "
    "Q-learning will have to match this **without ever seeing the dynamics** — it only gets sampled transitions."
)
add_code(
    "GAMMA = 0.99\n"
    "V_star, pi_star = ql.value_iteration(env, gamma=GAMMA)\n"
    "print(f'V*(start) = {V_star[env.start_state]:.4f}')\n"
    "steps = round(np.log(V_star[env.start_state]) / np.log(GAMMA)) + 1\n"
    "print(f'=> optimal path is {steps} steps  (reward 1 discounted by gamma^{steps-1} = {GAMMA**(steps-1):.4f})')\n"
    "print('optimal V* on the grid:')\n"
    "print(np.round(V_star.reshape(4, 4), 3))"
)

# ---- Step 3: the Q-table ----
add_md(
    "## Step 3 — The Q-table: what Q-learning actually stores\n"
    "\n"
    "Q-learning learns an **action-value** function $Q(s,a)$ = *how good is it to take action $a$ in state $s$, "
    "then act greedily forever after*. Tabular Q-learning stores this as a plain `[n_states × n_actions]` array, "
    "initialized to zeros. Once learned, the policy is trivial: in each state, take the action with the largest "
    "$Q(s,a)$."
)
add_code(
    "Q = np.zeros((env.n_states, env.n_actions))\n"
    "print(f'Q-table shape: {Q.shape}  (a value for every state-action pair)')\n"
    "print(f'Q[start] before learning: {Q[env.start_state]}  (all zeros -> no preference yet)')"
)

# ---- Step 4: epsilon-greedy ----
add_md(
    "## Step 4 — ε-greedy: balancing exploration and exploitation\n"
    "\n"
    "If the agent always took the current-best action it would never discover a *better* one — it must sometimes "
    "**explore**. **ε-greedy** does exactly this: with probability ε take a random action (explore), otherwise "
    "take $\\arg\\max_a Q(s,a)$ (exploit). Early on ε is high (explore a lot); we decay it toward 0 so the agent "
    "eventually exploits its learned values."
)
add_code(
    "rng = np.random.default_rng(0)\n"
    "Q_demo = np.array([[0.1, 0.9, 0.3, 0.2]])  # one state; action 1 is currently best\n"
    "picks = [ql.epsilon_greedy(Q_demo, 0, epsilon=0.3, rng=rng) for _ in range(2000)]\n"
    "vals, counts = np.unique(picks, return_counts=True)\n"
    "print('with eps=0.3, action counts over 2000 picks:', dict(zip(vals.tolist(), counts.tolist())))\n"
    "print('=> ~77.5% action 1 (greedy) + ~7.5% each of the 4 actions from the random branch')"
)

# ---- Step 5: the TD update by hand ----
add_md(
    "## Step 5 — The heart of it: one temporal-difference update, by hand\n"
    "\n"
    "This single line *is* Q-learning:\n"
    "\n"
    "$$Q(s,a) \\;\\leftarrow\\; Q(s,a) + \\alpha\\,[\\,\\underbrace{r + \\gamma \\max_{a'} Q(s',a')}_{\\text{TD "
    "target}} - Q(s,a)\\,]$$\n"
    "\n"
    "The bracket is the **TD error** $\\delta$: the gap between our current estimate $Q(s,a)$ and a better, "
    "one-step-bootstrapped estimate $r + \\gamma \\max_{a'} Q(s',a')$ (the reward we just got plus the discounted "
    "value of the *best* next action). We nudge $Q(s,a)$ a fraction $\\alpha$ of the way toward that target. The "
    "`max` over next actions is what makes Q-learning **off-policy** — the target follows the *greedy* policy no "
    "matter what exploratory action we actually take next. Let's do one update by hand."
)
add_code(
    "Q = np.zeros((env.n_states, env.n_actions))\n"
    "s, a, alpha = 14, 2, 0.1                       # state 14, action 2 (right) -> the goal (state 15)\n"
    "# pretend we just stepped and observed: reward +1, next state 15 (terminal)\n"
    "r, s_next, terminated = 1.0, 15, True\n"
    "td_target = r + GAMMA * Q[s_next].max() * (not terminated)  # terminal -> no bootstrap: target = r = 1.0\n"
    "td_error = td_target - Q[s, a]\n"
    "Q[s, a] += alpha * td_error\n"
    "print(f'TD target = {td_target:.3f},  TD error = {td_error:.3f}')\n"
    "print(f'Q[14, right] after one update: {Q[s, a]:.3f}  (moved 10% of the way from 0 toward the target 1.0)')"
)

# ---- Step 6: the training loop ----
add_md(
    "## Step 6 — Train: run the update over thousands of episodes\n"
    "\n"
    "Now we run that update inside the full loop: reset → repeatedly (ε-greedy action → step → TD update) until "
    "the episode ends, and decay ε after each episode. `train_q_learning` returns the learned Q-table plus the "
    "per-episode return, the ε schedule, and a trace of `Q(start)` (all from the *same* function the page uses)."
)
add_code(
    "res = ql.train_q_learning(env, gamma=GAMMA, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01,\n"
    "                          epsilon_decay=0.995, n_episodes=2000, max_steps=100, seed=0)\n"
    "print(f'trained {len(res.episode_returns)} episodes')\n"
    "print(f'mean return over the last 100 training episodes: {res.episode_returns[-100:].mean():.3f}')\n"
    "print(f'Q[start] after learning: {res.q_table[env.start_state].round(3)}')"
)

# ---- Step 7: the proof ----
add_md(
    "## Step 7 — The proof: does the learned greedy policy match the DP optimum?\n"
    "\n"
    "This is the moment of truth. We take the learned greedy policy and (a) roll it out to measure its success "
    "rate, and (b) compute its *exact* value on the true dynamics with `policy_evaluation`, then compare that to "
    "`V*`. If `V^pi(start) == V*(start)`, the policy Q-learning learned **from samples alone** is provably "
    "optimal."
)
add_code(
    "ev = ql.evaluate_greedy(env, res.q_table, n_episodes=100, max_steps=100, seed=0)\n"
    "V_pi = ql.policy_evaluation(env, ql.greedy_policy(res.q_table), gamma=GAMMA)\n"
    "gap = abs(V_pi[env.start_state] - V_star[env.start_state])\n"
    "print(f'greedy eval  : success {ev.success_rate:.0%}, mean return {ev.mean_return:.3f}, "
    "{ev.mean_length:.0f} steps')\n"
    "print(f'V^pi(start)  = {V_pi[env.start_state]:.4f}   vs   V*(start) = {V_star[env.start_state]:.4f}')\n"
    "print(f'optimality gap = {gap:.2e}')\n"
    "assert gap < 1e-6 and ev.success_rate == 1.0\n"
    "print('OK: Q-learning learned the OPTIMAL policy from sampled experience alone.')"
)

# ---- Step 8: visualize policy + value ----
add_md(
    "## Step 8 — See it: the learned policy and value, next to the ground truth\n"
    "\n"
    "The clearest way to *see* what Q-learning found: draw the greedy action in every state (arrow) over the "
    "state-value heatmap $V(s)=\\max_a Q(s,a)$, beside the value-iteration optimum. On the route from start to "
    "goal they match. (A few off-path states in the left column keep a suboptimal arrow — Q-learning only "
    "guarantees optimality for state-actions it visits often enough, which is exactly the convergence condition; "
    "those states are never entered on the optimal trajectory, so `V(start)` and the route are still optimal.)"
)
add_code(
    "arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}\n"
    "labels = 'SFFFFHFHFFFHHFFG'\n"
    "pol = ql.greedy_policy(res.q_table)\n"
    "print('learned greedy policy:            optimal policy (value iteration):')\n"
    "for r in range(4):\n"
    "    left  = ' '.join((labels[r*4+c] if labels[r*4+c] in 'HG' else arrows[int(pol[r*4+c])]) for c in range(4))\n"
    "    right = ' '.join((labels[r*4+c] if labels[r*4+c] in 'HG' else arrows[int(pi_star[r*4+c])]) for c in range(4))\n"
    "    print(f'   {left}                 {right}')\n"
    "\n"
    "V_learned = np.array([res.q_table[s].max() for s in range(16)]).reshape(4, 4)\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))\n"
    "for ax, grid, title in [(a1, V_learned, 'learned V = max_a Q'), (a2, V_star.reshape(4, 4), 'V* (DP)')]:\n"
    "    ax.imshow(grid, cmap='YlGnBu')\n"
    "    ax.set_title(title)\n"
    "    ax.set_xticks([])\n"
    "    ax.set_yticks([])\n"
    "    for r in range(4):\n"
    "        for c in range(4):\n"
    "            ax.text(c, r, f'{grid[r, c]:.2f}', ha='center', va='center', color='white', fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 9: learning curve + epsilon ----
add_md(
    "## Step 9 — The two dials: the learning curve and the ε schedule\n"
    "\n"
    "Two views of *how* it learned: the reward-per-episode curve climbing to 1.0 (the agent reaches the goal ever "
    "more reliably), and the ε schedule that drives it (explore early, exploit late). Notice the value estimate "
    "`Q(start)` bootstrapping up to `V*` as ε falls."
)
add_code(
    "def rolling(x, w=50):\n"
    "    return np.convolve(x, np.ones(w)/w, mode='valid')\n"
    "\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))\n"
    "sm = rolling(res.episode_returns)\n"
    "a1.plot(np.arange(len(sm)) + 50, sm, color='#2E7A5A', lw=2, label='rolling mean return')\n"
    "a1.axhline(1.0, ls='--', color='#3A6B96', label='optimal (=1.0)')\n"
    "a1.set_xlabel('episode')\n"
    "a1.set_ylabel('return')\n"
    "a1.set_title('learning curve')\n"
    "a1.legend()\n"
    "a2.plot(res.epsilons, color='#5D4A8A', lw=2, label='ε (exploration)')\n"
    "a2.plot(res.q_start_trace, color='#2E7A5A', lw=2, label='Q(start) -> V*')\n"
    "a2.axhline(V_star[env.start_state], ls='--', color='#3A6B96')\n"
    "a2.set_xlabel('episode')\n"
    "a2.set_title('ε decay and Q(start) convergence')\n"
    "a2.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 10: stochastic case ----
add_md(
    "## Step 10 — Honest interlude: stochastic ice is harder\n"
    "\n"
    "Real problems are noisy. With **slippery** ice, an action only *usually* does what you asked — the ice "
    "randomly pushes you sideways. Q-learning still converges to the optimal *policy*, but the optimal policy "
    "itself cannot guarantee the goal (you can be slipped into a hole), so the success rate is well below 100%. "
    "This is not a bug; it is the environment. We train longer here because noise slows learning."
)
add_code(
    "slippery = ql.make_env('frozenlake', slippery=True)\n"
    "Vs, _ = ql.value_iteration(slippery, gamma=GAMMA)\n"
    "res_s = ql.train_q_learning(slippery, gamma=GAMMA, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01,\n"
    "                            epsilon_decay=0.9995, n_episodes=20000, max_steps=200, seed=0)\n"
    "ev_s = ql.evaluate_greedy(slippery, res_s.q_table, n_episodes=200, max_steps=200, seed=0)\n"
    "print(f'slippery: V*(start) = {Vs[slippery.start_state]:.3f}  (best achievable under slipping)')\n"
    "print(f'greedy success rate = {ev_s.success_rate:.1%}, mean return = {ev_s.mean_return:.3f}')\n"
    "print('=> < 100%: even the optimal policy sometimes gets slipped into a hole. That is the noise, not the agent.')"
)

# ---- Step 11: SARSA, the on-policy sibling ----
add_md(
    "## Step 11 — SARSA: the same loop, one different word in the target\n"
    "\n"
    "**SARSA** is Q-learning's on-policy sibling. The *only* change is the TD target:\n"
    "\n"
    "$$\\text{Q-learning (off-policy): } r + \\gamma\\,\\max_{a'} Q(s',a') \\qquad "
    "\\text{SARSA (on-policy): } r + \\gamma\\,Q(s',a')\\ \\text{with } a'\\sim\\varepsilon\\text{-greedy}$$\n"
    "\n"
    "Q-learning bootstraps from the *greedy* next action (the policy it's learning); SARSA bootstraps from the "
    "action it *actually takes* next (the ε-greedy policy it's following, exploration included). That one word — "
    "`max` vs the sampled `a'` — produces strikingly different behaviour near danger, which we measure next."
)
add_code(
    "cliff = ql.make_env('cliffwalking')\n"
    "Vc, _ = ql.value_iteration(cliff, gamma=1.0)\n"
    "common = dict(gamma=1.0, alpha=0.5, epsilon_start=0.1, epsilon_end=0.1, epsilon_decay=1.0,\n"
    "              n_episodes=500, max_steps=200, seed=0)\n"
    "q_res = ql.train_q_learning(cliff, **common)\n"
    "s_res = ql.train_sarsa(cliff, **common)\n"
    "print(f'CliffWalking optimal return V*(start) = {Vc[cliff.start_state]:.0f}')\n"
    "print(f'Q-learning greedy return = {ql.evaluate_greedy(cliff, q_res.q_table, n_episodes=20, max_steps=200, seed=0).mean_return:.0f}')\n"
    "print(f'SARSA      greedy return = {ql.evaluate_greedy(cliff, s_res.q_table, n_episodes=20, max_steps=200, seed=0).mean_return:.0f}')"
)

# ---- Step 12: the cliff paths ----
add_md(
    "## Step 12 — Example 6.6: off-policy takes the optimal risk, on-policy plays it safe\n"
    "\n"
    "CliffWalking: start bottom-left, goal bottom-right, a **cliff** along the bottom row (fall in → −100, back to "
    "start). The optimal path walks **right along the edge**, one row above the cliff (13 steps, return −13). "
    "**Q-learning** learns exactly that — it evaluates the greedy path, which is optimal. **SARSA** learns a path "
    "that **detours up and away** from the cliff: because it is on-policy, it accounts for the ε-greedy chance of "
    "randomly stepping *into* the cliff, so it (correctly) values the risky edge lower. Let's draw both greedy "
    "paths."
)
add_code(
    "def path_rows(env, q):\n"
    "    p = ql.greedy_path(env, q, max_steps=200)\n"
    "    return [divmod(s, env.shape[1]) for s in p]\n"
    "\n"
    "qp, sp = path_rows(cliff, q_res.q_table), path_rows(cliff, s_res.q_table)\n"
    "fig, ax = plt.subplots(figsize=(8, 3))\n"
    "for s in range(37, 47):\n"
    "    r, c = divmod(s, 12)\n"
    "    ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#8B3B4A', alpha=0.5))\n"
    "ax.plot([c for _, c in qp], [r - 0.1 for r, _ in qp], '-o', color='#2E7A5A', label='Q-learning (optimal, −13)')\n"
    "ax.plot([c for _, c in sp], [r + 0.1 for r, _ in sp], '-o', color='#7A6528', label='SARSA (safe, −17)')\n"
    "ax.text(5, 3, 'CLIFF (−100)', ha='center', va='center', color='white', fontweight='bold')\n"
    "ax.text(0, 3, 'S', ha='center', fontweight='bold')\n"
    "ax.text(11, 3, 'G', ha='center', fontweight='bold')\n"
    "ax.set_xlim(-0.6, 11.6)\n"
    "ax.set_ylim(3.6, -0.6)\n"
    "ax.set_xticks([])\n"
    "ax.set_yticks([])\n"
    "ax.legend(loc='upper center')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 13: online return contrast ----
add_md(
    "## Step 13 — The twist: SARSA earns *more* while training\n"
    "\n"
    "Here is the beautiful subtlety. Q-learning's *greedy* policy is better (−13 vs −17), but while it is "
    "**training with exploration on** (ε = 0.1), it keeps randomly stepping off the cliff it walks beside — so its "
    "**online** return is *worse*. SARSA's safe detour survives the same random exploration, so it accumulates "
    "more reward during learning. Off-policy learns the better *target* policy; on-policy performs better *while "
    "behaving*. This is the on-policy/off-policy trade-off, measured."
)
add_code(
    "def rolling(x, w=20):\n"
    "    return np.convolve(x, np.ones(w)/w, mode='valid')\n"
    "\n"
    "plt.figure(figsize=(8, 4))\n"
    "plt.plot(np.arange(len(rolling(s_res.episode_returns)))+20, rolling(s_res.episode_returns),\n"
    "         color='#7A6528', lw=2, label=f'SARSA online ≈ {s_res.episode_returns[-200:].mean():.0f}')\n"
    "plt.plot(np.arange(len(rolling(q_res.episode_returns)))+20, rolling(q_res.episode_returns),\n"
    "         color='#2E7A5A', lw=2, label=f'Q-learning online ≈ {q_res.episode_returns[-200:].mean():.0f}')\n"
    "plt.axhline(Vc[cliff.start_state], ls='--', color='#3A6B96', label='optimal = −13')\n"
    "plt.ylim(-100, 0)\n"
    "plt.xlabel('episode')\n"
    "plt.ylabel('online return (ε=0.1)')\n"
    "plt.title('SARSA earns more online; Q-learning learns the better greedy policy')\n"
    "plt.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "assert s_res.episode_returns[-200:].mean() > q_res.episode_returns[-200:].mean()\n"
    "print('OK: measured — SARSA online > Q-learning online, yet Q-learning greedy return (−13) is optimal.')"
)

# ---- close ----
add_md(
    "## Recap\n"
    "\n"
    "You built tabular Q-learning from scratch and **proved** it: value iteration gives the exact optimum, and "
    "Q-learning — learning from sampled transitions alone — recovers it (0 optimality gap on FrozenLake, greedy "
    "return −13 = V* on CliffWalking). The engine is one line, the TD update "
    "$Q(s,a) \\leftarrow Q(s,a) + \\alpha[r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]$; the `max` makes it "
    "**off-policy**; ε-greedy with decay supplies the exploration the convergence guarantee needs. The cliff "
    "showed the on/off-policy trade-off in the flesh: off-policy Q-learning learns the optimal risky path, "
    "on-policy SARSA the safe one, and SARSA earns more *while exploring*.\n"
    "\n"
    "See the companion page for the full derivation (the Bellman optimality equation, Q-learning as its "
    "sample-based stochastic approximation, the Robbins–Monro convergence conditions), the pitfalls "
    "(maximization bias → Double Q-learning), and where it goes next: when the state space is too big for a "
    "table, `Q` becomes a neural network — **Deep Q-Networks**."
)


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
