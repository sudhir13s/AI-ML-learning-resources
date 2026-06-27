"""Manim scene: From RLHF to DPO.

A 3Blue1Brown-style conceptual animation for the chapter. It shows the RLHF
training loop — four models (policy, reward model, value/critic, frozen
reference) wired into an RL loop with a KL leash — then *transforms* it into
DPO: the reward model folds into the policy as the implicit reward
beta*log(pi/pi_ref), the RL loop and critic fall away, and a direct gradient on
preference pairs remains. Two models survive. This is the chapter's central
thesis, animated.

Render (writes a GIF to manim's media/ dir; move it into ../../images/):

    manim -ql --format gif manim_rlhf_to_dpo.py RLHFtoDPO

Uses Text only (no LaTeX dependency). Palette matches the chapter figures.
Verified on manim Community v0.20 / Python 3.12 (ml-py312).
"""

from __future__ import annotations

from manim import (
    BOLD, DOWN, LEFT, RIGHT, UP, WHITE,
    Arrow, Brace, CurvedArrow, DashedLine, FadeIn, FadeOut, RoundedRectangle,
    Scene, Text, VGroup, Write, ReplacementTransform, Indicate, config,
)

# Chapter palette (matches make_figures_15.py).
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
INK = "#10151C"

config.background_color = INK


def box(label: str, color: str, w: float = 2.5, h: float = 1.15, fs: int = 24) -> VGroup:
    """A rounded, tinted box with a centered white label — one node of the diagram."""
    rect = RoundedRectangle(
        corner_radius=0.18, width=w, height=h, color=color,
        fill_color=color, fill_opacity=0.22, stroke_width=3,
    )
    txt = Text(label, font_size=fs, color=WHITE, weight=BOLD).move_to(rect.get_center())
    return VGroup(rect, txt)


def edge_label(text: str, color: str = WHITE, fs: int = 18) -> Text:
    return Text(text, font_size=fs, color=color)


class RLHFtoDPO(Scene):
    def construct(self) -> None:
        title = Text("From RLHF to DPO", font_size=40, weight=BOLD, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.0)

        # ---- Phase 1: the RLHF loop (four models) ----------------------------------------
        prompt = box("Prompt", GREEN, w=2.0).shift(LEFT * 5.0)
        policy = box("Policy\npi_theta", BLUE).shift(LEFT * 0.6)
        rm = box("Reward\nModel", AMBER).shift(RIGHT * 4.2 + UP * 0.6)
        critic = box("Value /\nCritic", RED, w=2.2, h=1.0).shift(RIGHT * 4.2 + DOWN * 1.9)
        ref = box("Reference\npi_ref (frozen)", SLATE, w=2.9).shift(LEFT * 0.6 + DOWN * 2.3)

        a_prompt = Arrow(prompt.get_right(), policy.get_left(), buff=0.15, color=WHITE, stroke_width=3)
        a_samples = Arrow(policy.get_right(), rm.get_left(), buff=0.15, color=WHITE, stroke_width=3)
        samples_lbl = edge_label("responses", fs=16).next_to(a_samples, UP, buff=0.1)
        ppo_loop = CurvedArrow(rm.get_top(), policy.get_top(), angle=-1.1, color=RED, stroke_width=3)
        ppo_lbl = edge_label("PPO update (RL loop)", RED, fs=16).next_to(ppo_loop, UP, buff=0.05)
        kl = DashedLine(policy.get_bottom(), ref.get_top(), color=SLATE, stroke_width=3)
        kl_lbl = edge_label("KL leash", SLATE, fs=16).next_to(kl, RIGHT, buff=0.12)

        self.play(FadeIn(prompt), FadeIn(policy), run_time=0.7)
        self.play(Write(a_prompt), run_time=0.5)
        self.play(Write(a_samples), FadeIn(samples_lbl), FadeIn(rm), run_time=0.8)
        self.play(FadeIn(critic), Write(ppo_loop), FadeIn(ppo_lbl), run_time=0.8)
        self.play(Write(kl), FadeIn(kl_lbl), FadeIn(ref), run_time=0.8)

        cap = Text("RLHF: 4 models + an RL loop", font_size=26, weight=BOLD, color=AMBER).to_edge(DOWN, buff=0.5)
        self.play(Write(cap), run_time=0.8)
        self.wait(1.2)

        # ---- Phase 2: collapse to DPO -----------------------------------------------------
        # The reward model folds into the policy as the implicit reward; critic + RL loop go.
        implicit = Text("implicit reward  beta*log(pi / pi_ref)", font_size=20, color=AMBER, weight=BOLD)
        implicit.next_to(policy, UP, buff=0.25)

        self.play(FadeOut(critic), FadeOut(ppo_loop), FadeOut(ppo_lbl), run_time=0.8)
        self.play(
            FadeOut(a_samples), FadeOut(samples_lbl),
            ReplacementTransform(rm, implicit),
            run_time=1.1,
        )

        prefs = box("Preference pairs\n(chosen vs rejected)", PURPLE, w=3.4, h=1.3, fs=20)
        prefs.shift(RIGHT * 4.2)
        a_dpo = Arrow(prefs.get_left(), policy.get_right(), buff=0.15, color=PURPLE, stroke_width=4)
        dpo_lbl = edge_label("DPO loss: one direct gradient", PURPLE, fs=18).next_to(a_dpo, UP, buff=0.12)

        self.play(FadeIn(prefs), run_time=0.7)
        self.play(Write(a_dpo), FadeIn(dpo_lbl), run_time=0.8)

        # The two surviving models.
        survivors = VGroup(policy, ref)
        brace = Brace(survivors, LEFT, color=GREEN)
        brace_lbl = Text("2 models", font_size=22, weight=BOLD, color=GREEN).next_to(brace, LEFT, buff=0.15)
        self.play(Indicate(policy, color=GREEN), Indicate(ref, color=GREEN), run_time=0.9)
        self.play(FadeIn(brace), FadeIn(brace_lbl), run_time=0.7)

        cap2 = Text("DPO: no reward model, no RL loop  ·  2 models",
                    font_size=26, weight=BOLD, color=GREEN).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(cap, cap2), run_time=1.0)
        self.wait(0.8)

        final = Text("Same optimal policy — reached directly.",
                     font_size=24, color=WHITE).next_to(title, DOWN, buff=0.5)
        self.play(Write(final), run_time=1.0)
        self.wait(1.8)
