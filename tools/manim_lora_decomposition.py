"""3Blue1Brown-style animation of the LoRA low-rank update for 12-LoRA-and-PEFT.

Renders the headline GIF ``lora_decomposition.gif``: the central LoRA idea, built up in stages.
Start from a frozen pretrained weight W0 (d x d). Full fine-tuning would learn a second, equally
huge update ΔW (d x d) -- a million trainable parameters for one layer. LoRA instead factors that
update through a thin rank-r bottleneck, ΔW = (alpha/r)·B·A, where B is d x r and A is r x d. The
two skinny factors hold only 2rd parameters -- 64x fewer at d=1024, r=8 -- and the frozen W0 is
never touched: h = W0·x + (alpha/r)·B·A·x.

Text-only (no LaTeX/MathTex) so it renders without a TeX install. Render + convert to a small GIF
with the project's palette pipeline (manim MP4 -> ffmpeg palettegen/paletteuse):

    manim -qm --format mp4 manim_lora_decomposition.py LoRADecomposition
    ffmpeg -i ...mp4 -vf "fps=14,scale=860:-1:flags=lanczos,palettegen=max_colors=64:stats_mode=diff" -y palette.png
    ffmpeg -i ...mp4 -i palette.png -lavfi "fps=14,scale=860:-1:flags=lanczos,paletteuse=dither=none" -y ../../images/lora_decomposition.gif

Verified on manim Community v0.20.x.
"""

from __future__ import annotations

from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    ReplacementTransform,
    Rectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

# Muted palette matching the chapter's Mermaid classDefs / make_figures_12.
INK = "#1C2530"
NAVY = "#2A5B80"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
PURPLE = "#5D4A8A"
BG = "#FBFCFD"

D_SIDE = 1.9   # on-screen side of a "d x d" matrix
R_THIN = 0.42  # on-screen thickness of the rank-r dimension


def labelled(rect: Rectangle, label: str, scale: float = 0.34) -> VGroup:
    txt = Text(label, color=INK, weight=BOLD).scale(scale).move_to(rect.get_center())
    return VGroup(rect, txt)


class LoRADecomposition(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG

        title = Text("LoRA: freeze W, learn a low-rank update",
                     color=INK, weight=BOLD).scale(0.62).to_edge(UP, buff=0.45)
        self.play(Write(title))

        # ---- The frozen pretrained weight, on the left -----------------------------------
        w_rect = Rectangle(width=D_SIDE, height=D_SIDE, stroke_color=SLATE, stroke_width=3,
                           fill_color=SLATE, fill_opacity=0.16).move_to(LEFT * 3.7 + DOWN * 0.2)
        w_grp = labelled(w_rect, "W₀\n(d×d)", 0.4)
        w_sub = Text("pretrained — frozen", color=SLATE).scale(0.32).next_to(w_rect, DOWN, buff=0.2)
        plus = Text("+", color=INK, weight=BOLD).scale(0.8).next_to(w_rect, RIGHT, buff=0.55)
        self.play(Create(w_rect), FadeIn(w_grp[1]), FadeIn(w_sub))
        self.play(FadeIn(plus))

        # ---- Full fine-tuning: a second full d x d update --------------------------------
        ft_rect = Rectangle(width=D_SIDE, height=D_SIDE, stroke_color=RED, stroke_width=3,
                            fill_color=RED, fill_opacity=0.16).next_to(plus, RIGHT, buff=0.55)
        ft_grp = labelled(ft_rect, "ΔW\n(d×d)", 0.4)
        ft_cap = Text("full fine-tuning:\nd² = 1,048,576 params", color=RED, weight=BOLD)\
            .scale(0.34).next_to(ft_rect, DOWN, buff=0.2)
        self.play(Create(ft_rect), FadeIn(ft_grp[1]))
        self.play(FadeIn(ft_cap))
        self.wait(0.8)

        # ---- LoRA: factor that update through a rank-r bottleneck ------------------------
        explain = Text("LoRA: factor ΔW = B · A through a thin rank-r bottleneck",
                       color=NAVY, weight=BOLD).scale(0.44).move_to(DOWN * 2.55)
        self.play(FadeIn(explain))

        center = ft_rect.get_center()
        b_rect = Rectangle(width=R_THIN, height=D_SIDE, stroke_color=GREEN, stroke_width=3,
                           fill_color=GREEN, fill_opacity=0.18).move_to(center + LEFT * 0.95)
        a_rect = Rectangle(width=D_SIDE, height=R_THIN, stroke_color=GREEN, stroke_width=3,
                           fill_color=GREEN, fill_opacity=0.18).move_to(center + RIGHT * 0.95)
        b_grp = labelled(b_rect, "B\nd×r", 0.30)
        a_grp = labelled(a_rect, "A  (r×d)", 0.30)
        dot = Text("·", color=INK, weight=BOLD).scale(0.8).move_to(center)

        lora_cap = Text("LoRA:\n2rd = 16,384 params", color=GREEN, weight=BOLD)\
            .scale(0.34).next_to(a_rect, DOWN, buff=0.55)

        # Morph the full d x d update into the two skinny factors.
        self.play(
            ReplacementTransform(ft_rect, VGroup(b_rect, a_rect)),
            FadeOut(ft_grp[1]),
            FadeOut(ft_cap),
            run_time=1.1,
        )
        self.play(FadeIn(b_grp[1]), FadeIn(a_grp[1]), FadeIn(dot), FadeIn(lora_cap))
        self.wait(0.6)

        # Mark the rank bottleneck.
        r_brace = Text("r = 8\n(the bottleneck)", color=PURPLE, weight=BOLD)\
            .scale(0.30).next_to(b_rect, UP, buff=0.18)
        self.play(FadeIn(r_brace))
        self.wait(0.6)

        # ---- Punchline -------------------------------------------------------------------
        self.play(FadeOut(explain))
        punch = Text("1,048,576  →  16,384 trainable params   (64× fewer)",
                     color=INK, weight=BOLD).scale(0.5).move_to(DOWN * 2.5)
        forward = Text("h = W₀·x  +  (α/r)·B·A·x        — W₀ never changes",
                       color=NAVY).scale(0.4).next_to(punch, UP, buff=0.25)
        self.play(Write(forward))
        self.play(FadeIn(punch))
        self.wait(1.5)
