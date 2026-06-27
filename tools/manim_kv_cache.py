"""3Blue1Brown-style animation of the KV-cache mechanism for 05-KV-Cache.

Renders the headline GIF ``kv_cache_flow.gif``: the autoregressive decode loop, side by side.
Without a cache, every new token forces the model to re-project the keys and values of ALL
tokens so far (the whole stack flashes red — recomputed). With a cache, only the new token's
key/value is computed (green) and every earlier one is read straight from the cache (grey —
reused). A running tally makes the O(n^2) vs O(n) total concrete: 15 projections vs 5.

Text-only (no LaTeX/MathTex) so it renders without a TeX install. Render + convert to a small
GIF with the project's palette pipeline (manim MP4 -> ffmpeg palettegen/paletteuse):

    manim -qm --format mp4 manim_kv_cache.py KVCacheFlow
    ffmpeg -i media/videos/manim_kv_cache/720p30/KVCacheFlow.mp4 \
      -vf "fps=16,scale=900:-1:flags=lanczos,palettegen=max_colors=64:stats_mode=diff" -y palette.png
    ffmpeg -i ...mp4 -i palette.png -lavfi "fps=16,scale=900:-1:flags=lanczos,paletteuse=dither=none" \
      -y ../../images/kv_cache_flow.gif

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
    Indicate,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

# Muted palette matching the chapter's Mermaid classDefs.
INK = "#1C2530"
NAVY = "#2A5B80"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
GREY = "#9AA6B2"
BG = "#FBFCFD"

N = 5  # tokens decoded in the demo


def kv_box(label: str, fill: str) -> VGroup:
    """A small rounded box standing for one token's (key, value) pair."""
    rect = RoundedRectangle(
        width=1.15, height=0.54, corner_radius=0.10,
        stroke_color=fill, stroke_width=2.5, fill_color=fill, fill_opacity=0.18,
    )
    txt = Text(label, color=INK, weight=BOLD).scale(0.32).move_to(rect.get_center())
    return VGroup(rect, txt)


class KVCacheFlow(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG

        title = Text("KV cache: compute each key/value once, then reuse",
                     color=INK, weight=BOLD).scale(0.58).to_edge(UP, buff=0.45)
        self.play(Write(title))

        # Column headers.
        head_no = Text("Without a cache", color=RED, weight=BOLD).scale(0.5)
        head_yes = Text("With a KV cache", color=GREEN, weight=BOLD).scale(0.5)
        head_no.move_to(LEFT * 3.55 + UP * 1.8)
        head_yes.move_to(RIGHT * 3.55 + UP * 1.8)
        sub_no = Text("re-project K,V for every token, every step",
                      color=SLATE).scale(0.30).next_to(head_no, DOWN, buff=0.12)
        sub_yes = Text("project only the new token; reuse the rest",
                       color=SLATE).scale(0.30).next_to(head_yes, DOWN, buff=0.12)
        self.play(FadeIn(head_no), FadeIn(head_yes), FadeIn(sub_no), FadeIn(sub_yes))

        # Running tallies.
        tally_no = Text("projections: 0", color=RED, weight=BOLD).scale(0.42)
        tally_yes = Text("projections: 0", color=GREEN, weight=BOLD).scale(0.42)
        tally_no.move_to(LEFT * 3.55 + DOWN * 2.65)
        tally_yes.move_to(RIGHT * 3.55 + DOWN * 2.65)
        self.play(FadeIn(tally_no), FadeIn(tally_yes))

        left_x, right_x = -3.55, 3.55
        row_y0 = 0.95            # y of the first (top) cache row
        row_dy = 0.66            # rows grow DOWNWARD from row_y0
        total_no, total_yes = 0, 0
        cache_boxes: list[VGroup] = []  # persistent boxes on the RIGHT (the actual cache)

        for t in range(1, N + 1):
            step_lbl = Text(f"decode step {t}: token t{t} arrives",
                            color=NAVY, weight=BOLD).scale(0.42).move_to(UP * 2.45)
            self.play(FadeIn(step_lbl), run_time=0.4)

            # --- LEFT: recompute K,V for ALL t tokens (every box flashes red) ----------
            left_boxes = VGroup(*[
                kv_box(f"k{i},v{i}", RED).move_to(
                    [left_x, row_y0 - (i - 1) * row_dy, 0])
                for i in range(1, t + 1)
            ])
            self.play(FadeIn(left_boxes), run_time=0.45)
            self.play(*[Indicate(b, color=RED, scale_factor=1.12) for b in left_boxes],
                      run_time=0.5)
            total_no += t

            # --- RIGHT: compute ONLY the new token (green); earlier ones grey (reused) --
            for b in cache_boxes:  # everything already cached just gets reused
                b[0].set_stroke(GREY)
                b[0].set_fill(GREY, opacity=0.12)
                b[1].set_color(SLATE)
            new_box = kv_box(f"k{t},v{t}", GREEN).move_to(
                [right_x, row_y0 - (t - 1) * row_dy, 0])
            self.play(Create(new_box), run_time=0.45)
            self.play(Indicate(new_box, color=GREEN, scale_factor=1.18), run_time=0.45)
            cache_boxes.append(new_box)
            total_yes += 1

            # --- update tallies ---
            new_tally_no = Text(f"projections: {total_no}", color=RED, weight=BOLD).scale(0.42).move_to(tally_no)
            new_tally_yes = Text(f"projections: {total_yes}", color=GREEN, weight=BOLD).scale(0.42).move_to(tally_yes)
            self.play(
                tally_no.animate.become(new_tally_no),
                tally_yes.animate.become(new_tally_yes),
                FadeOut(left_boxes),
                FadeOut(step_lbl),
                run_time=0.5,
            )

        # Clear the side content so the punchline lands on a clean, centred card.
        self.play(
            *[FadeOut(b) for b in cache_boxes],
            FadeOut(head_no), FadeOut(head_yes), FadeOut(sub_no), FadeOut(sub_yes),
            FadeOut(tally_no), FadeOut(tally_yes),
            run_time=0.5,
        )

        # Punchline.
        punch = Text(f"{total_no} projections  vs  {total_yes}    —    O(n²)  →  O(n)",
                     color=INK, weight=BOLD).scale(0.7).move_to(DOWN * 0.1)
        box = Rectangle(width=punch.width + 0.7, height=punch.height + 0.45,
                        stroke_color=AMBER, stroke_width=2.5,
                        fill_color=AMBER, fill_opacity=0.10).move_to(punch)
        self.play(FadeIn(box), Write(punch))
        self.wait(1.4)
