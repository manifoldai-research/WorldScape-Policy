#!/usr/bin/env python3
"""Render the WorldScape Policy 2.0 method-introduction video."""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
VIDEO_DIR = ASSETS / "videos"
FIGURE_DIR = ASSETS / "figures"
BUILD_DIR = VIDEO_DIR / ".method-video-build"

WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 30, 87
FONT_REGULAR = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_LATIN_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

INK = (242, 244, 255)
MUTED = (161, 168, 192)
PURPLE = (145, 88, 231)
CYAN = (45, 224, 225)
ORANGE = (255, 145, 86)
GREEN = (111, 225, 160)
BG = (8, 10, 19)


def font(size: int, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_LATIN_BOLD if latin else (FONT_BOLD if bold else FONT_REGULAR)
    return ImageFont.truetype(path, size)


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def enter(t: float, start: float, duration: float = 0.7) -> float:
    return ease((t - start) / duration)


def leave(t: float, end: float, duration: float = 0.6) -> float:
    return ease((end - t) / duration)


def scene_alpha(t: float, start: float, end: float) -> float:
    return min(enter(t, start), leave(t, end))


def draw_background(frame: np.ndarray, t: float) -> None:
    frame[:] = BG[::-1]
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    glow1 = np.exp(-((xx - 350 - 40 * math.sin(t / 5)) ** 2 + (yy - 220) ** 2) / 350000)
    glow2 = np.exp(-((xx - 1600) ** 2 + (yy - 820 - 50 * math.cos(t / 6)) ** 2) / 420000)
    frame[:, :, 0] = np.clip(frame[:, :, 0] + glow1 * 36 + glow2 * 28, 0, 255)
    frame[:, :, 1] = np.clip(frame[:, :, 1] + glow1 * 13 + glow2 * 28, 0, 255)
    frame[:, :, 2] = np.clip(frame[:, :, 2] + glow1 * 25 + glow2 * 10, 0, 255)

    offset = int((t * 10) % 64)
    for x in range(-offset, WIDTH, 64):
        cv2.line(frame, (x, 0), (x, HEIGHT), (29, 27, 48), 1, cv2.LINE_AA)
    for y in range(-offset, HEIGHT, 64):
        cv2.line(frame, (0, y), (WIDTH, y), (29, 27, 48), 1, cv2.LINE_AA)
    cv2.line(frame, (90, 78), (1830, 78), (63, 56, 88), 1, cv2.LINE_AA)
    cv2.putText(frame, "MANIFOLD AI  /  WORLDSCAPE TEAM", (92, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 154, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(t):02d}  /  {DURATION:02d}", (1732, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 154, 180), 1, cv2.LINE_AA)


def pil_layer(frame: np.ndarray) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    return img, ImageDraw.Draw(img, "RGBA")


def commit(frame: np.ndarray, img: Image.Image) -> None:
    frame[:] = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int,
         color=INK, bold: bool = False, anchor: str = "la", alpha: int = 255,
         latin: bool = False, spacing: int = 8) -> None:
    rgba = (*color, max(0, min(255, alpha)))
    draw.multiline_text(xy, value, fill=rgba, font=font(size, bold, latin),
                        anchor=anchor, spacing=spacing)


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str,
         accent=PURPLE, alpha: int = 255) -> None:
    draw.rounded_rectangle(box, radius=22, fill=(18, 20, 35, int(alpha * 0.94)),
                           outline=(*accent, int(alpha * 0.72)), width=2)
    text(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), label, 24,
         color=INK, bold=True, anchor="mm", alpha=alpha)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], alpha: int = 255,
          accent=PURPLE) -> None:
    draw.rounded_rectangle(box, radius=30, fill=(14, 16, 29, int(alpha * 0.94)),
                           outline=(*accent, int(alpha * 0.32)), width=2)


FIG_FRAMEWORK = Image.open(FIGURE_DIR / "framework-hd.png").convert("RGB")
FIG_MEMORY = Image.open(FIGURE_DIR / "lstm-hd.png").convert("RGB")
FIG_PIPELINE = Image.open(FIGURE_DIR / "pipeline-hd.png").convert("RGB")


def paste_contain(img: Image.Image, source: Image.Image, box: tuple[int, int, int, int],
                  alpha: int, padding: tuple[int, int] = (36, 24)) -> None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = padding
    card = Image.new("RGBA", (w, h), (250, 250, 250, alpha))
    scale = min((w - 2 * px) / source.width, (h - 2 * py) / source.height)
    size = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
    src = source.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    src.putalpha(alpha)
    card.paste(src, ((w - size[0]) // 2, (h - size[1]) // 2), src)
    mask = Image.new("L", (w, h), alpha)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, w, h), radius=24, fill=alpha)
    img.paste(card, (x1, y1), mask)


def render_title(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 0, 6))
    img, draw = pil_layer(frame)
    shift = int(32 * (1 - enter(t, 0.4)))
    text(draw, (WIDTH // 2, 355 + shift), "WorldScape", 116, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 480 + shift), "Policy 2.0", 116, color=PURPLE,
         bold=True, anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 615), "Empowering Steerable World Action Modeling", 37,
         color=INK, anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 666), "with Reasoning-Augmented Memory", 37,
         color=CYAN, anchor="mm", alpha=a, latin=True)
    pill(draw, (690, 760, 1230, 820), "WORLD  ·  ACTION  ·  MEMORY", CYAN, a)
    commit(frame, img)


def render_challenge(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 6, 13))
    img, draw = pil_layer(frame)
    text(draw, (110, 150), "核心挑战", 32, color=CYAN, bold=True, alpha=a)
    text(draw, (110, 210), "WAMs need more than short context", 58, bold=True,
         alpha=a, latin=True)
    cards = [
        ("01", "Limited Temporal Context", "Similar scenes can require\ndifferent actions.", PURPLE),
        ("02", "Coarse Language Grounding", "Episode labels miss\natomic action intent.", ORANGE),
        ("03", "Text-Only Interaction", "Goals and demonstrations\nremain unused.", CYAN),
    ]
    for i, (num, title, body, accent) in enumerate(cards):
        local = enter(t, 7.0 + i * 0.4)
        aa = int(a * local)
        x = 105 + i * 575
        y = 375 + int(28 * (1 - local))
        panel(draw, (x, y, x + 515, y + 390), aa, accent)
        text(draw, (x + 34, y + 46), num, 27, color=accent, bold=True, alpha=aa, latin=True)
        text(draw, (x + 32, y + 124), title, 28, bold=True, alpha=aa, latin=True)
        draw.line((x + 32, y + 190, x + 465, y + 190), fill=(*accent, aa // 2), width=2)
        text(draw, (x + 32, y + 236), body, 23, color=MUTED, alpha=aa,
             latin=True, spacing=11)
    commit(frame, img)


def render_contributions(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 13, 22))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 145), "核心贡献  /  CONTRIBUTIONS", 31, color=CYAN,
         bold=True, anchor="mm", alpha=a)
    text(draw, (WIDTH // 2, 220), "A steerable, memory-grounded World Action Model", 53,
         bold=True, anchor="mm", alpha=a, latin=True)
    cards = [
        ("01", "Long Short-Term\nMemory", "Local dynamics +\nglobal task progress", CYAN),
        ("02", "Implicit Subgoal\nReasoning", "Progress-aware\nlatent planning", ORANGE),
        ("03", "Native Multimodal\nControl", "Text + goal image +\nvideo demonstration", PURPLE),
        ("04", "ManipEvent-5M\nDataset", "Event-grounded\nmultimodal pretraining", GREEN),
    ]
    for i, (num, title, body, accent) in enumerate(cards):
        local = enter(t, 14 + i * 0.28)
        aa = int(a * local)
        x = 105 + i * 435
        y = 365 + int(24 * (1 - local))
        panel(draw, (x, y, x + 390, y + 420), aa, accent)
        text(draw, (x + 34, y + 42), num, 25, color=accent, bold=True,
             alpha=aa, latin=True)
        text(draw, (x + 195, y + 165), title, 30, bold=True, anchor="mm",
             alpha=aa, latin=True, spacing=10)
        draw.line((x + 45, y + 255, x + 345, y + 255),
                  fill=(*accent, aa // 2), width=2)
        text(draw, (x + 195, y + 325), body, 22, color=MUTED, anchor="mm",
             alpha=aa, latin=True, spacing=9)
    text(draw, (WIDTH // 2, 875),
         "REASONING-AUGMENTED MEMORY  ·  EVENT-GROUNDED MODELING  ·  IN-CONTEXT ADAPTATION",
         24, color=MUTED, bold=True, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_insight(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 22, 30))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 165), "关键洞察  /  KEY INSIGHT", 30, color=CYAN,
         bold=True, anchor="mm", alpha=a)
    text(draw, (WIDTH // 2, 245), "Memory at two complementary timescales", 59,
         bold=True, anchor="mm", alpha=a, latin=True)
    p = enter(t, 15)
    center = (960, 520)
    draw.ellipse((875, 435, 1045, 605), fill=(28, 25, 50, a),
                 outline=(*PURPLE, a), width=4)
    text(draw, center, "WAM", 38, bold=True, anchor="mm", alpha=a, latin=True)
    left, right = (475, 520), (1445, 520)
    draw.line((center[0] - 85, center[1], left[0] + 210, left[1]),
              fill=(*CYAN, int(a * p)), width=5)
    draw.line((center[0] + 85, center[1], right[0] - 210, right[1]),
              fill=(*ORANGE, int(a * p)), width=5)
    panel(draw, (170, 390, 685, 735), a, CYAN)
    panel(draw, (1235, 390, 1750, 735), a, ORANGE)
    text(draw, left, "Short-Term\nVisual Memory", 43, color=CYAN, bold=True,
         anchor="mm", alpha=a, spacing=14, latin=True)
    text(draw, (475, 650), "Recent frames preserve\nlocal interaction dynamics", 27,
         color=MUTED, anchor="mm", alpha=a, spacing=10, latin=True)
    text(draw, right, "Long-Term\nEvent Memory", 43, color=ORANGE, bold=True,
         anchor="mm", alpha=a, spacing=14, latin=True)
    text(draw, (1445, 650), "Task progress enables\nimplicit subgoal planning", 27,
         color=MUTED, anchor="mm", alpha=a, spacing=10, latin=True)
    text(draw, (WIDTH // 2, 865), "LOCAL DYNAMICS  +  GLOBAL PROGRESS", 30,
         color=INK, bold=True, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_framework(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 22, 34))
    img, draw = pil_layer(frame)
    text(draw, (110, 145), "统一的可控世界动作模型", 32, color=CYAN, bold=True, alpha=a)
    text(draw, (110, 205), "One backbone. Multiple ways to steer.", 58,
         bold=True, alpha=a, latin=True)
    panel(draw, (110, 305, 1810, 945), a, PURPLE)
    paste_contain(img, FIG_FRAMEWORK, (145, 335, 1775, 790), a, (55, 34))
    labels = [
        (150, "MULTIMODAL PROMPTS", CYAN),
        (700, "REASONING MEMORY", ORANGE),
        (1240, "CAUSAL VIDEO-ACTION", PURPLE),
    ]
    for x, label, accent in labels:
        pill(draw, (x, 835, x + 430, 895), label, accent, a)
    commit(frame, img)


def render_memory(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 34, 47))
    img, draw = pil_layer(frame)
    text(draw, (110, 135), "推理增强的长短期记忆", 32, color=ORANGE, bold=True, alpha=a)
    text(draw, (110, 195), "Retrieve what matters. Infer what comes next.", 55,
         bold=True, alpha=a, latin=True)
    panel(draw, (95, 290, 1825, 950), a, ORANGE)
    paste_contain(img, FIG_MEMORY, (140, 320, 1780, 755), int(a * 0.94), (55, 28))
    y = 805
    nodes = [
        (155, "GLOBAL\nHISTORY", ORANGE),
        (410, "LOCAL\nACTIVE", ORANGE),
        (665, "EVENT\nBOUNDARY", ORANGE),
        (990, "RETRIEVAL\n+ GATING", CYAN),
        (1320, "IMPLICIT\nSUBGOAL", PURPLE),
        (1590, "ACTION", GREEN),
    ]
    for i, (x, label, accent) in enumerate(nodes):
        aa = int(a * enter(t, 35 + i * 0.35))
        draw.rounded_rectangle((x, y, x + 205, y + 105), radius=18,
                               fill=(10, 12, 23, aa), outline=(*accent, aa), width=3)
        text(draw, (x + 102, y + 52), label, 20, color=accent, bold=True,
             anchor="mm", alpha=aa, latin=True, spacing=5)
        if i < len(nodes) - 1:
            nx = nodes[i + 1][0]
            draw.line((x + 205, y + 52, nx, y + 52), fill=(*INK, aa // 2), width=3)
    commit(frame, img)


def render_multimodal(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 47, 55))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 170), "原生多模态控制  /  NATIVE MULTIMODAL CONTROL", 31,
         color=CYAN, bold=True, anchor="mm", alpha=a)
    text(draw, (WIDTH // 2, 245), "Tell it. Show it. Demonstrate it.", 62,
         bold=True, anchor="mm", alpha=a, latin=True)
    cards = [
        (160, "TEXT", "Fine-grained\nsubtask caption", CYAN),
        (540, "GOAL IMAGE", "First- or\nthird-view target", ORANGE),
        (920, "VIDEO", "Human-to-robot\nor robot-to-robot", PURPLE),
    ]
    for i, (x, title, body, accent) in enumerate(cards):
        aa = int(a * enter(t, 48 + i * 0.3))
        panel(draw, (x, 390, x + 310, 735), aa, accent)
        text(draw, (x + 155, 465), title, 27, color=accent, bold=True,
             anchor="mm", alpha=aa, latin=True)
        text(draw, (x + 155, 600), body, 29, color=INK, anchor="mm",
             alpha=aa, spacing=12, latin=True)
    draw.line((1230, 562, 1375, 562), fill=(*INK, a // 2), width=4)
    panel(draw, (1375, 405, 1760, 720), a, GREEN)
    text(draw, (1567, 495), "UNIFIED WAM", 31, color=GREEN, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (1567, 605), "Video + Action\nPrediction", 33, bold=True,
         anchor="mm", alpha=a, spacing=12, latin=True)
    text(draw, (WIDTH // 2, 870), "AUTONOMOUS PLANNING  ·  INSTRUCTION FOLLOWING  ·  IN-CONTEXT ADAPTATION",
         27, color=MUTED, bold=True, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_training(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 55, 63))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 155), "三阶段训练课程  /  STAGED TRAINING", 31,
         color=CYAN, bold=True, anchor="mm", alpha=a)
    text(draw, (WIDTH // 2, 230), "Control first. Reason next. Adapt last.", 56,
         bold=True, anchor="mm", alpha=a, latin=True)
    stages = [
        ("STAGE 01", "Event-Grounded\nPretraining",
         "Fine-grained grounding\nMultimodal prompts\nShort-term visual memory", CYAN),
        ("STAGE 02", "Memory-Aware\nMid-Training",
         "Event-memory retrieval\nSemantic forcing\nImplicit subgoal planning", ORANGE),
        ("STAGE 03", "Interactive\nPost-Training",
         "Downstream embodiments\nInstruction following\nVisual-prompted control", PURPLE),
    ]
    for i, (stage, title, body, accent) in enumerate(stages):
        aa = int(a * enter(t, 56 + i * 0.35))
        x = 140 + i * 570
        panel(draw, (x, 365, x + 500, 790), aa, accent)
        text(draw, (x + 250, 430), stage, 24, color=accent, bold=True,
             anchor="mm", alpha=aa, latin=True)
        text(draw, (x + 250, 535), title, 33, bold=True, anchor="mm",
             alpha=aa, latin=True, spacing=10)
        draw.line((x + 75, 625, x + 425, 625), fill=(*accent, aa // 2), width=2)
        text(draw, (x + 250, 705), body, 21, color=MUTED, anchor="mm",
             alpha=aa, latin=True, spacing=8)
        if i < 2:
            text(draw, (x + 535, 575), "→", 46, color=INK, bold=True,
                 anchor="mm", alpha=aa)
    text(draw, (WIDTH // 2, 890),
         "EVENT-LEVEL SEMANTICS ARE TRANSFERRED INTO AUTONOMOUS LATENT PLANNING",
         25, color=MUTED, bold=True, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_dataset(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 63, 70))
    img, draw = pil_layer(frame)
    text(draw, (110, 140), "事件级多模态预训练", 32, color=CYAN, bold=True, alpha=a)
    text(draw, (110, 200), "ManipEvent-5M", 67, bold=True, alpha=a, latin=True)
    panel(draw, (705, 300, 1810, 720), a, PURPLE)
    paste_contain(img, FIG_PIPELINE, (745, 340, 1770, 675), int(a * 0.96), (70, 34))
    stats = [("4.89M", "EVENT SEGMENTS"), ("512M", "VIDEO FRAMES"), ("744K", "EPISODES")]
    for i, (value, label) in enumerate(stats):
        y = 350 + i * 175
        aa = int(a * enter(t, 63.8 + i * 0.3))
        text(draw, (125, y), value, 68, color=(CYAN, ORANGE, PURPLE)[i],
             bold=True, alpha=aa, latin=True)
        text(draw, (130, y + 82), label, 25, color=MUTED, bold=True,
             alpha=aa, latin=True)
    pill(draw, (780, 790, 1735, 860),
         "TEXT  +  GOAL IMAGE  +  VIDEO DEMO  +  ACTION TRAJECTORY", PURPLE, a)
    text(draw, (1250, 925), "Heterogeneous data → ordered events → aligned multimodal supervision",
         23, color=MUTED, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_demo_base(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 70, 81))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 155), "From world modeling to real-world control", 56,
         bold=True, anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 220), "预测世界，也执行动作", 31, color=CYAN,
         bold=True, anchor="mm", alpha=a)
    panel(draw, (100, 310, 920, 820), a, CYAN)
    panel(draw, (1000, 310, 1820, 820), a, PURPLE)
    text(draw, (510, 865), "AUTONOMOUS PLANNING", 28, color=CYAN, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (1410, 865), "FINE-GRAINED CONTROL", 28, color=PURPLE, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 965), "Joint video-action prediction provides dense supervision for physically grounded actions.",
         26, color=MUTED, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_outro(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 81, 87))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 280), "WorldScape Policy 2.0", 91, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 410), "Reason. Remember. Act.", 54, color=CYAN,
         bold=True, anchor="mm", alpha=a, latin=True)
    outcomes = [
        ("LONG-HORIZON", "Autonomous Planning"),
        ("STEERABLE", "Fine-Grained Control"),
        ("MULTIMODAL", "In-Context Adaptation"),
    ]
    for i, (top, bottom) in enumerate(outcomes):
        x = 260 + i * 540
        panel(draw, (x, 565, x + 430, 755), a, (ORANGE, PURPLE, CYAN)[i])
        text(draw, (x + 215, 625), top, 24, color=(ORANGE, PURPLE, CYAN)[i],
             bold=True, anchor="mm", alpha=a, latin=True)
        text(draw, (x + 215, 695), bottom, 27, anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 900), "MANIFOLD AI  ·  WORLDSCAPE TEAM", 26,
         color=MUTED, bold=True, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_silent_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Unable to open intermediate video writer")
    scenes = [
        (0, 6, render_title),
        (6, 13, render_challenge),
        (13, 22, render_contributions),
        (22, 34, render_framework),
        (34, 47, render_memory),
        (47, 55, render_multimodal),
        (55, 63, render_training),
        (63, 70, render_dataset),
        (70, 81, render_demo_base),
        (81, 87, render_outro),
    ]
    for index in range(DURATION * FPS):
        t = index / FPS
        frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_background(frame, t)
        for start, end, renderer in scenes:
            if start <= t < end:
                renderer(frame, t)
                break
        writer.write(frame)
        if index % (FPS * 5) == 0:
            print(f"Rendered {index // FPS:02d}s / {DURATION}s", flush=True)
    writer.release()


def generate_music(path: Path) -> None:
    sample_rate = 48000
    count = sample_rate * DURATION
    t = np.arange(count, dtype=np.float64) / sample_rate
    audio = np.zeros(count, dtype=np.float64)
    progression = [
        (55.00, 65.41, 82.41),
        (49.00, 61.74, 73.42),
        (41.20, 55.00, 65.41),
        (49.00, 61.74, 82.41),
    ]
    for block in range(15):
        start = block * 6
        chord = progression[block % len(progression)]
        env = np.clip((t - start) / 1.8, 0, 1) * np.clip((start + 6.5 - t) / 1.8, 0, 1)
        pad = np.zeros_like(t)
        for note in chord:
            pad += np.sin(2 * np.pi * note * t)
            pad += 0.38 * np.sin(2 * np.pi * note * 2.003 * t + 0.8)
        audio += 0.018 * env * pad
    notes = [220.00, 246.94, 329.63, 293.66, 246.94, 369.99, 329.63, 293.66]
    for i, hit in enumerate(np.arange(2.0, DURATION, 0.75)):
        dt = t - hit
        active = ((dt >= 0) & (dt < 1.3)).astype(np.float64)
        env = active * np.exp(-np.maximum(dt, 0) * 4.2)
        note = notes[i % len(notes)]
        pluck = np.sin(2 * np.pi * note * dt) + 0.28 * np.sin(2 * np.pi * note * 2 * dt)
        audio += 0.022 * env * pluck
    for hit in np.arange(4.0, DURATION, 2.0):
        dt = t - hit
        active = ((dt >= 0) & (dt < 0.55)).astype(np.float64)
        env = active * np.exp(-np.maximum(dt, 0) * 11)
        pulse = np.sin(2 * np.pi * (46 * dt - 7 * dt * dt)) * env
        audio += 0.036 * pulse
    for marker in (13, 22, 34, 47, 55, 63, 70, 81):
        dt = t - marker
        active = ((dt >= 0) & (dt < 3.0)).astype(np.float64)
        env = active * np.exp(-np.maximum(dt, 0) * 1.5)
        swell = np.sin(2 * np.pi * 110 * dt) + 0.5 * np.sin(2 * np.pi * 164.81 * dt)
        audio += 0.018 * env * swell
    audio *= np.clip(t / 2.5, 0, 1) * np.clip((DURATION - t) / 4.0, 0, 1)
    audio = np.tanh(audio * 1.15)
    stereo = np.stack((audio, np.roll(audio, 960) * 0.96), axis=1)
    pcm = np.int16(np.clip(stereo, -1, 1) * 32767)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def compose(intermediate: Path, music: Path, output: Path) -> None:
    left = VIDEO_DIR / "demo-memory.mp4"
    right = VIDEO_DIR / "demo-prompt.mp4"
    filters = (
        "[1:v]setpts=PTS-STARTPTS+70/TB,"
        "scale=760:460:force_original_aspect_ratio=decrease,"
        "pad=760:460:(ow-iw)/2:(oh-ih)/2:color=0x090b14[left];"
        "[2:v]setpts=PTS-STARTPTS+70/TB,"
        "scale=760:460:force_original_aspect_ratio=decrease,"
        "pad=760:460:(ow-iw)/2:(oh-ih)/2:color=0x090b14[right];"
        "[0:v][left]overlay=130:335:enable='between(t,70.2,80.7)'[v1];"
        "[v1][right]overlay=1030:335:enable='between(t,70.2,80.7)'[vout]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-v", "warning",
        "-i", str(intermediate), "-i", str(left), "-i", str(right), "-i", str(music),
        "-filter_complex", filters,
        "-map", "[vout]", "-map", "3:a:0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-21:LRA=7:TP=-1.5",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "-t", str(DURATION), str(output),
    ], check=True)


def make_poster(video: Path, output: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", "3.0", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(output),
    ], check=True)


def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    silent = BUILD_DIR / "silent.mp4"
    music = BUILD_DIR / "music.wav"
    output = VIDEO_DIR / "worldscape-policy-method-intro.mp4"
    poster = VIDEO_DIR / "worldscape-policy-method-intro-poster.jpg"
    render_silent_video(silent)
    generate_music(music)
    compose(silent, music, output)
    make_poster(output, poster)
    print(f"Created: {output}")
    print(f"Created: {poster}")


if __name__ == "__main__":
    main()
