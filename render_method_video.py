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

WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 30, 130
FONT_REGULAR = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"

INK = (242, 244, 255)
MUTED = (161, 168, 192)
PURPLE = (145, 88, 231)
CYAN = (45, 224, 225)
ORANGE = (255, 145, 86)
GREEN = (111, 225, 160)
BG = (8, 10, 19)


def font(size: int, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
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


def paste_fill(img: Image.Image, source: Image.Image, box: tuple[int, int, int, int],
               alpha: int) -> None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    src = source.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    src.putalpha(alpha)
    mask = Image.new("L", (w, h), alpha)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, w, h), radius=24, fill=alpha)
    img.paste(src, (x1, y1), mask)


def paste_padded(img: Image.Image, source: Image.Image, box: tuple[int, int, int, int],
                 alpha: int, padding_x: int = 30, padding_y: int = 8) -> None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    card = Image.new("RGBA", (w, h), (255, 255, 255, alpha))
    available_w = w - 2 * padding_x
    available_h = h - 2 * padding_y
    scale = min(available_w / source.width, available_h / source.height)
    size = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
    src = source.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    src.putalpha(alpha)
    card.paste(src, ((w - size[0]) // 2, (h - size[1]) // 2), src)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w, h), radius=24, fill=alpha)
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
        ("01", "Limited Temporal Context", "Similar scenes can imply\ndifferent next robot actions.", PURPLE),
        ("02", "Coarse Language Grounding", "Episode labels obscure\nfine-grained atomic intent.", ORANGE),
        ("03", "Text-Only Interaction", "Visual goals & demonstrations\nremain unavailable to the model.", CYAN),
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
        ("01", "Long Short-Term\nMemory", "Local interaction dynamics\n& global progress tracking", CYAN),
        ("02", "Implicit Subgoal\nReasoning", "Progress-aware retrieval\n& latent subgoal inference", ORANGE),
        ("03", "Native Multimodal\nControl", "Text, goal image & video\nwithin one control interface", PURPLE),
        ("04", "ManipEvent-5M\nDataset", "Nearly five million events\nwith aligned supervision", GREEN),
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
    text(draw, (WIDTH // 2, 865), "LOCAL DYNAMICS  &  GLOBAL PROGRESS", 30,
         color=INK, bold=True, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_framework(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 22, 34))
    img, draw = pil_layer(frame)
    text(draw, (110, 145), "统一的可控世界动作模型", 32, color=CYAN, bold=True, alpha=a)
    text(draw, (110, 205), "One backbone. Multiple ways to steer.", 58,
         bold=True, alpha=a, latin=True)
    panel(draw, (110, 305, 1810, 945), a, PURPLE)
    paste_padded(img, FIG_FRAMEWORK, (145, 335, 1775, 792), a,
                 padding_x=42, padding_y=8)
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
    text(draw, (WIDTH // 2, 115), "推理增强的长短期记忆", 29, color=ORANGE,
         bold=True, anchor="mm", alpha=a)
    text(draw, (WIDTH // 2, 175), "Retrieve what matters. Infer what comes next.", 48,
         bold=True, anchor="mm", alpha=a, latin=True)
    panel(draw, (220, 205, 1700, 1035), a, ORANGE)
    paste_fill(img, FIG_MEMORY, (240, 225, 1680, 1016), int(a * 0.97))
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
    text(draw, (1567, 605), "Video & Action\nPrediction", 33, bold=True,
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
         "Learn multimodal grounding\nwith short-term visual memory", CYAN),
        ("STAGE 02", "Memory-Aware\nMid-Training",
         "Introduce memory retrieval\n& semantic latent forcing", ORANGE),
        ("STAGE 03", "Interactive\nPost-Training",
         "Adapt to target embodiments\n& interactive control modes", PURPLE),
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
        text(draw, (x + 250, 700), body, 21, color=MUTED, anchor="mm",
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
    panel(draw, (680, 280, 1820, 660), a, PURPLE)
    paste_padded(img, FIG_PIPELINE, (690, 290, 1810, 650), int(a * 0.97),
                 padding_x=28, padding_y=10)
    stats = [("4.89M", "EVENT SEGMENTS"), ("512M", "VIDEO FRAMES"), ("744K", "EPISODES")]
    for i, (value, label) in enumerate(stats):
        y = 350 + i * 175
        aa = int(a * enter(t, 63.8 + i * 0.3))
        text(draw, (125, y), value, 68, color=(CYAN, ORANGE, PURPLE)[i],
             bold=True, alpha=aa, latin=True)
        text(draw, (130, y + 82), label, 25, color=MUTED, bold=True,
             alpha=aa, latin=True)
    pill(draw, (780, 735, 1735, 805),
         "TEXT  &  GOAL IMAGE  &  VIDEO DEMO  &  ACTION TRAJECTORY", PURPLE, a)
    text(draw, (1250, 875), "Heterogeneous data → ordered events → aligned multimodal supervision",
         23, color=MUTED, anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_capability_demo(frame: np.ndarray, t: float, start: float, end: float,
                           number: str, title: str, subtitle: str, accent) -> None:
    a = int(255 * scene_alpha(t, start, end))
    img, draw = pil_layer(frame)
    text(draw, (95, 125), f"CAPABILITY  {number} / 04", 24, color=accent,
         bold=True, alpha=a, latin=True)
    text(draw, (WIDTH // 2, 125), title, 46, bold=True, anchor="mm",
         alpha=a, latin=True)
    text(draw, (1820, 125), subtitle, 25, color=accent, bold=True,
         anchor="ra", alpha=a)
    panel(draw, (250, 205, 1670, 1018), a, accent)
    commit(frame, img)


def render_capability_transition(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 70, 74))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 315), "真实世界评估  /  REAL-WORLD EVALUATION", 30,
         color=CYAN, bold=True, anchor="mm", alpha=a)
    text(draw, (WIDTH // 2, 420), "Four Core Capabilities", 76, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 505),
         "Memory-grounded planning, reasoning, transfer & precise control", 30,
         color=MUTED, anchor="mm", alpha=a, latin=True)
    labels = [
        ("01", "LONG-HORIZON", CYAN),
        ("02", "VISUAL REASONING", ORANGE),
        ("03", "SKILL TRANSFER", PURPLE),
        ("04", "INSTRUCTION FOLLOWING", GREEN),
    ]
    for i, (number, label, accent) in enumerate(labels):
        x = 145 + i * 420
        pill(draw, (x, 650, x + 370, 725), f"{number}  {label}", accent, a)
    commit(frame, img)


def render_demo_long(frame: np.ndarray, t: float) -> None:
    render_capability_demo(frame, t, 74, 83, "01",
                           "Long-Horizon Robotic Manipulation", "长程自主规划", CYAN)


def render_demo_memory(frame: np.ndarray, t: float) -> None:
    render_capability_demo(frame, t, 83, 91, "02",
                           "Memory-Dependent Visual Reasoning", "记忆依赖的视觉推理", ORANGE)


def render_demo_cross(frame: np.ndarray, t: float) -> None:
    render_capability_demo(frame, t, 91, 99, "03",
                           "Cross-Embodiment Skill Transfer", "跨形态技能迁移", PURPLE)


def render_demo_fine(frame: np.ndarray, t: float) -> None:
    render_capability_demo(frame, t, 99, 108, "04",
                           "Fine-Grained Instruction Following", "细粒度指令跟随", GREEN)


def render_inference_transition(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 108, 112))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 300), "推理模式  /  INFERENCE MODES", 30,
         color=CYAN, bold=True, anchor="mm", alpha=a)
    text(draw, (WIDTH // 2, 405), "From Global Intent to Precise Control", 69,
         bold=True, anchor="mm", alpha=a, latin=True)
    panel(draw, (280, 600, 890, 790), a, CYAN)
    panel(draw, (1030, 600, 1640, 790), a, PURPLE)
    text(draw, (585, 660), "AUTONOMOUS", 25, color=CYAN, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (585, 730), "Global instruction & memory", 28,
         anchor="mm", alpha=a, latin=True)
    text(draw, (1335, 660), "INTERACTIVE", 25, color=PURPLE, bold=True,
         anchor="mm", alpha=a, latin=True)
    text(draw, (1335, 730), "Fine-grained language control", 28,
         anchor="mm", alpha=a, latin=True)
    commit(frame, img)


def render_inference_modes(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 112, 124))
    img, draw = pil_layer(frame)
    text(draw, (WIDTH // 2, 135), "Two Complementary Inference Modes", 52,
         bold=True, anchor="mm", alpha=a, latin=True)
    text(draw, (WIDTH // 2, 190), "自主规划与交互控制", 27, color=CYAN,
         bold=True, anchor="mm", alpha=a)
    panel(draw, (85, 235, 925, 1005), a, CYAN)
    panel(draw, (995, 235, 1835, 1005), a, PURPLE)
    text(draw, (505, 835), "Autonomous Inference Mode", 29, color=CYAN,
         bold=True, anchor="mm", alpha=a, latin=True)
    text(draw, (505, 915), "Memory infers active subgoals\nfrom one global instruction.", 23,
         color=MUTED, anchor="mm", alpha=a, latin=True, spacing=8)
    text(draw, (1415, 835), "Interactive Inference Mode", 29, color=PURPLE,
         bold=True, anchor="mm", alpha=a, latin=True)
    text(draw, (1415, 915), "Fine-grained language directly\nsteers precise execution.", 23,
         color=MUTED, anchor="mm", alpha=a, latin=True, spacing=8)
    commit(frame, img)


def render_outro(frame: np.ndarray, t: float) -> None:
    a = int(255 * scene_alpha(t, 124, 130))
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
        (70, 74, render_capability_transition),
        (74, 83, render_demo_long),
        (83, 91, render_demo_memory),
        (91, 99, render_demo_cross),
        (99, 108, render_demo_fine),
        (108, 112, render_inference_transition),
        (112, 124, render_inference_modes),
        (124, 130, render_outro),
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
    beat = 60.0 / 108.0
    roots = [73.42, 58.27, 87.31, 65.41]
    chords = [
        (73.42, 87.31, 110.00),
        (58.27, 73.42, 87.31),
        (87.31, 110.00, 130.81),
        (65.41, 82.41, 98.00),
    ]

    def span(start: float, duration: float) -> tuple[slice, np.ndarray]:
        i0 = max(0, int(start * sample_rate))
        i1 = min(count, int((start + duration) * sample_rate))
        local_t = np.arange(max(0, i1 - i0), dtype=np.float64) / sample_rate
        return slice(i0, i1), local_t

    phrase = beat * 8
    for block, start in enumerate(np.arange(0, DURATION, phrase)):
        section, local_t = span(start, phrase + beat * 2)
        attack = np.clip(local_t / 1.2, 0, 1)
        release = np.clip((phrase + beat * 2 - local_t) / 1.4, 0, 1)
        pad = np.zeros_like(local_t)
        for note in chords[block % 4]:
            pad += np.sin(2 * np.pi * note * local_t)
            pad += 0.20 * np.sin(2 * np.pi * note * 2.002 * local_t + 0.7)
        audio[section] += 0.014 * attack * release * pad

    for index, start in enumerate(np.arange(0, DURATION, beat)):
        section, local_t = span(start, 0.26)
        kick_env = np.exp(-local_t * 16)
        kick = np.sin(2 * np.pi * (66 * local_t - 36 * local_t * local_t))
        kick_level = 0.060 if index % 4 == 0 else 0.035
        audio[section] += kick_level * kick_env * kick

        bass_section, bass_t = span(start, beat * 0.78)
        root = roots[(index // 8) % 4]
        bass_env = np.exp(-bass_t * 4.3)
        bass = np.sin(2 * np.pi * root * bass_t) + 0.16 * np.sin(4 * np.pi * root * bass_t)
        audio[bass_section] += 0.034 * bass_env * bass

        if index % 2 == 1:
            clap_section, clap_t = span(start, 0.18)
            clap_env = np.exp(-clap_t * 24)
            clap = np.sin(2 * np.pi * 920 * clap_t) + 0.35 * np.sin(2 * np.pi * 1380 * clap_t)
            audio[clap_section] += 0.009 * clap_env * clap

        hat_section, hat_t = span(start + beat / 2, 0.07)
        hat_env = np.exp(-hat_t * 62)
        hat = np.sin(2 * np.pi * 2100 * hat_t) + 0.22 * np.sin(2 * np.pi * 3150 * hat_t)
        audio[hat_section] += 0.005 * hat_env * hat

    arp_notes = [293.66, 349.23, 440.00, 587.33, 349.23, 523.25, 440.00, 659.25]
    for index, start in enumerate(np.arange(beat / 2, DURATION, beat / 2)):
        arp_section, arp_t = span(start, beat * 0.42)
        note = arp_notes[index % len(arp_notes)]
        arp_env = np.exp(-arp_t * 12)
        arp = np.sin(2 * np.pi * note * arp_t) + 0.14 * np.sin(4 * np.pi * note * arp_t)
        audio[arp_section] += 0.012 * arp_env * arp

    for marker in (13, 22, 34, 47, 55, 63, 70, 74, 83, 91, 99, 108, 112, 124):
        section, local_t = span(marker - 1.0, 1.6)
        rise_env = np.sin(np.pi * np.clip(local_t / 1.6, 0, 1)) ** 2
        phase = 2 * np.pi * (150 * local_t + 190 * local_t * local_t)
        audio[section] += 0.014 * rise_env * np.sin(phase)

    audio *= np.clip(t / 2.2, 0, 1) * np.clip((DURATION - t) / 4.0, 0, 1)
    audio = np.tanh(audio * 1.2)
    stereo = np.stack((audio, np.roll(audio, 880) * 0.95), axis=1)
    pcm = np.int16(np.clip(stereo, -1, 1) * 32767)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def compose(intermediate: Path, music: Path, output: Path) -> None:
    long_horizon = VIDEO_DIR / "long-horizon-robotic-manipulation.mp4"
    memory_reasoning = VIDEO_DIR / "memory-dependent-visual-reasoning.mp4"
    cross_embodiment = VIDEO_DIR / "cross-embodiment-skill-transfer.mp4"
    fine_grained = VIDEO_DIR / "fine-grained-instruction-following.mp4"
    autonomous_mode = VIDEO_DIR / "demo-memory.mp4"
    interactive_mode = VIDEO_DIR / "demo-prompt.mp4"
    filters = (
        "[1:v]trim=duration=9,setpts=PTS-STARTPTS,scale=1380:776,setsar=1,"
        "format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st=8.4:d=0.6:alpha=1,"
        "setpts=PTS+74/TB[demo1];"
        "[2:v]trim=duration=8,setpts=PTS-STARTPTS,scale=1380:776,setsar=1,"
        "format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st=7.4:d=0.6:alpha=1,"
        "setpts=PTS+83/TB[demo2];"
        "[3:v]trim=duration=8,setpts=PTS-STARTPTS,scale=1380:776,setsar=1,"
        "format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st=7.4:d=0.6:alpha=1,"
        "setpts=PTS+91/TB[demo3];"
        "[4:v]trim=duration=9,setpts=PTS-STARTPTS,scale=1380:776,setsar=1,"
        "format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st=8.4:d=0.6:alpha=1,"
        "setpts=PTS+99/TB[demo4];"
        "[5:v]trim=duration=12,setpts=PTS-STARTPTS,scale=700:560,setsar=1,"
        "format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st=11.4:d=0.6:alpha=1,"
        "setpts=PTS+112/TB[auto];"
        "[6:v]trim=duration=12,setpts=PTS-STARTPTS,scale=700:560,setsar=1,"
        "format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st=11.4:d=0.6:alpha=1,"
        "setpts=PTS+112/TB[interactive];"
        "[0:v][demo1]overlay=270:225:enable='between(t,74,83)'[v1];"
        "[v1][demo2]overlay=270:225:enable='between(t,83,91)'[v2];"
        "[v2][demo3]overlay=270:225:enable='between(t,91,99)'[v3];"
        "[v3][demo4]overlay=270:225:enable='between(t,99,108)'[v4];"
        "[v4][auto]overlay=155:255:enable='between(t,112,124)'[v5];"
        "[v5][interactive]overlay=1065:255:enable='between(t,112,124)'[vout]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-v", "warning",
        "-i", str(intermediate),
        "-stream_loop", "-1", "-i", str(long_horizon),
        "-stream_loop", "-1", "-i", str(memory_reasoning),
        "-stream_loop", "-1", "-i", str(cross_embodiment),
        "-stream_loop", "-1", "-i", str(fine_grained),
        "-stream_loop", "-1", "-i", str(autonomous_mode),
        "-stream_loop", "-1", "-i", str(interactive_mode),
        "-i", str(music),
        "-filter_complex", filters,
        "-map", "[vout]", "-map", "7:a:0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-20:LRA=7:TP=-1.5",
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
