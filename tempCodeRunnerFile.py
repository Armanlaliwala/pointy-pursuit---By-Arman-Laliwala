
"""
Fingertip Catch Game (OpenCV + MediaPipe Hands)
-----------------------------------------------
Single-file Python script.

Requirements:
    pip install opencv-python mediapipe

Usage:
    python cv_fingertip_game.py
"""

import cv2
import time
import random
import math
from dataclasses import dataclass
from typing import Tuple, List

# ========== Adjustable Parameters ==========
WINDOW_W, WINDOW_H = 960, 540          # Camera frame resolution (helps FPS). Try 1280x720 on faster machines.
TARGET_FPS = 30                        # For spawn timing & UI pacing.
MAX_MISSES = 3

# Ball behavior
BALL_RADIUS_RANGE = (16, 28)           # Min/Max pixel radius
BALL_SPEED_RANGE = (4.0, 10.0)         # Min/Max downward speed (pixels per frame)
SPAWN_INTERVAL_SEC = 0.5               # Average seconds between spawns (Poisson-ish using jitter)

# Probabilities and point values
# Red 50% (+10), Green 40% (+20), Blue 10% (+50) by default
BALL_PROBABILITIES = {
    "red": 0.50,
    "green": 0.40,
    "blue": 0.10,
}
BALL_POINTS = {
    "red": 10,
    "green": 20,
    "blue": 50,
}

# Rarity hint: make blue rare by keeping prob low
# =============================================

# MediaPipe imports (kept local so script still loads without immediately failing if not installed)
try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("mediapipe is required. Install with: pip install mediapipe") from e

# ========== Game Objects ==========
@dataclass
class Ball:
    x: float
    y: float
    r: int
    color: Tuple[int, int, int]  # BGR
    vy: float
    points: int
    alive: bool = True

    def update(self):
        self.y += self.vy

    def draw(self, frame):
        if not self.alive:
            return
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, thickness=-1)
        # Optional: subtle outline to help visibility
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, (30, 30, 30), thickness=2)

# ========== Utility ==========
def weighted_choice(d):
    # d: dict of key->prob; returns key by weight
    rnd = random.random()
    cum = 0.0
    for k, p in d.items():
        cum += p
        if rnd <= cum:
            return k
    # Fallback due to float precision
    return list(d.keys())[-1]

def color_bgr(name: str) -> Tuple[int, int, int]:
    if name == "red":
        return (36, 36, 230)
    if name == "green":
        return (46, 168, 79)
    if name == "blue":
        return (245, 152, 70)  # OpenCV uses BGR; this is orange-ish if wrong; adjust below
    # Correct pure-ish BGR options:
    if name == "red_pure":
        return (0, 0, 255)
    if name == "green_pure":
        return (0, 200, 0)
    if name == "blue_pure":
        return (255, 0, 0)
    return (255, 255, 255)

# Use vivid pure colors for clarity
COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 0, 0),
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ========== Main ==========
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    # MediaPipe Hands setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,  # Faster
    )

    # Game state
    score = 0
    misses = 0
    balls: List[Ball] = []
    last_spawn_time = 0.0
    spawn_jitter = 0.0
    start_screen = True
    game_over = False

    # Start button (circle) in center
    btn_radius = 80
    btn_center = (WINDOW_W // 2, WINDOW_H // 2)

    # For FPS
    prev_time = time.time()
    frame_time_accum = 0.0
    frame_counter = 0
    measured_fps = TARGET_FPS

    # Fingertip state
    fingertip_px = None  # (x, y) in pixels
    fingertip_visible = False

    def reset_game():
        nonlocal score, misses, balls, last_spawn_time, spawn_jitter, start_screen, game_over
        score = 0
        misses = 0
        balls.clear()
        last_spawn_time = time.time()
        spawn_jitter = random.uniform(-0.25, 0.25) * SPAWN_INTERVAL_SEC
        start_screen = False
        game_over = False

    # Ensure probabilities sum to 1
    psum = sum(BALL_PROBABILITIES.values())
    if abs(psum - 1.0) > 1e-6:
        # Normalize
        for k in list(BALL_PROBABILITIES.keys()):
            BALL_PROBABILITIES[k] /= psum

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        h, w = frame.shape[:2]

        # Update FPS stats
        now = time.time()
        dt = now - prev_time
        prev_time = now
        frame_time_accum += dt
        frame_counter += 1
        if frame_time_accum >= 0.5:  # update twice a second
            measured_fps = frame_counter / frame_time_accum
            frame_counter = 0
            frame_time_accum = 0.0

        # ---- Detect hands & fingertip ----
        fingertip_visible = False
        fingertip_px = None

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            # Use first hand
            hand_landmarks = result.multi_hand_landmarks[0]
            # Index fingertip landmark: 8
            tip = hand_landmarks.landmark[8]
            # Convert normalized coords to pixels
            x_px = int(tip.x * w)
            y_px = int(tip.y * h)
            fingertip_px = (x_px, y_px)
            fingertip_visible = True

            # Optional: render a small crosshair on fingertip
            cv2.circle(frame, (x_px, y_px), 6, (255, 255, 255), -1)
            cv2.circle(frame, (x_px, y_px), 10, (0, 0, 0), 2)

        # ---- Start screen ----
        if start_screen:
            # Dim background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), thickness=-1)
            alpha = 0.35
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Render start button
            cv2.circle(frame, btn_center, btn_radius, (60, 200, 255), -1)
            cv2.circle(frame, btn_center, btn_radius, (0, 0, 0), 3)
            cv2.putText(frame, "START GAME", (btn_center[0] - 150, btn_center[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.putText(frame, "Touch the button with your index fingertip",
                        (int(w*0.14), int(h*0.15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            if fingertip_visible:
                dx = fingertip_px[0] - btn_center[0]
                dy = fingertip_px[1] - btn_center[1]
                if dx*dx + dy*dy <= btn_radius*btn_radius:
                    reset_game()

        else:
            # ---- Game loop ----
            # Spawn balls based on time
            if not game_over:
                if now - last_spawn_time >= max(0.1, SPAWN_INTERVAL_SEC + spawn_jitter):
                    # Decide color by weight
                    cname = weighted_choice(BALL_PROBABILITIES)
                    color = COLOR_MAP[cname]
                    points = BALL_POINTS[cname]

                    r = random.randint(BALL_RADIUS_RANGE[0], BALL_RADIUS_RANGE[1])
                    x = random.randint(r + 2, w - r - 2)
                    y = -r  # start just above the screen
                    vy = random.uniform(BALL_SPEED_RANGE[0], BALL_SPEED_RANGE[1])

                    balls.append(Ball(x=x, y=y, r=r, color=color, vy=vy, points=points))

                    last_spawn_time = now
                    # New jitter to avoid rhythm
                    spawn_jitter = random.uniform(-0.25, 0.25) * SPAWN_INTERVAL_SEC

                # Update and draw balls
                for ball in balls:
                    if not ball.alive:
                        continue
                    ball.update()
                    ball.draw(frame)

                # Collision with fingertip
                if fingertip_visible:
                    fx, fy = fingertip_px
                    for ball in balls:
                        if not ball.alive:
                            continue
                        dx = fx - ball.x
                        dy = fy - ball.y
                        if dx*dx + dy*dy <= (ball.r * ball.r):
                            score += ball.points
                            ball.alive = False

                # Remove off-screen or dead (miss tracking)
                for ball in balls:
                    if ball.alive and (ball.y - ball.r > h):
                        misses += 1
                        ball.alive = False

                balls = [b for b in balls if b.alive and b.y - b.r <= h + 5]

                # Game over?
                if misses >= MAX_MISSES:
                    game_over = True

            # ---- HUD ----
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Score: {score}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Missed: {misses}/{MAX_MISSES}", (230, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {int(measured_fps)}", (480, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 255, 100), 2, cv2.LINE_AA)

            # Hint text
            cv2.putText(frame, "Red:+10  Green:+20  Blue:+50  |  Miss 3 = Game Over",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Game over screen overlay
            if game_over:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), thickness=-1)
                alpha = 0.5
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                cv2.putText(frame, "GAME OVER", (int(w*0.33), int(h*0.4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)
                cv2.putText(frame, f"Final Score: {score}", (int(w*0.35), int(h*0.52)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

                # Re-use START button as "RESTART"
                cv2.circle(frame, btn_center, btn_radius, (60, 200, 255), -1)
                cv2.circle(frame, btn_center, btn_radius, (0, 0, 0), 3)
                cv2.putText(frame, "RESTART", (btn_center[0] - 115, btn_center[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)

                if fingertip_visible:
                    dx = fingertip_px[0] - btn_center[0]
                    dy = fingertip_px[1] - btn_center[1]
                    if dx*dx + dy*dy <= btn_radius*btn_radius:
                        reset_game()

        # ---- Render ----
        cv2.imshow("Fingertip Catch (OpenCV + MediaPipe)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
