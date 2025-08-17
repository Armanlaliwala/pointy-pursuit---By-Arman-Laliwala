import cv2
import time
import random
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

# ==== Game Parameters ====
BALL_RADIUS_RANGE = (22, 36)
BALL_SPEED_RANGE = (2.5, 6.0)
SPAWN_INTERVAL_SEC = 0.8
MAX_MISSES = 3
COLOR_MAP = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}
BALL_POINTS = {"red": 10, "green": 20, "blue": 50}
BALL_PROBABILITIES = {"red": 0.45, "green": 0.40, "blue": 0.15}

# ==== Ball class ====
class Ball:
    def __init__(self, x, y, r, color, vy, points):
        self.x, self.y, self.r = x, y, r
        self.color, self.vy, self.points = color, vy, points
        self.alive = True

    def update(self):
        self.y += self.vy

    def draw(self, frame):
        if self.alive:
            cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.r, (30, 30, 30), 2)

# ==== Weighted choice ====
def weighted_choice(d):
    rnd, cum = random.random(), 0.0
    for k, p in d.items():
        cum += p
        if rnd <= cum:
            return k
    return list(d.keys())[-1]

# ==== Streamlit Video Processor ====
class GameProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.reset_game()
        self.started = False  # game starts only after pressing Start

    def reset_game(self):
        self.score, self.misses, self.balls = 0, 0, []
        self.last_spawn_time, self.spawn_jitter = time.time(), 0.0
        self.game_over = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)   # Fix mirror

        h, w = img.shape[:2]
        now = time.time()

        # Detect fingertip
        fingertip = None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if result.multi_hand_landmarks:
            tip = result.multi_hand_landmarks[0].landmark[8]
            fingertip = (int(tip.x * w), int(tip.y * h))
            cv2.circle(img, fingertip, 8, (255, 255, 255), -1)

        # === Start Screen ===
        if not self.started:
            cv2.putText(img, "Fingertip Catch Game", (int(w*0.15), int(h*0.35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
            cv2.putText(img, "You have 3 lives. Miss 3 balls = Game Over",
                        (int(w*0.1), int(h*0.45)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            # Draw Start button
            bx, by, bw, bh = int(w*0.4), int(h*0.6), 180, 70
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (0,255,0), -1)
            cv2.putText(img, "START", (bx+20, by+45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)

            if fingertip and bx < fingertip[0] < bx+bw and by < fingertip[1] < by+bh:
                self.started = True
                self.reset_game()
            return img

        # === Spawn balls ===
        if not self.game_over and now - self.last_spawn_time > SPAWN_INTERVAL_SEC + self.spawn_jitter:
            cname = weighted_choice(BALL_PROBABILITIES)
            r = random.randint(*BALL_RADIUS_RANGE)
            x = random.randint(r, w-r)
            vy = random.uniform(*BALL_SPEED_RANGE)
            self.balls.append(Ball(x, -r, r, COLOR_MAP[cname], vy, BALL_POINTS[cname]))
            self.last_spawn_time, self.spawn_jitter = now, random.uniform(-0.25, 0.25) * SPAWN_INTERVAL_SEC

        # === Update balls ===
        for ball in self.balls:
            if not ball.alive: continue
            ball.update()
            ball.draw(img)
            if fingertip:
                dx, dy = fingertip[0]-ball.x, fingertip[1]-ball.y
                if dx*dx + dy*dy <= (ball.r+8)**2:
                    self.score += ball.points
                    ball.alive = False
            if ball.alive and ball.y - ball.r > h:
                self.misses += 1
                ball.alive = False

        self.balls = [b for b in self.balls if b.alive]
        if self.misses >= MAX_MISSES:
            self.game_over = True

        # === HUD ===
        cv2.rectangle(img, (0,0), (w,50), (0,0,0), -1)
        cv2.putText(img, f"Score: {self.score}", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, f"Misses: {self.misses}/{MAX_MISSES}", (230,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # === Game Over Screen ===
        if self.game_over:
            cv2.putText(img, "GAME OVER", (int(w*0.25), int(h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

            # Restart button
            bx, by, bw, bh = int(w*0.4), int(h*0.65), 220, 80
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255,0,0), -1)
            cv2.putText(img, "RESTART", (bx+20, by+55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

            if fingertip and bx < fingertip[0] < bx+bw and by < fingertip[1] < by+bh:
                self.reset_game()
                self.started = True

        return img

# ==== Streamlit App ====
st.title("🎯 Fingertip Catch Game (Web Version)")
st.markdown("Move your index finger to catch falling balls. Miss 3 = Game Over!")

webrtc_streamer(
    key="game",
    video_transformer_factory=GameProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
