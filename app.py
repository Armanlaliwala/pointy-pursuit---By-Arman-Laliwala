import cv2
import time
import random
import numpy as np
import streamlit as st
import mediapipe as mp
from collections import deque

# Game constants
BALL_RADIUS_RANGE = (22, 36)
BALL_SPEED_RANGE = (2.5, 6.0)
SPAWN_INTERVAL = 0.8
MAX_MISSES = 3
COLORS = {
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 0, 0)
}
POINTS = {"red": 10, "green": 20, "blue": 50}
PROBABILITIES = {"red": 0.45, "green": 0.40, "blue": 0.15}

class Ball:
    def __init__(self, x, r, color, vy, points):
        self.x = x
        self.y = -r
        self.r = r
        self.color = color
        self.vy = vy
        self.points = points
        self.active = True

    def update(self):
        self.y += self.vy

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, (30, 30, 30), 2)

def main():
    st.title("🎯 Fingertip Catch Game")
    st.markdown("Use your finger to catch falling balls!")
    
    # Initialize game state
    if 'game' not in st.session_state:
        st.session_state.game = {
            'score': 0,
            'misses': 0,
            'balls': deque(),
            'last_spawn': time.time(),
            'game_over': False,
            'started': False,
            'camera_error': False
        }
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    
    # Camera setup with error handling
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.session_state.game['camera_error'] = True
            st.error("Could not access camera. Please ensure camera permissions are granted.")
            return
    except Exception as e:
        st.session_state.game['camera_error'] = True
        st.error(f"Camera error: {str(e)}")
        return
    
    img_placeholder = st.empty()
    
    while True:
        # Create a blank frame as fallback
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        
        # Try to read camera frame
        if not st.session_state.game['camera_error']:
            ret, camera_frame = cap.read()
            if ret:
                frame = cv2.flip(camera_frame, 1)
                h, w = frame.shape[:2]
        
        # Process hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        fingertip = None
        
        if results.multi_hand_landmarks:
            tip = results.multi_hand_landmarks[0].landmark[8]
            fingertip = (int(tip.x * w), int(tip.y * h))
            cv2.circle(frame, fingertip, 8, (255, 255, 255), -1)
        
        # Start screen
        if not st.session_state.game['started']:
            cv2.putText(frame, "Fingertip Catch Game", (int(w*0.15), int(h*0.35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
            cv2.putText(frame, "Touch START to begin", 
                        (int(w*0.25), int(h*0.45)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            
            bx, by, bw, bh = int(w*0.4), int(h*0.6), 180, 70
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,255,0), -1)
            cv2.putText(frame, "START", (bx+20, by+45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
            
            if fingertip and bx < fingertip[0] < bx+bw and by < fingertip[1] < by+bh:
                st.session_state.game['started'] = True
                st.session_state.game.update({
                    'score': 0,
                    'misses': 0,
                    'balls': deque(),
                    'last_spawn': time.time(),
                    'game_over': False
                })
            
            img_placeholder.image(frame, channels="BGR")
            continue
        
        # Game over screen
        if st.session_state.game['game_over']:
            cv2.putText(frame, "GAME OVER", (int(w*0.25), int(h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            
            bx, by, bw, bh = int(w*0.4), int(h*0.65), 220, 80
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255,0,0), -1)
            cv2.putText(frame, "RESTART", (bx+20, by+55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            
            if fingertip and bx < fingertip[0] < bx+bw and by < fingertip[1] < by+bh:
                st.session_state.game.update({
                    'score': 0,
                    'misses': 0,
                    'balls': deque(),
                    'last_spawn': time.time(),
                    'game_over': False
                })
            
            img_placeholder.image(frame, channels="BGR")
            continue
        
        # Game logic
        now = time.time()
        if now - st.session_state.game['last_spawn'] > SPAWN_INTERVAL:
            color = random.choices(list(PROBABILITIES.keys()), 
                                 weights=list(PROBABILITIES.values()))[0]
            r = random.randint(*BALL_RADIUS_RANGE)
            x = random.randint(r, w-r)
            vy = random.uniform(*BALL_SPEED_RANGE)
            st.session_state.game['balls'].append(
                Ball(x, r, COLORS[color], vy, POINTS[color])
            )
            st.session_state.game['last_spawn'] = now
        
        for ball in list(st.session_state.game['balls']):
            ball.update()
            ball.draw(frame)
            
            if fingertip:
                dx = fingertip[0] - ball.x
                dy = fingertip[1] - ball.y
                if dx*dx + dy*dy <= (ball.r + 8)**2:
                    st.session_state.game['score'] += ball.points
                    ball.active = False
            
            if ball.y - ball.r > h:
                st.session_state.game['misses'] += 1
                ball.active = False
            
            if not ball.active:
                st.session_state.game['balls'].remove(ball)
        
        if st.session_state.game['misses'] >= MAX_MISSES:
            st.session_state.game['game_over'] = True
        
        # Draw HUD
        cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
        cv2.putText(frame, f"Score: {st.session_state.game['score']}", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Misses: {st.session_state.game['misses']}/{MAX_MISSES}", 
                    (230,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
        img_placeholder.image(frame, channels="BGR")
        
        # Small delay to prevent high CPU usage
        time.sleep(0.01)
    
    if cap:
        cap.release()

if __name__ == "__main__":
    main()
