import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
#--------- Face Setup ---------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


#--------- Punch Circle Class ---------
punches = []
class PunchCircle:
    max_radius = 80
    landed_threshold_time = 2.0
    def __init__(self, x, y, radius, grow_speed, last_punch_time):
        self.x = x
        self.y = y
        self.radius = radius
        self.grow_speed = grow_speed
        self.last_punch_time = last_punch_time
        self.landed = False
        self.COLOR = (0, 100, 255)
    def draw_circle(self, frame, time_now=None):
        if time_now is None:
            time_now = time.time()
        if time_now - self.last_punch_time > 0.01:
            self.radius += self.grow_speed
        if self.radius > self.max_radius:
            self.grow_speed = 0
            self.landed = True
            self.COLOR = (0, 0, 255)
        if self.landed and time_now - self.last_punch_time > self.landed_threshold_time:
            punches.remove(self)
        cv2.circle(frame, (self.x, self.y), self.radius, self.COLOR, -1)
    def get_coord(self):
        return (self.x, self.y, self.radius)
    def hit(self, x, y):
        dist = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
        return dist < self.radius


#--------- Facial Point Class ---------
class FacialPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.hit = False
    def update(self, x, y):
        self.x = x
        self.y = y
    def setColor(self, color):
        if color:
            self.color = color
        else:
            self.color = (0,255,0)
    def check_hit(self, punches):
        for punch in punches:
            if punch.hit(self.x, self.y) and punch.landed:
                self.hit = True
                return True
        self.hit = False
        return False
    def draw(self, frame):
        color = (0, 255, 0) if not self.hit else (0, 255, 255)
        size = 2 if not self.hit else 4
        cv2.circle(frame, (self.x, self.y), size, color, -1)

facial_points = [FacialPoint(0, 0) for _ in range(468)]


#--------- Main Loop ---------
cap = cv2.VideoCapture(0)
punch_spawn_cooldown = 2.0
last_spawn_time = time.time() - punch_spawn_cooldown
health = 500
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
player_hit = False
player_recover_time = 2.0
player_hit_time = 0
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not success:
        break

    #spawn punches
    for punch in punches:
        punch.draw_circle(frame)
        
    #process frame for face landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    #track face landmarks
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w,_ = frame.shape
        #only check hits if player not recently hit
        if not player_hit:
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                facial_points[i].update(x, y)
                if facial_points[i].check_hit(punches):
                    health -= 1
                    health = max(0, health)
                    player_hit = True
                    player_hit_time = time.time()
                facial_points[i].draw(frame)
        #reset player being hit after recover time
        else:
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                facial_points[i].update(x, y)
                facial_points[i].draw(frame)
            if time.time() - player_hit_time > player_recover_time:
                player_hit = False
    #health bar
    cv2.rectangle(frame, (10, 10), (10 + health, 30), GREEN, -1)
    #spawn new punch
    current_time = time.time()
    if current_time - last_spawn_time > punch_spawn_cooldown:
        new_x = int((0.1 + 0.8 * time.time() % 1) * frame.shape[1])
        new_y = int((0.1 + 0.8 * time.time() % 1) * frame.shape[0])
        punches.append(PunchCircle(new_x, new_y, 5, 3, current_time))
        last_spawn_time = current_time
    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break



cap.release()
cv2.destroyAllWindows()
