import math
import time
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class PositionFilter:
    def __init__(self, min_factor=0.15, max_factor=0.45, distance_scale=50.0):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.distance_scale = distance_scale

    def smooth(self, prev: float, target: float) -> float:
        dist = abs(target - prev)
        factor = self.min_factor + (
            self.max_factor - self.min_factor
        ) * min(dist / self.distance_scale, 1.0)
        return prev + (target - prev) * factor


class VirtualMouse:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.screen_w, self.screen_h = pyautogui.size()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.filter_x = PositionFilter()
        self.filter_y = PositionFilter()

        self.prev_x, self.prev_y = 0.0, 0.0
        self.first_move = True
        self.is_dragging = False
        self.prev_left_y = None
        self.pinch_ema = 0.0

        self.margin_ratio = 0.15
        self.pinch_threshold = 0.22

    def run(self):
        if not self.cap.isOpened():
            print("HATA: Kamera açılamadı!")
            return

        print("Sanal Fare Başlatıldı. Çıkmak için 'q' tuşuna basın.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            margin_x = int(w * self.margin_ratio)
            margin_y = int(h * self.margin_ratio)
            cv2.rectangle(
                frame,
                (margin_x, margin_y),
                (w - margin_x, h - margin_y),
                (0, 255, 0),
                2,
            )

            right_found = False
            left_found = False

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    label = hand_handedness.classification[0].label
                    landmarks = results.multi_hand_landmarks[idx].landmark

                    self.mp_draw.draw_landmarks(
                        frame,
                        results.multi_hand_landmarks[idx],
                        self.mp_hands.HAND_CONNECTIONS,
                    )

                    if label == "Right":
                        right_found = True
                        self._handle_right_hand(landmarks, w, h)
                    elif label == "Left":
                        left_found = True
                        self._handle_left_hand(landmarks, h)

            if not right_found:
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                self.first_move = True

            if not left_found:
                self.prev_left_y = None

            cv2.imshow("Sanal Fare (Aktif)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

    def _handle_right_hand(self, lm, w, h):
        idx_mcp = lm[5]
        ring_mcp = lm[9]
        ref_x = (idx_mcp.x + ring_mcp.x) / 2
        ref_y = (idx_mcp.y + ring_mcp.y) / 2

        margin_x = w * self.margin_ratio
        margin_y = h * self.margin_ratio

        mapped_x = (
            ((ref_x * w) - margin_x) / max(1, (w - 2 * margin_x))
        ) * self.screen_w
        mapped_y = (
            ((ref_y * h) - margin_y) / max(1, (h - 2 * margin_y))
        ) * self.screen_h

        target_x = max(0.0, min(float(self.screen_w), mapped_x))
        target_y = max(0.0, min(float(self.screen_h), mapped_y))

        if self.first_move:
            self.prev_x, self.prev_y = target_x, target_y
            self.first_move = False

        curr_x = self.filter_x.smooth(self.prev_x, target_x)
        curr_y = self.filter_y.smooth(self.prev_y, target_y)

        pyautogui.moveTo(int(curr_x), int(curr_y))
        self.prev_x, self.prev_y = curr_x, curr_y

        idx_tip = (lm[8].x * w, lm[8].y * h)
        th_tip = (lm[4].x * w, lm[4].y * h)
        wrist = (lm[0].x * w, lm[0].y * h)
        mid_mcp = (lm[9].x * w, lm[9].y * h)

        hand_size = max(
            1.0, math.hypot(mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1])
        )
        raw_dist = (
            math.hypot(idx_tip[0] - th_tip[0], idx_tip[1] - th_tip[1])
            / hand_size
        )

        self.pinch_ema = 0.4 * raw_dist + 0.6 * self.pinch_ema

        if self.pinch_ema < self.pinch_threshold:
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
        else:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False

    def _handle_left_hand(self, lm, h):
        palm_y = lm[9].y * h
        if self.prev_left_y is not None:
            delta_y = palm_y - self.prev_left_y
            if abs(delta_y) > 3.0:
                scroll_amount = int(-delta_y * 0.5)
                if scroll_amount != 0:
                    pyautogui.scroll(scroll_amount)
        self.prev_left_y = palm_y


if __name__ == "__main__":
    app = VirtualMouse()
    app.run()