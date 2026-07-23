import cv2
import mediapipe as mp
import pyautogui

# MediaPipe el takibi
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Kamera
cap = cv2.VideoCapture(0)

# Ekran boyutu
screen_w, screen_h = pyautogui.size()

# İmleç hassasiyeti için smoothing
prev_x, prev_y = 0, 0
smooth_factor = 0.5

# PyAutoGUI ayarları
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

click_cooldown = 0  # üst üste tıklamayı engellemek için

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # ayna görüntü
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # İşaret parmağı ucu (landmark 8)
        index_finger = hand.landmark[8]
        # Başparmak ucu (landmark 4)
        thumb_finger = hand.landmark[4]
        # Orta parmak ucu (landmark 12)
        middle_finger = hand.landmark[12]

        # İşaret parmağı koordinatlarını ekran boyutuna map et
        ix = index_finger.x * screen_w
        iy = index_finger.y * screen_h

        # Yumuşatma (titremeyi azalt)
        curr_x = prev_x + (ix - prev_x) * smooth_factor
        curr_y = prev_y + (iy - prev_y) * smooth_factor

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Tıklama: başparmak ve işaret parmağı mesafesi
        distance_click = abs(index_finger.x - thumb_finger.x) + abs(index_finger.y - thumb_finger.y)

        # Sağ tık: başparmak ve orta parmak mesafesi
        distance_right = abs(middle_finger.x - thumb_finger.x) + abs(middle_finger.y - thumb_finger.y)

        if distance_click < 0.05 and click_cooldown == 0:
            pyautogui.click()
            click_cooldown = 10  # frame cinsinden bekleme

        if distance_right < 0.05 and click_cooldown == 0:
            pyautogui.rightClick()
            click_cooldown = 10

        if click_cooldown > 0:
            click_cooldown -= 1

        # Görsel: el çizimi
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Moruk Imlecini Elinle Yonet", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()