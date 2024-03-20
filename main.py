import math

import cv2
import mediapipe as mp
import pyautogui

# PyAutoGUIのセットアップ
pyautogui.FAILSAFE = False
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# Mediapipeのセットアップ
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# OpenCVのセットアップ
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * frame.shape[1])
                wrist_y = int(wrist.y * frame.shape[0])

                mouse_x, mouse_y = pyautogui.position()

                distance = math.sqrt((wrist_x - mouse_x) ** 2 + (wrist_y - mouse_y) ** 2)

                if distance < 12:
                    # 右手かどうかを判定
                    if results.multi_handedness[0].classification[0].label == "Right":
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                        # 各指の位置を取得
                        thumb_x, thumb_y = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])
                        index_x, index_y = int(index.x * frame.shape[1]), int(index.y * frame.shape[0])
                        middle_x, middle_y = int(middle.x * frame.shape[1]), int(middle.y * frame.shape[0])
                        ring_x, ring_y = int(ring.x * frame.shape[1]), int(ring.y * frame.shape[0])
                        pinky_x, pinky_y = int(pinky.x * frame.shape[1]), int(pinky.y * frame.shape[0])

                        # ジェスチャーの判定
                        if (
                                thumb_y > index_y > middle_y > ring_y > pinky_y and
                                thumb_x < index_x < middle_x < ring_x < pinky_x
                        ):
                            pyautogui.doubleClick()  # 握りこぶしのジェスチャーであればダブルクリック
                        else:
                            pyautogui.rightClick()  # ピースのジェスチャーであれば右クリック

                pyautogui.moveTo(wrist_x, wrist_y)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
