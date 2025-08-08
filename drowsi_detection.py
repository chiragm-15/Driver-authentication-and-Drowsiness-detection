import os
import glob
import cv2
import dlib
import numpy as np
from math import hypot
from pygame import mixer
import face_recognition
from twilio.rest import Client
import time

# --------------------------- Config ---------------------------
PREDICTOR_PATH = r".\shape_predictor_68_face_landmarks.dat"
IMAGES_DIR = r"Images"
CAM_INDEX = 0
FRAME_SIZE = (850, 850)   # (width, height)
SELFIE_FLIP = True        # True -> mirror view

# Thresholds (tune per environment/driver as needed)
EYE_RATIO_THRESHOLD = 4.00
EYE_RATIO_STRONG = 4.30
MOUTH_OPEN_THRESHOLD = 0.380
FOREHEAD_CHANGE_THRESHOLD = 0.25
FRAMES_OVER_LIMIT_FOR_WARNING = 8

# Stranger/SMS behavior
SMS_COOLDOWN_SEC = 15
MAX_SMS = 2
MAX_STRANGER_FRAMES = 13  # exit after this many frames of stranger

# Face recognition behavior
ENCODING_FRAME_DOWNSCALE = 0.25
FACE_TOLERANCE = 0.50  # lower = stricter (fewer false accepts)

# Audio files (keep next to script or give absolute paths)
ALARM_WAV = r"alarm.wav"   # stranger
DANGER_WAV = r"danger.wav" # drowsiness warning
# --------------------------------------------------------------


class SimpleFacerec:
    def __init__(self, frame_resizing=ENCODING_FRAME_DOWNSCALE, tolerance=FACE_TOLERANCE):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing
        self.tolerance = tolerance

    def load_encoding_images(self, images_path):
        images = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images)} encoding images found.")
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[skip] unreadable: {img_path}")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_img)
            if not encs:
                print(f"[skip] no face: {img_path}")
                continue
            self.known_face_encodings.append(encs[0])
            name = os.path.splitext(os.path.basename(img_path))[0]
            self.known_face_names.append(name)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        """Return 1 if any face in frame matches a known encoding (within tolerance), else 0."""
        small = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        if not encs or not self.known_face_encodings:
            return 0

        for enc in encs:
            dists = face_recognition.face_distance(self.known_face_encodings, enc)
            j = np.argmin(dists)
            if dists[j] <= self.tolerance:
                return 1
        return 0


def mid(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def eye_aspect_ratio(eye_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(eye_landmark[0]).x, face_roi_landmark.part(eye_landmark[0]).y)
    right_point = (face_roi_landmark.part(eye_landmark[3]).x, face_roi_landmark.part(eye_landmark[3]).y)
    center_top = mid(face_roi_landmark.part(eye_landmark[1]), face_roi_landmark.part(eye_landmark[2]))
    center_bottom = mid(face_roi_landmark.part(eye_landmark[5]), face_roi_landmark.part(eye_landmark[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if ver_line_length == 0:
        return float('inf')  # treat as "very open" for your current logic
    return hor_line_length / ver_line_length  # note: inverse of common EAR, matching your original logic


def mouth_aspect_ratio(lips_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(lips_landmark[0]).x, face_roi_landmark.part(lips_landmark[0]).y)
    right_point = (face_roi_landmark.part(lips_landmark[2]).x, face_roi_landmark.part(lips_landmark[2]).y)
    center_top = (face_roi_landmark.part(lips_landmark[1]).x, face_roi_landmark.part(lips_landmark[1]).y)
    center_bottom = (face_roi_landmark.part(lips_landmark[3]).x, face_roi_landmark.part(lips_landmark[3]).y)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if hor_line_length == 0:
        return ver_line_length
    return ver_line_length / hor_line_length


def fore_aspect_ratio(fore_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(fore_landmark[0]).x, face_roi_landmark.part(fore_landmark[0]).y)
    right_point = (face_roi_landmark.part(fore_landmark[2]).x, face_roi_landmark.part(fore_landmark[2]).y)
    center_top = (face_roi_landmark.part(fore_landmark[1]).x, face_roi_landmark.part(fore_landmark[1]).y)
    center_bottom = (face_roi_landmark.part(fore_landmark[4]).x, face_roi_landmark.part(fore_landmark[4]).y)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if hor_line_length == 0:
        return ver_line_length
    return ver_line_length / hor_line_length


def init_twilio():
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    to_num = os.getenv("TWILIO_TO")
    from_num = os.getenv("TWILIO_FROM")
    if sid and token and to_num and from_num:
        try:
            client = Client(sid, token)
            return client, to_num, from_num
        except Exception as e:
            print(f"[twilio] init failed: {e}")
    else:
        print("[twilio] env vars missing; SMS disabled")
    return None, None, None


def main():
    # Init recognition
    sfr = SimpleFacerec()
    sfr.load_encoding_images(IMAGES_DIR)

    # Dlib
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Missing predictor file: {PREDICTOR_PATH}")
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(PREDICTOR_PATH)

    # Audio
    mixer.init()
    danger = mixer.Sound(ALARM_WAV)
    warning = mixer.Sound(DANGER_WAV)

    # Twilio
    tw_client, TW_TO, TW_FROM = init_twilio()
    last_sms_ts = 0.0
    sms_left = MAX_SMS

    # Video
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    count = 0
    fcount = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[video] frame read failed")
                break

            # Resize + flip
            frame = cv2.resize(frame, FRAME_SIZE)
            if SELFIE_FLIP:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face auth
            face_match = sfr.detect_known_faces(frame)
            if face_match:
                cv2.putText(frame, "Driver Authenticated", (380, 575),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                faces = detect(gray)
                for face_roi in faces:
                    landmark_list = predict(gray, face_roi)

                    left_eye_ratio = eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmark_list)
                    right_eye_ratio = eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmark_list)
                    eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

                    left_fore_ratio = fore_aspect_ratio([17, 19, 21, 38, 37, 36], landmark_list)
                    right_fore_ratio = fore_aspect_ratio([22, 24, 26, 45, 44, 43], landmark_list)
                    fore_change_ratio = (left_fore_ratio + right_fore_ratio) / 2.0

                    inner_lip_ratio = mouth_aspect_ratio([60, 62, 64, 66], landmark_list)
                    outer_lip_ratio = mouth_aspect_ratio([48, 51, 54, 57], landmark_list)
                    mouth_open_ratio = (inner_lip_ratio + outer_lip_ratio) / 2.0

                    if ((mouth_open_ratio > MOUTH_OPEN_THRESHOLD and eye_open_ratio > EYE_RATIO_THRESHOLD) or
                        (eye_open_ratio > EYE_RATIO_STRONG) or
                        (fore_change_ratio < FOREHEAD_CHANGE_THRESHOLD)):
                        count += 1
                    else:
                        count = 0

                    x, y = face_roi.left(), face_roi.top()
                    x1, y1 = face_roi.right(), face_roi.bottom()

                    if count > FRAMES_OVER_LIMIT_FOR_WARNING:
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                        cv2.putText(frame, "****************WARNING!****************", (200, 600),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if not mixer.get_busy():
                            warning.play()
                    else:
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            else:
                # Stranger
                fcount += 1
                cv2.putText(frame, "****************STRANGER!!****************", (130, 600),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not mixer.get_busy():
                    danger.play()

                now = time.time()
                if tw_client and (now - last_sms_ts) >= SMS_COOLDOWN_SEC and sms_left > 0:
                    try:
                        tw_client.messages.create(
                            to=TW_TO,
                            from_=TW_FROM,
                            body="Stranger in car. Alert!!"
                        )
                        last_sms_ts = now
                        sms_left -= 1
                        print("[twilio] alert sent")
                    except Exception as e:
                        print(f"[twilio] send failed: {e}")

                cv2.imwrite('stranger.png', frame)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or fcount > MAX_STRANGER_FRAMES:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            mixer.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
