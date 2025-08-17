import argparse
import os

import cv2
import mediapipe as mp


def blur_image(passed_image, face_detector):
    image_rgb = cv2.cvtColor(passed_image, cv2.COLOR_BGR2RGB)

    output = face_detector.process(image_rgb)

    if output.detections is not None:
        for detection in output.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # image = cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            passed_image[y1:y1 + h, x1:x1 + w, :] = cv2.blur(passed_image[y1:y1 + h, x1:x1 + w, :], (25, 25))
    return passed_image


args = argparse.ArgumentParser()

args.add_argument("--mode", default='image')
args.add_argument("--filepath", default='./data/testImage.jpg')

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Detect Face
mp_face_detection = mp.solutions.face_detection

# print(face_detection)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    if args.mode in ["image"]:
        image = cv2.imread(args.filepath)

        H, W, _ = image.shape
        image = blur_image(image, face_detection)

        cv2.imwrite(os.path.join(output_dir, 'output(image).png'), image)

cv2.imshow('Face Detection', image)
cv2.waitKey(0)
