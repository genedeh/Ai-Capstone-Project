import os

import cv2
import mediapipe as mp


class FaceBlurrer:
    def __init__(self, output_dir="./output", blur_strength=50):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.blur_strength = blur_strength
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=1
        )

    def _blur_frame(self, frame):
        """Apply face blur to a single frame."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        output = self.face_detector.process(image_rgb)

        if output.detections:
            for detection in output.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * width)
                y1 = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x1 + w)
                y2 = min(height, y1 + h)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    frame[y1:y2, x1:x2] = cv2.blur(face_roi, (self.blur_strength, self.blur_strength))

        return frame

    def blur_image_file(self, filepath):
        """Blur all faces in an image."""
        image = cv2.imread(filepath)
        result = self._blur_frame(image)
        out_path = os.path.join(self.output_dir, "blurred_image.png")
        cv2.imwrite(out_path, result)
        return out_path, result

    def blur_video_file(self, filepath):
        """Blur all faces in a video."""
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            raise ValueError("Could not read video")

        out_path = os.path.join(self.output_dir, "blurred_video.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"MP4V"),
            fps,
            (frame.shape[1], frame.shape[0])
        )

        while ret:
            frame = self._blur_frame(frame)
            writer.write(frame)
            ret, frame = cap.read()

        cap.release()
        writer.release()
        return out_path

    def blur_webcam(self):
        """Stream webcam with blurred faces (no saving)."""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._blur_frame(frame)
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # yield for streaming
        cap.release()
