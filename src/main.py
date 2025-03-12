

"""
Object detection system by Ibrahim Abubakar Buba
"""

import cv2
import threading
import numpy as np
from ultralytics import YOLO

from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.metrics import dp

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.toolbar import MDTopAppBar  # Updated import
from kivymd.uix.card import MDCard


class VideoThread(threading.Thread):
    def __init__(self, model_path, update_callback):
        super(VideoThread, self).__init__()
        self.model = YOLO(model_path)
        self.update_callback = update_callback
        self.running = False
        self.cap = None
        # Define class names (COCO dataset)
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
            "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair dryer", "toothbrush"
        ]

    def run(self):
        self.cap = cv2.VideoCapture(0)
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Perform object detection using YOLO
            results = self.model(frame)
            # Loop through each detected box and draw it
            for box in results[0].boxes:
                # Retrieve bounding box coordinates, confidence, and class index
                xyxy = box.xyxy.cpu().numpy().flatten()  # Should be a 1D array of 4 values
                conf = box.conf.cpu().numpy().item()       # Convert to a scalar
                cls = box.cls.cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                label = self.class_names[int(cls)]
                color = (0, 255, 0)  # Green color for the box

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf*100:.2f}%"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert frame from BGR to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use the callback to update the UI (scheduling it on the main thread)
            self.update_callback(frame)

        self.cap.release()


class MainApp(MDApp):
    def build(self):
        # Apply a dark theme and a primary palette
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

        # Main layout using MDBoxLayout with vertical orientation
        self.layout = MDBoxLayout(orientation='vertical', spacing=10)

        # Use MDTopAppBar for the app title at the top
        top_app_bar = MDTopAppBar(
            title="Object Detection System by Ibrahim Abubakar Buba",
            elevation=10
        )
        self.layout.add_widget(top_app_bar)

        # Image widget for displaying video frames inside a card for a better look
        self.image_widget = Image()
        video_card = MDCard(
            size_hint=(1, 1),
            radius=[15],
            elevation=5,
            padding=10,
        )
        video_card.add_widget(self.image_widget)
        self.layout.add_widget(video_card)

        # Button layout at the bottom
        button_layout = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=10,
            padding=10
        )

        # Start and Stop buttons with callbacks
        start_button = MDRaisedButton(text="Start", on_release=self.start_detection)
        stop_button = MDRaisedButton(text="Stop", on_release=self.stop_detection)

        button_layout.add_widget(start_button)
        button_layout.add_widget(stop_button)
        self.layout.add_widget(button_layout)

        self.video_thread = None

        return self.layout

    def start_detection(self, instance):
        """ Start video capture if not already running """
        if self.video_thread is None or not self.video_thread.running:
            # Ensure the model weights file path is correct (adjust if needed)
            self.video_thread = VideoThread('weights/yolov10n.pt', self.update_image)
            self.video_thread.start()

    def stop_detection(self, instance):
        """ Stop the video capture thread gracefully """
        if self.video_thread and self.video_thread.running:
            self.video_thread.running = False
            self.video_thread.join()

    def update_image(self, frame):
        """ Schedule the update of the image widget on the main thread """
        Clock.schedule_once(lambda dt: self.display_frame(frame), 0)

    def display_frame(self, frame):
        """ Convert the numpy array (frame) into a Kivy texture and display it """
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        self.image_widget.texture = texture
        self.image_widget.canvas.ask_update()


if __name__ == '__main__':
    MainApp().run()
