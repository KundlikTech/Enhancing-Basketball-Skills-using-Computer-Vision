from ultralytics import YOLO
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load custom-trained model (basketball and hoop detection)
model = YOLO('basketballHoop.pt')


class ShotPredictorApp(App):
    def build(self):
        self.capture = None
        self.video_path = None
        self.mode = None
        self.img_widget = KivyImage()
        self.score_label = Label(text="Hits: 0", size_hint=(1, 0.1))

        # Tracking variables for video prediction (using the original 'update' method)
        self.total_shots = 0
        self.successful_shots = 0
        self.cooldown = 0
        self.prediction_started = False
        self.predicted_path = []
        self.prediction_result = ""
        self.prediction_path = []
        self.drawing_prediction = []
        self.hoop_center_history = []
        self.shot_initiated = False

        # Tracking variables for live counting
        self.live_hits = 0
        self.hoop_center_live = None
        self.ball_center_live = None
        self.live_cooldown = 0 # Cooldown for live hit counting

        layout = BoxLayout(orientation='vertical')
        self.menu_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.2))

        self.live_button = Button(text="Live Basketball Hit Counting")
        self.live_button.bind(on_press=self.start_live_counting)

        self.video_button = Button(text="Prediction from Video")
        self.video_button.bind(on_press=self.start_video_prediction)

        self.menu_layout.add_widget(self.live_button)
        self.menu_layout.add_widget(self.video_button)

        layout.add_widget(self.menu_layout)
        layout.add_widget(self.img_widget)
        layout.add_widget(self.score_label)

        return layout

    def start_live_counting(self, instance):
        self.mode = 'live_counting'
        self.capture = cv2.VideoCapture(0)
        self.menu_layout.clear_widgets()
        self.live_hits = 0
        self.score_label.text = "Hits: 0"
        Clock.schedule_interval(self.update_live, 1.0 / 30.0)

    def start_video_prediction(self, instance):
        self.mode = 'video'
        self.capture = None # Reset capture in case live was used before
        self.menu_layout.clear_widgets()
        self.reset_video_tracking()
        self.open_file_chooser()

    def reset_video_tracking(self):
        self.total_shots = 0
        self.successful_shots = 0
        self.cooldown = 0
        self.prediction_started = False
        self.predicted_path = []
        self.prediction_result = ""
        self.prediction_path = []
        self.drawing_prediction = []
        self.hoop_center_history = []
        self.shot_initiated = False
        self.score_label.text = "Shots: 0 | Hits: 0" # Reset score label for video

    def open_file_chooser(self):
        filechooser = FileChooserIconView(filters=['*.mp4', '*.avi', '*.mov'])
        self.popup = Popup(title="Select Video File", content=filechooser, size_hint=(0.9, 0.9))

        def on_submit(instance, selection, touch):
            if selection:
                self.popup.dismiss()
                self.load_video(selection[0])

        filechooser.bind(on_submit=on_submit)
        self.popup.open()

    def load_video(self, path):
        self.video_path = path
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            print(f"Error: Could not open video file at {path}")
            self.video_path = None
            return
        self.reset_video_tracking()
        Clock.schedule_interval(self.update, 1.0 / 30.0) # Use the original 'update' for video

    def update_live(self, dt):
        if self.capture is None or self.mode != 'live_counting':
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        results = model.predict(frame, conf=0.6) # Higher confidence for direct matching
        if not results or not results[0].boxes:
            self.hoop_center_live = None
            self.ball_center_live = None
            self.update_frame(frame)
            return

        boxes_data = results[0].boxes
        ball_live = None
        hoop_live = None

        try:
            boxes = boxes_data.xyxy.cpu().numpy()
            labels = boxes_data.cls.cpu().numpy()
        except Exception:
            self.update_frame(frame)
            return

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            label = int(labels[i])
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            if label == 0:
                ball_live = {'cx': center_x, 'cy': center_y, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
                self.ball_center_live = (center_x, center_y)
            elif label == 1:
                hoop_live = {'cx': center_x, 'cy': center_y, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
                self.hoop_center_live = (center_x, center_y)

        if ball_live and hoop_live and self.hoop_center_live and self.ball_center_live and self.live_cooldown == 0:
            ball_x, ball_y = self.ball_center_live
            hoop_x, hoop_y = self.hoop_center_live

            # Check if the center of the ball is very close to the center of the hoop
            distance = np.sqrt((ball_x - hoop_x) ** 2 + (ball_y - hoop_y) ** 2)
            hit_threshold = 30 # Adjust as needed

            if distance <= hit_threshold:
                self.live_hits += 1
                self.score_label.text = f"Hits: {self.live_hits}"
                self.live_cooldown = 30 # Start cooldown (adjust as needed)

        if self.live_cooldown > 0:
            self.live_cooldown -= 1

        # Draw detections for live mode
        if ball_live:
            cv2.rectangle(frame, (int(ball_live['xmin']), int(ball_live['ymin'])),
                          (int(ball_live['xmax']), int(ball_live['ymax'])), (0, 255, 255), 2)
            cv2.putText(frame, "Basketball", (int(ball_live['xmin']), int(ball_live['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (ball_live['cx'], ball_live['cy']), 5, (255, 255, 0), -1)

        if hoop_live:
            cv2.rectangle(frame, (int(hoop_live['xmin']), int(hoop_live['ymin'])),
                          (int(hoop_live['xmax']), int(hoop_live['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, "Hoop", (int(hoop_live['xmin']), int(hoop_live['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (hoop_live['cx'], hoop_live['cy']), 5, (0, 0, 255), -1)

        self.update_frame(frame)

    def update(self, dt):
        if self.capture is None or self.mode != 'video':
            return

        ret, frame = self.capture.read()
        if not ret:
            if self.mode == 'video':
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        results = model.predict(frame, conf=0.4)
        if not results or not results[0].boxes:
            self.update_frame(frame)
            return

        boxes_data = results[0].boxes
        try:
            boxes = boxes_data.xyxy.cpu().numpy()
            labels = boxes_data.cls.cpu().numpy()
        except Exception:
            self.update_frame(frame)
            return

        ball, hoop = None, None
        trigger_radius = 600
        ball_center_x = ball_center_y = hoop_center_x = hoop_center_y = None

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            label = int(labels[i])

            if label == 0:
                ball_center_x = int((xmin + xmax) / 2)
                ball_center_y = int((ymin + ymax) / 2)
                ball = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            elif label == 1:
                current_hoop_center_x = int((xmin + xmax) / 2)
                current_hoop_center_y = int((ymin + ymax) / 2)
                self.hoop_center_history.append((current_hoop_center_x, current_hoop_center_y))
                if len(self.hoop_center_history) > 10:
                    self.hoop_center_history.pop(0)
                hoop_center_x = int(np.mean([h[0] for h in self.hoop_center_history]))
                hoop_center_y = int(np.mean([h[1] for h in self.hoop_center_history]))
                hoop = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                        'cx': current_hoop_center_x, 'cy': current_hoop_center_y}

        if ball:
            self.prediction_path.append((ball_center_x, ball_center_y))
            if len(self.prediction_path) > 30:
                self.prediction_path.pop(0)

        if ball and hoop:
            cv2.circle(frame, (hoop['cx'], hoop['cy']), trigger_radius, (255, 0, 255), 2)
            distance = np.sqrt((ball_center_x - hoop['cx']) ** 2 + (ball_center_y - hoop['cy']) ** 2)

            if distance <= trigger_radius and len(self.prediction_path) >= 5:
                pts = np.array(self.prediction_path[-5:])
                X = pts[:, 0].reshape(-1, 1)
                y = pts[:, 1]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model_poly = LinearRegression()
                model_poly.fit(X_poly, y)

                self.drawing_prediction = []
                num_points = 50
                x_predictions = np.linspace(ball_center_x, hoop['cx'], num_points)
                y_predictions = model_poly.predict(poly.transform(x_predictions.reshape(-1, 1)))

                for i in range(num_points):
                    self.drawing_prediction.append((int(x_predictions[i]), int(y_predictions[i])))

                predicted_hit = False
                hoop_radius_check = 25
                if len(self.drawing_prediction) > 0:
                    last_pred_x, last_pred_y = self.drawing_prediction[-1]
                    dist_to_hoop_pred = np.sqrt((last_pred_x - hoop['cx']) ** 2 + (last_pred_y - hoop['cy']) ** 2)
                    if dist_to_hoop_pred <= hoop_radius_check and last_pred_x > ball_center_x:
                        predicted_hit = True

                self.prediction_result = "Likely Basket" if predicted_hit else "Unlikely Basket"
            else:
                self.drawing_prediction = []
                self.prediction_result = ""

            if distance <= trigger_radius and not self.prediction_started and len(self.prediction_path) >= 15:
                self.prediction_started = True
                self.total_shots += 1
                self.cooldown = 30

                pts = np.array(self.prediction_path[-15:])
                X = pts[:, 0].reshape(-1, 1)
                y = pts[:, 1]
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                model_poly = LinearRegression()
                model_poly.fit(X_poly, y)

                self.predicted_path = []
                hit = False
                last_pred_y = ball_center_y

                for dx in range(0, 800, 10):
                    pred_x = ball_center_x + dx
                    pred_y = model_poly.predict(poly.transform([[pred_x]]))[0]
                    pred_pt = (int(pred_x), int(pred_y))
                    self.predicted_path.append(pred_pt)

                    dist_to_hoop = np.sqrt((pred_x - hoop['cx']) ** 2 + (pred_y - hoop['cy']) ** 2)
                    if dist_to_hoop <= 15:
                        hit = True
                        break
                    if pred_y > last_pred_y and len(self.predicted_path) > 5:
                        break
                    last_pred_y = pred_y
                    if pred_x > hoop['cx'] + 300:
                        break

                if hit:
                    self.successful_shots += 1
            elif self.prediction_started and self.cooldown == 0:
                self.prediction_started = False
                self.predicted_path = []
                self.prediction_result = ""

        if self.cooldown > 0:
            self.cooldown -= 1

        # Draw detections and predictions
        if ball:
            cv2.rectangle(frame, (int(ball['xmin']), int(ball['ymin'])),
                          (int(ball['xmax']), int(ball['ymax'])), (0, 255, 255), 2)
            cv2.putText(frame, "Basketball", (int(ball['xmin']), int(ball['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (ball_center_x, ball_center_y), 5, (255, 255, 0), -1)

        if hoop:
            cv2.rectangle(frame, (int(hoop['xmin']), int(hoop['ymin'])),
                          (int(hoop['xmax']), int(hoop['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, "Hoop", (int(hoop['xmin']), int(hoop['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (hoop['cx'], hoop['cy']), 5, (0, 0, 255), -1)

        for i in range(1, len(self.drawing_prediction)):
            cv2.line(frame, self.drawing_prediction[i - 1], self.drawing_prediction[i], (255, 165, 0), 2)
        if self.drawing_prediction:
            cv2.circle(frame, self.drawing_prediction[-1], 5, (0, 165, 255), -1)

        for i in range(1, len(self.predicted_path)):
            cv2.line(frame, self.predicted_path[i - 1], self.predicted_path[i], (255, 0, 0), 2)
        if self.predicted_path:
            cv2.circle(frame, self.predicted_path[-1], 5, (0, 0, 255), -1)

        if self.prediction_result:
            cv2.putText(frame, self.prediction_result, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        self.score_label.text = f"Shots: {self.total_shots} | Hits: {self.successful_shots}"

        self.update_frame(frame)

    def update_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(frame_rgb, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def on_stop(self):
        if self.capture:
            self.capture.release()


if __name__ == '__main__':
    ShotPredictorApp().run()