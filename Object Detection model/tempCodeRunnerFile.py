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
        self.score_label = Label(text="Shots: 0 | Hits: 0", size_hint=(1, 0.1))

        # Tracking variables
        self.total_shots = 0
        self.successful_shots = 0
        self.cooldown = 0
        self.prediction_started = False
        self.predicted_path = []
        self.prediction_result = ""
        self.prediction_path = []

        layout = BoxLayout(orientation='vertical')
        self.menu_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.2))

        self.live_button = Button(text="Live Prediction")
        self.live_button.bind(on_press=self.start_live_prediction)

        self.video_button = Button(text="Prediction from Video")
        self.video_button.bind(on_press=self.start_video_prediction)

        self.menu_layout.add_widget(self.live_button)
        self.menu_layout.add_widget(self.video_button)

        layout.add_widget(self.menu_layout)
        layout.add_widget(self.img_widget)
        layout.add_widget(self.score_label)

        return layout

    def start_live_prediction(self, instance):
        self.mode = 'real_time'
        self.capture = cv2.VideoCapture(0)
        self.menu_layout.clear_widgets()
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def start_video_prediction(self, instance):
        self.mode = 'video'
        self.menu_layout.clear_widgets()
        self.open_file_chooser()

    def open_file_chooser(self):
        filechooser = FileChooserIconView(filters=['*.mp4', '*.avi', '*.mov'])
        self.popup = Popup(title="Select Video File", content=filechooser, size_hint=(0.9, 0.9))

        def is_valid_file(file_path):
            system_files = ['hiberfil.sys', 'pagefile.sys', 'swapfile.sys', '.sys']
            return not any(sf in file_path.lower() for sf in system_files)

        def on_submit(instance, selection, touch):
            filtered = [f for f in selection if is_valid_file(f)]
            if filtered:
                self.popup.dismiss()
                self.load_video(filtered[0])

        filechooser.bind(on_submit=on_submit)
        self.popup.open()

    def load_video(self, path):
        self.video_path = path
        self.capture = cv2.VideoCapture(self.video_path)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            if self.mode == 'video':
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        results = model.predict(frame, conf=0.4)
        if not results or not results[0].boxes:
            return

        boxes_data = results[0].boxes
        try:
            boxes = boxes_data.xyxy.cpu().numpy()
            labels = boxes_data.cls.cpu().numpy()
        except Exception:
            return

        ball, hoop = None, None
        trigger_radius = 400
        ball_center_x = ball_center_y = hoop_center_x = hoop_center_y = None

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            label = int(labels[i])

            if label == 0:
                ball_center_x = int((xmin + xmax) / 2)
                ball_center_y = int((ymin + ymax) / 2)
                ball = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            elif label == 1:
                hoop_center_x = int((xmin + xmax) / 2)
                hoop_center_y = int((ymin + ymax) / 2)
                hoop = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}

        if ball:
            self.prediction_path.append((ball_center_x, ball_center_y))
            if len(self.prediction_path) > 30:
                self.prediction_path.pop(0)

        if ball and hoop:
            cv2.circle(frame, (hoop_center_x, hoop_center_y), trigger_radius, (255, 0, 255), 2)
            distance = np.sqrt((ball_center_x - hoop_center_x) ** 2 + (ball_center_y - hoop_center_y) ** 2)

            if distance <= trigger_radius and not self.prediction_started and len(self.prediction_path) >= 3:
                self.prediction_started = True
                self.total_shots += 1

                pts = np.array(self.prediction_path[:10])
                X = pts[:, 0].reshape(-1, 1)
                y = pts[:, 1]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model_poly = LinearRegression()
                model_poly.fit(X_poly, y)

                pred_pts = []
                hit = False

                for dx in range(0, 600, 10):
                    pred_x = ball_center_x + dx
                    pred_y = model_poly.predict(poly.transform([[pred_x]]))[0]
                    pred_pt = (int(pred_x), int(pred_y))
                    pred_pts.append(pred_pt)

                    dist_to_hoop = np.sqrt((pred_x - hoop_center_x) ** 2 + (pred_y - hoop_center_y) ** 2)
                    if dist_to_hoop <= 10:
                        hit = True
                        break
                    if dist_to_hoop >= trigger_radius:
                        break

                self.predicted_path = pred_pts
                self.prediction_result = "Basket!" if hit else "No Basket"
                if hit:
                    self.successful_shots += 1

        if self.cooldown > 0:
            self.cooldown -= 1
        elif self.prediction_started:
            self.prediction_started = False
            self.predicted_path = []
            self.prediction_result = ""
            self.prediction_path = []
            self.cooldown = 30

        # Draw detection boxes
        if ball:
            cv2.rectangle(frame, (int(ball['xmin']), int(ball['ymin'])),
                          (int(ball['xmax']), int(ball['ymax'])), (0, 255, 255), 2)
            cv2.putText(frame, "Basketball", (int(ball['xmin']), int(ball['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if hoop:
            cv2.rectangle(frame, (int(hoop['xmin']), int(hoop['ymin'])),
                          (int(hoop['xmax']), int(hoop['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, "Hoop", (int(hoop['xmin']), int(hoop['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw prediction curve
        for i in range(1, len(self.predicted_path)):
            cv2.line(frame, self.predicted_path[i - 1], self.predicted_path[i], (255, 0, 0), 2)
        if self.predicted_path:
            cv2.circle(frame, self.predicted_path[-1], 5, (0, 0, 255), -1)

        # Show result
        if self.prediction_result:
            cv2.putText(frame, self.prediction_result, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        self.score_label.text = f"Shots: {self.total_shots} | Hits: {self.successful_shots}"

        # Convert and display image
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
