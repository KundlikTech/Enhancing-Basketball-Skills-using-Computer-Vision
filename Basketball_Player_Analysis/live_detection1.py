from tkinter import Tk
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math
from scipy.spatial import distance


class BasketballShot:
    def __init__(self):
        self.cap = None
        self.hsvVals = {'hmin': 0, 'smin': 76, 'vmin': 0, 'hmax': 23, 'smax': 255, 'vmax': 255}
        self.myColorFinder = ColorFinder(False)

        self.posListX = []
        self.posListY = []
        self.listX = list(range(0, 2000))

        self.start = True
        self.prediction = False
        self.coff = None
        self.pred_text = ""
        self.pred_color = (0, 0, 200)

        self.basket_y = 300
        self.basket_x_min = 800
        self.basket_x_max = 1100

        self.roi = None
        self.frame_rate = None
        self.time_interval = None
        self.conversion_factor = 0.01

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)

        # Set camera resolution to 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.time_interval = 1 / self.frame_rate
        return True

    def select_roi(self):
        success, img = self.cap.read()
        if not success:
            print("Failed to read from camera")
            return False
        self.roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return True

    def calculate_speed(self, x1, y1, x2, y2, time_interval):
        dist = distance.euclidean((x1, y1), (x2, y2))
        speed = dist / time_interval * self.conversion_factor
        return speed

    def calculate_angle(self, x1, y1, x2, y2):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle

    def process_video(self):
        frame_count = 0
        while True:
            success, img = self.cap.read()
            if not success:
                break

            frame_count += 1

            # Skip every second frame for speed boost
            if frame_count % 2 != 0:
                cv2.imshow("Live Camera - Basketball Shot Detection", img)
                if cv2.waitKey(1) == 27:  # ESC key
                    break
                continue

            x, y, w, h = self.roi
            imgRoi = img[y:y + h, x:x + w]
            imgResult = imgRoi.copy()

            imgBall, mask = self.myColorFinder.update(imgRoi, self.hsvVals)
            imgCon, contours = cvzone.findContours(img, mask, 200)

            if contours:
                cx, cy = contours[0]['center']
                self.posListX.append(cx)
                self.posListY.append(cy)

            if len(self.posListX) == 10 and self.start:
                self.start = False
                self.coff = np.polyfit(self.posListX, self.posListY, 2)
                a, b, c = self.coff
                c -= self.basket_y
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:
                    x1 = int((-b - math.sqrt(discriminant)) / (2 * a))
                    x2 = int((-b + math.sqrt(discriminant)) / (2 * a))
                    self.prediction = (self.basket_x_min < x1 < self.basket_x_max) or (self.basket_x_min < x2 < self.basket_x_max)
                else:
                    self.prediction = False

                self.pred_text = "Basket" if self.prediction else "No Basket"
                self.pred_color = (0, 200, 0) if self.prediction else (0, 0, 200)

            for i, (posX, posY) in enumerate(zip(self.posListX, self.posListY)):
                cv2.circle(imgResult, (posX, posY), 6, (0, 255, 0), cv2.FILLED)
                if i > 0:
                    cv2.line(imgResult, (self.posListX[i - 1], self.posListY[i - 1]), (posX, posY), (0, 255, 0), 2)

            if len(self.posListX) > 1:
                x1, y1 = self.posListX[-2], self.posListY[-2]
                x2, y2 = self.posListX[-1], self.posListY[-1]
                speed = self.calculate_speed(x1, y1, x2, y2, self.time_interval)
                angle = self.calculate_angle(x1, y1, x2, y2)
                cv2.putText(img, f"{speed:.2f} m/s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img, f"{angle:.1f} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if self.coff is not None:
                for x_val in self.listX[::15]:  # fewer points
                    y_val = int(self.coff[0] * x_val ** 2 + self.coff[1] * x_val + self.coff[2])
                    if 0 <= x_val < w and 0 <= y_val < h:
                        cv2.circle(imgResult, (x_val, y_val), 2, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(imgResult, self.pred_text, (30, 30), colorR=self.pred_color, scale=2, thickness=2, offset=5)

            img[y:y + h, x:x + w] = cv2.resize(imgResult, (w, h))
            cv2.imshow("Live Camera - Basketball Shot Detection", img)
            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    app = BasketballShot()
    if app.initialize_camera() and app.select_roi():
        app.process_video()
