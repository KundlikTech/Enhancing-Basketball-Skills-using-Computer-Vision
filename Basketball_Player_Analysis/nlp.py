import cv2
import ffmpeg
from ultralytics import YOLO
import os

# Paths
video_path = "1.mp4"  # Input video
output_video_path = "output_basketball_video.mp4"  # Final output video
audio_files = {
    "dribbling": "dribble.mp3",
    "shooting": "dribble.mp3",
    "dunking": "dribble.mp3",
    "passing": "dribble.mp3",
}

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with trained model for basketball actions

# Detect actions in the video
def detect_actions(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    actions_timestamps = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

        for result in results:
            for box in result.boxes:
                label_id = int(box.cls)
                label = model.names.get(label_id, "unknown")
                timestamp = frame_idx / fps

                if label in audio_files:
                    actions_timestamps.append((label, timestamp))

        frame_idx += 1

    cap.release()
    return actions_timestamps

# Add audio overlays at action timestamps
def add_audio_to_video(video_path, actions_timestamps, audio_files, output_path):
    temp_audio_path = "temp_audio_track.mp3"
    audio_inputs = []

    for label, timestamp in actions_timestamps:
        if label in audio_files:
            audio_inputs.append(
                ffmpeg.input(audio_files[label], ss=0, t=1)
                .filter("adelay", f"{int(timestamp*1000)}|{int(timestamp*1000)}")
            )

    if not audio_inputs:
        print("⚠️ No matching actions detected — skipping audio overlay.")
        ffmpeg.input(video_path).output(output_path, vcodec="copy", acodec="copy").run(overwrite_output=True)
        return

    merged_audio = ffmpeg.concat(*audio_inputs, v=0, a=1)
    merged_audio.output(temp_audio_path).run(overwrite_output=True)

    ffmpeg.input(video_path).output(output_path, audio=temp_audio_path, vcodec="copy").run(overwrite_output=True)

    os.remove(temp_audio_path)

# Run the pipeline
actions_timestamps = detect_actions(video_path)

# Print detected actions for debug
print("\n🎯 Detected Actions and Timestamps:")
if actions_timestamps:
    for label, ts in actions_timestamps:
        print(f" - {label} at {ts:.2f} sec")
else:
    print("No actions detected.")

add_audio_to_video(video_path, actions_timestamps, audio_files, output_video_path)

print(f"\n✅ Output video saved at: {output_video_path}")
