import cv2

input_video = "traffic_video.mp4"
output_video = "small_video.mp4"

# seconds of video you want
duration_seconds = 10

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(fps * duration_seconds)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret or frame_count >= total_frames:
        break

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print("✅ Small test video created:", output_video)