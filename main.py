import cv2
from ultralytics import YOLO
from datetime import datetime
from ultralytics.solutions import object_counter

model = YOLO("./models/yolov8n.pt")
cap = cv2.VideoCapture("./videos/people.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(120, 380), (580, 252)]  # line or region points
classes_to_count = [0,]  # person and car classes for count

# Video writer
video_writer = cv2.VideoWriter("./videos/object_counting_output.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2)

prev_in_counts = 0
prev_out_counts = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    tracks = model.track(im0, persist=True, show=False,
                         classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)

    in_diff = counter.in_counts - prev_in_counts
    out_diff = counter.out_counts - prev_out_counts

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if in_diff > 0:
        with open("report.txt", "a") as file:
            file.write(f"{in_diff} person entered {current_time}\n")
    elif out_diff > 0:
        with open("report.txt", "a") as file:
            file.write(f"{out_diff} person exited {current_time}\n")

    prev_in_counts = counter.in_counts
    prev_out_counts = counter.out_counts

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()