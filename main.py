import cv2
from ultralytics import YOLO
from datetime import datetime
from centroidtracker import CentroidTracker

model = YOLO('./models/yolov8s.pt')

ct = CentroidTracker(maxDisappeared=10)

entry_exit_line = (120, 380), (580, 252)

cap = cv2.VideoCapture('./videos/people.mp4')

def determine_crossing(prev_point, curr_point, entry_exit_line):
    (x1, y1), (x2, y2) = entry_exit_line
    
    side_prev = (x2 - x1) * (prev_point[1] - y1) - (y2 - y1) * (prev_point[0] - x1)
    side_curr = (x2 - x1) * (curr_point[1] - y1) - (y2 - y1) * (curr_point[0] - x1)

    if side_prev * side_curr < 0:
        if side_curr < 0:
            return "entered"
        else:
            return "exited"
    else:
        return None

def is_crossed_entry_exit_line(prev_point, curr_point, entry_exit_line):
    (x1, y1), (x2, y2) = entry_exit_line
    
    if x1 == x2:
        if (prev_point[0] < x1 and curr_point[0] > x1) or \
           (prev_point[0] > x1 and curr_point[0] < x1):
            if min(prev_point[1], curr_point[1]) < max(y1, y2) and max(prev_point[1], curr_point[1]) > min(y1, y2):
                return determine_crossing(prev_point, curr_point, entry_exit_line)
        return None
    else:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        prev_y_on_line = m * prev_point[0] + c
        curr_y_on_line = m * curr_point[0] + c

        if (prev_point[1] < prev_y_on_line and curr_point[1] > curr_y_on_line) or \
           (prev_point[1] > prev_y_on_line and curr_point[1] < curr_y_on_line):
            if min(prev_point[0], curr_point[0]) < max(x1, x2) and max(prev_point[0], curr_point[0]) > min(x1, x2):
                return determine_crossing(prev_point, curr_point, entry_exit_line)
        return None

def is_not_on_line(point, line):
    (x1, y1), (x2, y2) = line
    x, y = point[0], point[1]
    return abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) > 1

prev_centroids = {}

entry_count = 0
exit_count = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
ret, frame = cap.read()
out = cv2.VideoWriter('./videos/output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = model.predict(frame)

    cv2.line(frame, entry_exit_line[0], entry_exit_line[1], (0, 140, 255), 3)

    result_boxes = result[0].boxes.xyxy

    rects = []

    for i in range(len(result_boxes)):
        is_person = result[0].boxes.cls[i] == 0
        if is_person and result[0].boxes.conf[i] > 0.3:
            box = result_boxes[i]
            startX, startY, endX, endY = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            rects.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (int((startX + endX) / 2), int((startY + endY) / 2)), 5, (0, 0, 255), -1)

    objects = ct.update(rects)

    for objectID, centroid in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        if objectID in prev_centroids:
            prev_centroid = prev_centroids[objectID]
            curr_centroid = centroid

            result = is_crossed_entry_exit_line(prev_centroid, curr_centroid, entry_exit_line)
            if result is not None:
                with open('report.txt', 'a') as f:
                    time = datetime.now()
                    f.write(f"a person {result} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if result == "entered":
                        entry_count += 1
                    elif result == "exited":
                        exit_count += 1

        if is_not_on_line(centroid, entry_exit_line):
            prev_centroids[objectID] = centroid

    cv2.rectangle(frame, (750, 300), (950, 370), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, f"Entry: {entry_count}", (750, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    cv2.rectangle(frame, (1000, 300), (1170, 370), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, f"Exit: {exit_count}", (1000, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()