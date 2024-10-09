import cv2
from ultralytics import YOLO

# Load the trained model
best_model = YOLO('best.pt')

# Open the video file
video_path = 'drones1.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Define the codec and create a VideoWriter object to save the output video
output_path = 'out.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the detection on the frame
    results = best_model.predict(source=frame, conf=0.5)

    # Extract bounding boxes and class labels
    for box in results[0].boxes:
        # Get the coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Get the class name
        class_name = best_model.names[int(box.cls[0])]
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put the class name text
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the frame with detections (optional)
    cv2.imshow('Video', frame)  # Use cv2.imshow for local display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Detection completed. Output saved to {output_path}")
