import cv2 # Download opencv-python in the python packages 
import numpy as np # Download numpy as well 

# Load YOLOv3 weights and configuration
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
coco_names_path = 'coco.names'

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO class labels
with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set up webcam capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# Define the processing parameters
scale_factor = 1/255.0
size = (416, 416)
swap_rb = True
crop = False

# Get output layer names
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale_factor, size=size, swapRB=swap_rb, crop=crop)
    net.setInput(blob)

    # Perform forward pass
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        # Print the shape of the output layer
        print(f"Output layer shape: {out.shape}")

        # Check if output is a 2D array or 3D array
        if len(out.shape) == 2:
            num_detections, num_values = out.shape
            if num_values == 85:
                for det in out:
                    # Extract values from the detection array
                    x_center = det[0]
                    y_center = det[1]
                    w = det[2]
                    h = det[3]
                    objectness = det[4]
                    scores = det[5:]

                    # Convert to image coordinates
                    center_x = int(x_center * width)
                    center_y = int(y_center * height)
                    w = int(w * width)
                    h = int(h * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Filter based on objectness and confidence
                    if objectness > 0.5:
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:  # Confidence threshold
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            else:
                print(f"Unexpected number of values per detection: {num_values}")
                
        elif len(out.shape) == 3:
            for detection in out:
                if detection.shape[1] == 85:
                    for det in detection:
                        # Extract values from the detection array
                        x_center = det[0]
                        y_center = det[1]
                        w = det[2]
                        h = det[3]
                        objectness = det[4]
                        scores = det[5:]

                        # Convert to image coordinates
                        center_x = int(x_center * width)
                        center_y = int(y_center * height)
                        w = int(w * width)
                        h = int(h * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Filter based on objectness and confidence
                        if objectness > 0.5:
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:  # Confidence threshold
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                else:
                    print(f"Unexpected number of values per detection: {detection.shape[1]}")
        else:
            print(f"Unexpected output shape: {out.shape}")

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw bounding boxes and labels on the frame
    if len(indices) > 0:
        indices = indices.flatten()  # Flatten indices to 1D array
        for i in indices:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Real-Time Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
