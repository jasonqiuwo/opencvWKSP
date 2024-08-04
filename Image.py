import cv2
import numpy as np

# Load YOLOv3 weights and configuration
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
coco_names_path = 'coco.names'

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO class labels
with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
image_path = '4.jpg'  # Change this to the path of your image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Perform forward pass
outs = net.forward(output_layers)

# Process detections
class_ids = []
confidences = []
boxes = []

for i, out in enumerate(outs):
    print(f"Processing output layer {i} with shape: {out.shape}")

    # Each output should be a 2D array of shape [num_detections, num_values]
    if len(out.shape) == 2:
        num_detections, num_values = out.shape
        print(f"Number of detections: {num_detections}, Number of values per detection: {num_values}")

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
            print("Unexpected number of values per detection:", num_values)
    else:
        print("Unexpected output shape:", out.shape)

# Apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw bounding boxes and labels on image
if len(indices) > 0:
    indices = indices.flatten()  # Flatten indices to 1D array
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and show output image
cv2.imwrite('output_image.jpg', image)
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
