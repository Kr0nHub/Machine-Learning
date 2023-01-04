import cv2
import numpy as np


with open('coco-classes.txt', 'r') as f:
    class_name = f.read().splitlines()

model = cv2.dnn.readNet('yolo/yolov3-spp.weights', 'yolo/yolov3-spp.cfg')

image = "images/persona.jpg"
#image = "images/mustang.jpg"
#image = "images/calle.jpg"
img = cv2.imread(image)
img_name = image.split("/")[-1].split(".")[0]

blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
img_height, img_width, _ = img.shape


model.setInput(blob)
output_layers_names = model.getUnconnectedOutLayersNames()
layerOutputs = model.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * img_width)
            center_y = int(detection[1] * img_height)
            w = int(detection[2] * img_width)
            h = int(detection[3] * img_height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(class_name[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (0, 0, 0), 2)

    title = "detections/" + img_name + '_detection.jpg'
    cv2.imshow(title, img)
    cv2.imwrite(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se detectaron objetos")
    
