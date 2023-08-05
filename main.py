# import confidence as confidence
import cv2
import numpy as np

net = cv2.dnn.readNet('yolov4.weights path', 'yolov4.cfg path')
classes = []
with open('obj.names') as f:
    classes = f.read().splitlines()


img = cv2.imread('image.jpg path')


while True:
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    #
    # # showing the output after the blob thing:
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    net.setInput(blob)  # set the input from blob into the Net

    output_layers_names = net.getUnconnectedOutLayersNames()  # to get the Output layers names
    layerOutputs = net.forward(output_layers_names)  # pass the output layers names into net.forward function
    predictions = net.forward()

    boxes = []
    confidences = []
    class_ids = []
    #
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x1 = int(center_x - w * 0.5)  # Start X coordinate
                y1 = int(center_y - h * 0.5)  # Start Y coordinate
                x2 = int(center_x + w * 0.5)  # End X coordinate
                y2 = int(center_y + h * 0.5)  # End y coordinate
                #
                boxes.append([x1, y1, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

        # print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes.flatten())
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = np.random.uniform(120, 250, size=(len(boxes), 3))
    #
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
    cv2.putText(img, "License Plate", (x1, y1 + 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 120), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(0)  
    

cv2.destroyAllWindows()
