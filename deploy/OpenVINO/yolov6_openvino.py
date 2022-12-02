import cv2
import numpy as np
import argparse
import os
from openvino.runtime import Core

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5			# cls score
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45		# obj confidence

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(image, model, weights):
    """Load the model"""
    
    ie = Core()
    
    model = ie.read_model(model = model, weights = weights)
    compiled_model = ie.compile_model(model = model, device_name = "CPU")
    
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)
    
    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = input_layer_ir.shape
    
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(image, (W, H))
    
    # Reshape to the network input shape.
    input_image = np.expand_dims(
        resized_image.transpose(2, 0, 1), 0
    )
    
    # Create an inference request.
    output = compiled_model([input_image])[output_layer_ir]
    
    return output

def post_process(input_image, outputs):
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []

	# Rows.
	rows = outputs.shape[1]

	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width / INPUT_WIDTH
	y_factor =  image_height / INPUT_HEIGHT

	# Iterate through 25200 detections.
	for r in range(rows):
		row = outputs[0][r]
		confidence = row[4]

		# Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# Get the index of max class score.
			class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)

				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left, top)

	return input_image

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/yolov6s.xml', help="Input your xml model.")
    parser.add_argument('--weights', default='models/yolov6s.bin', help="Input your weights.")
    parser.add_argument('--img', default='sample.jpg', help="Path to your input image.")
    parser.add_argument('--classesFile', default='coco.names', help="Path to your classesFile.")
    args = parser.parse_args()
    
    # Load class names.
    model_path, weights_path, img_path, classesFile = args.model, args.weights , args.img, args.classesFile
    window_name = os.path.splitext(os.path.basename(model_path))[0]
    
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    
    # Load image.
    image = cv2.imread(img_path)
    input = image.copy()
    
    #Detect objects.
    detections = pre_process(input.copy(), model_path, weights_path)
    img = post_process(image.copy(), detections)
    
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    
    
    
    

    

