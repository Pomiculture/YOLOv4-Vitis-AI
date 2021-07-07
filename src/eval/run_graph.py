# - Source : https://github.com/Xilinx/Vitis-Tutorials/blob/ce44f7a1667cfacd7b329fa77a454fce1e0450b3/Machine_Learning/Design_Tutorials/07-yolov4-tutorial/scripts/tf_eval_yolov4_coco_2017.py#L286

# - Source 2 : https://towardsdatascience.com/yolo2-walkthrough-with-examples-e40452ca265f

# - Source 3 : https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

###############################################################################################################

import os
import argparse
import time

import cv2
import numpy as np
import tensorflow as tf

import tensorflow.contrib.decent_q

###############################################################################################################

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################################################

# Global variables
start_t = 0.0
finish_t = 0.0
delta_t = 0.0

###############################################################################################################

def correct_boxes(box_xy, box_wh, input_shape, image_shape):
        """
        Get the delimitation of the bounding boxes and scale them to fit the image.
        """
	# Invert data order
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
	# Cast tensor input and image input data type to float 
        input_shape = tf.cast(input_shape, dtype = tf.float32)
        image_shape = tf.cast(image_shape, dtype = tf.float32)
	# Compute the scaling factor 
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
	# Get top-left box corner
        box_mins = box_yx - (box_hw / 2.)
	# Get bottom-right box corner
        box_maxes = box_yx + (box_hw / 2.)
	# Concatenate bboxes and set them to image scale
        boxes = tf.concat([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis = -1)
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        return boxes


def get_feats(feats, anchors, num_classes, input_shape):
        """
        Get the x,y,width,height of each detected bbox, and the related detection and classification scores.
        """
	# Get number of anchors
        num_anchors = len(anchors)
	# Create anchors tensor
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
	# Set grid size
        grid_size = tf.shape(feats)[1:3]
	# Get predictions
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # Create grid for bboxes areas
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)
        grid = tf.cast(grid, tf.float32)
        # Normalization of the x and y coordinates of the bboxes 
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[...,::-1], tf.float32)
        # Normalization of the width and height of the bboxes 
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[...,::-1], tf.float32)
	# Get object detection confidence
        box_confidence = tf.sigmoid(predictions[..., 4:5])
	# Get classification probabilities
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs


def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
        """
        Get bounding boxes coordinates and corresponding scores
        """
	# Retrieve the bboxes data, and the detection and classification scores associated
        box_xy, box_wh, box_confidence, box_class_probs = get_feats(feats, anchors, classes_num, input_shape)
        # Process bboxes dimensions
        boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
	# Reshape bboxes
        boxes = tf.reshape(boxes, [-1, 4])
	# Compute confidence score
        box_scores = box_confidence * box_class_probs	
	# Reshape confidence score
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores


def predict(yolo_outputs, image_shape, anchors, class_names, obj_threshold, nms_threshold, max_boxes = 1000):
	"""
	Process the results of the Yolo inference to retrieve the detected bounding boxes,
	the corresponding class label, and the confidence score associated.
	The threshold value 'obj_threshold' serves to discard low confidence predictions.
	The 'nms_threshold' value is used to discard duplicate boxes for a same object (IoU metric).
	"""
	# Init
	anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	total_boxes = []
	total_box_scores = []
	input_shape = tf.shape(yolo_outputs[0])[1 : 3] * 32                      

	# Process output tensors
	for i in range(len(yolo_outputs)):
		# Get bboxes and associated scores
		detected_boxes, box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
		# Append bboxes and level of confidence to list
		total_boxes.append(detected_boxes)
		total_box_scores.append(box_scores)

	# Concatenate results
	total_boxes = tf.concat(total_boxes, axis=0)
	total_box_scores = tf.concat(total_box_scores, axis=0)
	
	#print('------------------------------------')
	#print('Boxe scores', box_scores)

        # Mask to filter out low confidence detections
	mask = box_scores >= obj_threshold
	# Set boxes limit
	max_boxes_tensor = tf.constant(max_boxes, dtype = tf.int32)
	boxes_ = []
	scores_ = []
	classes_ = []
	items_ = []
	for c in range(len(class_names)):
		# Get boxes labels
		class_boxes = tf.boolean_mask(total_boxes, mask[:, c])
		# Get associated score
		class_box_scores = tf.boolean_mask(total_box_scores[:, c], mask[:, c])
		# Concatenate label and score
		item = [class_boxes, class_box_scores]
		# Filter out duplicates when multiple boxes are predicted for a same object
		nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = nms_threshold)
		# Remove the duplicates from the list of classes and scores
		class_boxes = tf.gather(class_boxes, nms_index)
		class_box_scores = tf.gather(class_box_scores, nms_index)
		# Multiply score by class type
		classes = tf.ones_like(class_box_scores, 'int32') * c
		# Append results to lists
		boxes_.append(class_boxes)
		scores_.append(class_box_scores)
		classes_.append(classes)
        # Concatenate results
	boxes_ = tf.concat(boxes_, axis = 0)
	scores_ = tf.concat(scores_, axis = 0)
	classes_ = tf.concat(classes_, axis = 0)
	return boxes_, scores_, classes_


###############################################################################################################

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def pred_img(image, model_image_size, sess, input_x, input_image_shape, output_y, pred_boxes, pred_scores, pred_classes, class_names):

	global start_t
	global finish_t
	global delta_t

	#image = image[...,::-1]  					 

	# Get image height and width
	image_h, image_w, _ = image.shape

	# Resize image to fit input tensor shape
	resized_image = cv2.resize(image, model_image_size, interpolation=cv2.INTER_NEAREST)

	# Get data from resized image
	image_data = np.array(resized_image, dtype='float32')
	
	# Normalize image
	image_data /= 255.                       				
	
	 # Add batch dimension
	image_data = np.expand_dims(image_data, 0) 			

	# Get start time
	start_t = time.time()

	# Run Yolo predictor over the single image    
	out_boxes, out_scores, out_classes, out_y = sess.run([pred_boxes, pred_scores, pred_classes, output_y], feed_dict={input_x: image_data, input_image_shape: (image_h, image_w)})

	# Get end time
	finish_t = time.time()

	# Update inference duration
	delta_t += finish_t - start_t

	print("Boxes :", out_boxes)
	print("Scores :", out_scores)
	print("Classes :", out_classes)
	
	# Postprocess the detected boxes in the image
	items = []
	for i, c in reversed(list(enumerate(out_classes))):
		
		# Get class name
		predicted_class = class_names[c]
		
		# Get score
		score = out_scores[i]
	
		# Get detected box anchor
		top, left, bottom, right = out_boxes[i]
		# Adjust box coordinates
		top = max(0, np.floor(top + 0.5).astype('int32'))
		left = max(0, np.floor(left + 0.5).astype('int32'))
		bottom = min(image_h, np.floor(bottom + 0.5).astype('int32'))        
		right = min(image_w, np.floor(right + 0.5).astype('int32'))
		# Set clean bbox
		box = [left, top, right, bottom]
	
		item  = [box, score, predicted_class]
		items.append(item)

	return items


###############################################################################################################

def get_anchors(anchors_path):
        """
        Get the list of predefined boxes that best match the desired objects, from file 'anchors_path'.
        """
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors


def get_classes(classes_path):
        """
      	Get the name of the classes to detect, from file 'classes_path'.
        """
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


###############################################################################################################

def get_content_from_folder(folder_name, image_format):
    """
    Get the files from the folder named 'folder_name' having for extension 'image_format'.
    """
    try:
        content = os.listdir(folder_name)
        return [f for f in content if f.endswith('.' + image_format)]
    except FileNotFoundError:
        raise FileNotFoundError("The path {0} doesn't exist.".format(folder_name)) 


def get_images_from_files(folder_name, content):
    """
    Convert the 'content' files from folder 'folder_name' to images.
    """
    images = []
    for i in range(len(content)) :
        # Read respective image from both folders
        image = cv2.imread(os.path.join(folder_name, content[i]))
        # Append image to list
        images.append(image)
    return images


def write_results(filename, data):
	"""
	Create a text file 'filename' in which each line follows the template 'image_name.image_format label score bbox'.
	"""

	# Create output file
	label_file = open(filename, 'w')

	# Get number of processed images
	num_images = len(data)

	# Write out results
	for i in range(num_images):
		# Retrieve data
		img_name, results = data[i]
		# Check if at least one object has been detected
		#if not results:
		#	# Write data
		#	label_file.write(img_name + ' No object detected' + '\n')	
		# Get number of detected bboxes
		num_bboxes = len(results)
		for j in range(num_bboxes):
			box, score, predicted_class = results[j]
			# Write data (box is [left, top, right, bottom])
			label_file.write(img_name + ' ' + predicted_class + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + '\n')		

	# Close output file
	label_file.close()	
	
	print("The file", filename, "contains the label and detected objects of each image from the dataset.")
	

###############################################################################################################

def graph_eval(graph, input_tensor_name, output_tensors_name, anchors_path, classes_path, detection_threshold, nms_threshold, dataset, image_format, output_file):
	"""
        Run graph 'graph' (of input 'input_tensor_name' and outputs 'output_tensors_name') to detect objects over the images from the 'dataset' of format 'image_format'. 
	The files 'anchors_path' and 'classes_path' respectively indicate the anchors types and the classes to detect.
	The threshold values 'detection_threshold' and 'nms_threshold' serve to filter out unwanted detected boxes.
	Output the reuslts to file 'results'.
        """

	# Get Yolo v4 anchors (from Darknet CFG file)
	anchors = get_anchors(anchors_path)
	print('Anchors\n', anchors)

	# Retrieve classes
	class_names = get_classes(classes_path)		
	print('Classes :', class_names)
	
	# Create TF session
	sess = tf.compat.v1.Session()

	with tf.compat.v1.io.gfile.GFile(graph, 'rb') as f: 
		# Prepare a TF dataflow graph
		graph_def = tf.compat.v1.GraphDef()
		# Get graph definition from file
		graph_def.ParseFromString(f.read()) 
		sess.graph.as_default()
		# Import graph
		tf.import_graph_def(graph_def, name='')  

	# Initialize global variables
	sess.run(tf.compat.v1.global_variables_initializer())

	# Get input tensor
	input_x = sess.graph.get_tensor_by_name(input_tensor_name + ':0')         
	print("Input tensor :\n", input_x)
	# Get input width and height
	_, input_width, input_height, _ = input_x.get_shape().as_list()
	# Check whether shape is specified or not ('?')
	if(input_width == None or input_height == None):           
		input_width = int(os.environ['INPUT_WIDTH'])
		input_height = int(os.environ['INPUT_HEIGHT'])

	# Get output tensors
	output_y = [] 
	for tensor_name in output_tensors_name.split(',') :
		output_yi = sess.graph.get_tensor_by_name(tensor_name + ':0')
		output_y.append(output_yi)
	print("Output tensors :\n", output_y)

  	# Set input placeholder
	input_image_shape = tf.compat.v1.placeholder(tf.int32, shape=(2))
    	
	# Prepare Yolo inference
	pred_boxes, pred_scores, pred_classes = predict(output_y, input_image_shape, anchors, class_names, detection_threshold, nms_threshold)
	print('Boxes : ', pred_boxes)
	print('Scores : ', pred_scores)
	print('Classes : ', pred_classes)

	# Get the name of the image files from the dataset
	filenames = get_content_from_folder(dataset, image_format)
	
	# Convert files to images
	images = get_images_from_files(dataset, filenames)

	# Process images in dataset
	i = 0
	output_data = []
	for image in images :
		print('Image', filenames[i])
		# Run Yolo detection over a single image
		results = pred_img(image, (input_width, input_height), sess, input_x, input_image_shape, output_y, pred_boxes, pred_scores, pred_classes, class_names)
		print('Results', results)
		# Store output data (name of image file, [box, score, predicted_class]) 
		output_data.append([filenames[i], results])
		i+=1
	
	# Display the FPS
	#print('FPS :', len(images) / (delta_t / 1000))

	# Write out results to output file
	write_results(output_file, output_data)

	# TODO : draw boxes on image + output folder


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g',  '--graph',      type=str,   	default='./coco_frozen.pb',  	help="Path to the TensorFlow graph. Default is './yolov4_frozen.pb'.")
	parser.add_argument('-i',  '--input',      type=str, 	default='coco/net1',        	help="Name of input tensor. Default is 'coco/net1'.")
	#parser.add_argument('-o',  '--outputs',    type=str,	nargs="+", 	        	help="Name of output tensors.",  default=["coco/convolutional110/BiasAdd", "coco/convolutional102/BiasAdd", "coco/convolutional94/BiasAdd"])
	parser.add_argument('-o',  '--outputs',      type=str, 	default='conv2d_94/BiasAdd', 	help="Name of output tensors. Default is 'conv2d_94/BiasAdd'.")
	parser.add_argument('-a',  '--anchors',    type=str,   	default='./anchors.txt',        help="Path to the file that describes the Yolov4 anchors. Default is './anchors.txt'.")
	parser.add_argument('-c',  '--classes',    type=str,   	default='./classes.txt',        help="Path to the file that describes the Yolov4 classes to detect. Default is './classes.txt'.")
	parser.add_argument('-t',  '--det_thresh', type=float, 	default=0.000001,         	help="Confidence target detection threshold as an object. Default is '0.25'.")
	parser.add_argument('-n',  '--nms_thresh', type=float, 	default=0.2,         		help="Non-Max Suppression threshold to avoid ducplicate detections. Default is '0.45'.")
	parser.add_argument('-d',  '--dataset',    type=str, 	default='./data/tmp2',          help="Path to dataset images. Default is './data/tmp2'.")
	parser.add_argument('-f',  '--img_format', type=str, 	default='jpg',        		help="Format of image files in dataset. Default is 'jpg'.")
	parser.add_argument('-r',  '--results',    type=str, 	default='./output.txt',        	help="Output file containing the results following the pattern 'image_name.image_format label bbox'. Default is './output.txt'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --graph:', 	args.graph)
	print(' --input:',      args.input)
	print(' --outputs:',    args.outputs)
	print(' --anchors:', 	args.anchors)
	print(' --classes:', 	args.classes)
	print(' --det_thresh:', args.det_thresh)
	print(' --nms_thresh:', args.nms_thresh)
	print(' --dataset:',    args.dataset)
	print(' --img_format:', args.img_format)
	print(' --results:', 	args.results)
	print('------------------------------------\n')
	
	# Run the graph over the test set
	graph_eval(args.graph, args.input, args.outputs, args.anchors, args.classes, args.det_thresh, args.nms_thresh, args.dataset, args.img_format, args.results)                       
	
###############################################################################################################

if __name__ == '__main__':
	main()
		
