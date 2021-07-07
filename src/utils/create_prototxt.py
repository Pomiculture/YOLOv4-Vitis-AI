###############################################################################################################

# Source : https://www.xilinx.com/html_docs/vitis_ai/1_3/prog_examples.html

###############################################################################################################

import os
import argparse

###############################################################################################################

def get_num_classes(classes_path):
        """
      	Get the number of classes to detect, from file 'classes_path'.
        """
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return len(class_names)


def get_anchors(anchors_path):
        """
        Get the list of predefined boxes that best match the desired objects, from file 'anchors_path'. 
        """
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [int(x) for x in anchors.split(',')]
        return anchors


def write_to_file(filename, prefix, content):
	"""
	Write 'prefix' and 'content' to file 'filename' and carriage return.
	"""
	filename.write(prefix + content + "\n")
	

###############################################################################################################

def create_prototxt_file(output_folder, model_name, output_layers, conf_threshold, nms_threshold, classes_path, anchors_path, mean, scale):
	"""
	Create a prototxt file to run a YOLOv3 or YOLOv4 model on the accelerator card (along with the xmodel).
	"""
	# Get number of classes
	num_classes = get_num_classes(classes_path)
	# Get anchors data
	anchor_data = get_anchors(anchors_path)
	# Number of occurrences of the anchors in the Darknet model (cfg file).
	num_anchors = len(output_layers)

	# Create new file in write-only mode
	f = open(os.path.join(output_folder, model_name + ".prototxt"), "w")
	# Set model wrapper
	write_to_file(f, "", "model {")
	# Set model name
	write_to_file(f, "\t", "name: \"" + model_name + "\"")
	# Set kernel wrapper
	write_to_file(f, "\t", "kernel {")
	for _ in range(3):
		write_to_file(f, "\t\t", "mean: "+ str(mean))
	for _ in range(3):
		write_to_file(f, "\t\t", "scale: "+ str(scale))
	write_to_file(f, "\t", "}")
	# Set model type
	write_to_file(f, "\t", "model_type: YOLOv3")    
	write_to_file(f, "\t", "yolo_v3_param { ")       
	# Set YOLOv3 params wrapper
	write_to_file(f, "\t\t", "num_classes: " + str(num_classes)) 
	write_to_file(f, "\t\t", "anchorCnt: " + str(num_anchors)) 
	# Indicate name of output layers
	for layer_name in output_layers:
		write_to_file(f, "\t\t", "layer_name: \"" + layer_name + "\"")
	# Set boxes confidence threshold 
	write_to_file(f, "\t\t", "conf_threshold: " + str(conf_threshold)) 
	# Set NMS (Non-maximum Suppression) threshold 
	write_to_file(f, "\t\t", "nms_threshold: " + str(nms_threshold))
	# Write out anchor values
	for anchor_value in anchor_data:
		write_to_file(f, "\t\t", "biases: " + str(anchor_value))
	# Model not trained with letterbox
	write_to_file(f, "\t\t", "test_mAP: false")
	write_to_file(f, "\t", "}")
	write_to_file(f, "", "}")
	# End writing
	f.close()


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-m',  '--model_name', 	type=str,   	default='yolov4_apple',  	help="Name of the YOLOv3/YOLOv4 model. Default is 'yolov4_apple'.")
	parser.add_argument('-l',  '--output_layers', 	type=str,   	default='110,102,94',  		help="Name of the model output layers. Default is '110,102,94'.")
	parser.add_argument('-t',  '--conf_threshold', 	type=float,   	default=0.8,  			help="Object detection threshold. Default is '0.8'.")
	parser.add_argument('-n',  '--nms_threshold', 	type=float,   	default=0.4,  			help="Non-Maximum Suppression threshold. Default is '0.4'.")
	parser.add_argument('-c',  '--classes_path', 	type=str,   	default='./classes.txt',  	help="Path to the name of the classes to detect. Default is './classes.txt'.")
	parser.add_argument('-a',  '--anchors_path', 	type=str,   	default='./anchors.txt',  	help="Path to the name of the YOLO anchors. Default is './anchors.txt'.")
	parser.add_argument('-o',  '--output_folder', 	type=str,   	default='./prototxt',  		help="Name of output folder. Default is '.prototxt'.")

	# Parse arguments
	args = parser.parse_args()  
	
	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print('--model_name:', args.model_name)
	print('--output_layers:', args.output_layers)
	print('--conf_threshold:', args.conf_threshold)
	print('--nms_threshold:', args.nms_threshold)
	print('--classes_path:', args.classes_path)
	print('--anchors_path:', args.anchors_path)
	print('--output_folder:', args.output_folder)
	print('------------------------------------\n')

	# Mean-value of “BRG”
	mean = 0.0
	# RGB-normalized scale
	scale = 0.00390625 

	# Split output layers 
	output_layers = args.output_layers.split(',')[::-1]
	output_indexes = [None] * len(output_layers)
	i = 0

	for layer_name in output_layers:
		layer_index = layer_name.split('_')[-1]
		number = ''.join(filter(str.isdigit, layer_index))
		output_indexes[i] = number
		i += 1
	
	# Create prototxt file
	create_prototxt_file(args.output_folder, args.model_name, output_indexes, args.conf_threshold, args.nms_threshold, args.classes_path, args.anchors_path, mean, scale)


###############################################################################################################

if __name__ == '__main__':
	main()

