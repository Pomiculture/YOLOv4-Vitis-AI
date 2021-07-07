import os
import argparse

import numpy as np

###############################################################################################################

def read_file(filename):
	"""
	Read the lines of the 'filename'.
	"""
	# Open file in read-only mode
	file_content = open(filename, 'r')
	# Read lines from file
	lines = list(filter(None, [item.strip() for item in file_content.readlines()]))
	# Check whether the file is empty
	if len(lines) < 1:
		print ("File '{0}' is empty ! Exiting program...".format(filename))
		exit(-1)	
	return file_content, lines


def close_files(file_list):
	"""
	Close a list of files 'file_list'.
	"""
	for file_name in file_list :
		# Close file
		file_name.close()	


###############################################################################################################

def voc_ap(recall, precision):
	"""
	Compute VOC AP (Average Precision) given 'precision' and 'recall'.
	"""
	# Append sentinel values at the end
	m_recall = np.concatenate(([0.], recall, [1.]))
	m_precision = np.concatenate(([0.], precision, [0.]))

	# Compute the precision envelope
	for i in range(m_precision.size - 1, 0, -1):
		m_precision[i - 1] = np.maximum(m_precision[i - 1], m_precision[i])

	# Look for points where recall changes value
	i = np.where(m_recall[1:] != m_recall[:-1])[0]

	# Calculate the area under Precision-Recall curve (AUC)
	ap = np.sum((m_recall[i + 1] - m_recall[i]) * m_precision[i + 1])
	return ap
	

###############################################################################################################

def build_ground_truth_dict(ground_truth): 
	"""
	Build ground truth dictionary from ground_truth file data : detected_objects[label][images][bboxes].
	"""
	image_names = set()
	ground_truth_objects = {}   
	class_num_positives = {}

	for line in ground_truth:
		# Extract each item from the line data
		gt_data = line.split(' ')
		# Check usage
		if len(gt_data) != 6:
			print("Wrong usage : Ground truth data must follow the template 'image_name class_label x_top_left y_top_left width height'")
			exit(-1)
		# Get data
		image_name, label, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = gt_data
		# Define bounding box
		bbox = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]
		# Check whether the label is not in the dictionary yet
		if label not in ground_truth_objects.keys():
			# Add label to the keys
			ground_truth_objects[label] = {}
			class_num_positives[label] = 0
		# Check whether the image is not in the dictionary yet
		if image_name not in ground_truth_objects[label].keys():
			ground_truth_objects[label][image_name] = {'bboxes': bbox}
			# Add image to list of images
			image_names.add(image_name)												 
		else:
			# Append data to dictionary
			ground_truth_objects[label][image_name]['bboxes'] = np.vstack((ground_truth_objects[label][image_name]['bboxes'], np.array(bbox)))
		class_num_positives[label] += 1

	return ground_truth_objects, class_num_positives, image_names


def build_detected_dict(results, detection_thresh, labels):
	"""
	Build detected boxes dictionary from results file data : detected_objects[label] = {'images': [], 'scores': [], 'bboxes': []}.
	Only the bounding boxes whose confidence score is over 'detection_thresh' are used. 
	The valid labels are listed in 'labels'.
	"""
	detected_objects = {}   
	for line in results:
		# Extract each item from the line data
		result_data = line.split(' ')
		# Check usage
		if len(result_data) != 7:
			print("Wrong usage : Results data must follow the template 'image_name class_label score x_top_left y_top_left width height'")
			exit(-1)

		# Get data
		image_name, label, confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = result_data
		# Define bounding box
		bbox = [float(bbox_xmin), float(bbox_ymin), float(bbox_xmax), float(bbox_ymax)]
		# Filter relevant bounding boxes
		if float(confidence) < detection_thresh:
			print('yo')
			continue
		# Check whether the label is valid according to ground truth labels
		if label not in labels:
			print('Invalid label detected.')
			continue
		# Check whether the label is not in the dictionary yet
		if label not in detected_objects.keys():
			detected_objects[label] = {'images': [], 'scores': [], 'bboxes': []}
		# Append results to dictionary
		detected_objects[label]['images'].append(image_name)
		detected_objects[label]['bboxes'].append(bbox)
		detected_objects[label]['scores'].append(confidence)		

	return detected_objects


###############################################################################################################

def compute_iou(gt_bboxes, detected_bbox):
	"""
	Compute the Intersection over Union (IoU) between ground truth bboxes 'gt_bboxes' and a detected bbox detected_bbox'.
	"""
	# Get zone coordinates
	inter_xmin = np.maximum(gt_bboxes[:, 0], detected_bbox[0])
	inter_ymin = np.maximum(gt_bboxes[:, 1], detected_bbox[1])
	inter_xmax = np.minimum(gt_bboxes[:, 2], detected_bbox[2])
	inter_ymax = np.minimum(gt_bboxes[:, 3], detected_bbox[3])
	# Get overlap width and height
	inter_width = np.maximum(inter_xmax - inter_xmin + 1., 0.)
	inter_height = np.maximum(inter_ymax - inter_ymin + 1., 0.)
	# Compute overlap area
	inters = inter_width * inter_height

	# Compute union
	unions = ((detected_bbox[2] - detected_bbox[0] + 1.) * (detected_bbox[3] - detected_bbox[1] + 1.) \
			+ (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.) \
			- inters)

	#Get IoU
	overlaps =  inters / unions
	return overlaps

###############################################################################################################

def compute_detection_ap(results, ground_truth, detection_thresh, overlap_thresh):
	"""
	Evaluate detection results confronting the 'results' (image_name class_label score x_top_left y_top_left width height) 
	obtained after the inference phase to the 'ground_truth' data (image_name class_label x_top_left y_top_left width height).
	Only the bounding boxes whose confidence score is over 'detection_thresh' are used. 
	A match box is determined based on the IoU (Intersection over Union) ratio 'overlap_thresh'.
	"""

	# Process ground truth data
	ground_truth_objects, class_num_positives, image_names = build_ground_truth_dict(ground_truth)
	
	# Get labels from ground truth
	labels = ground_truth_objects.keys()
	
	# Process results data  
	detected_objects = build_detected_dict(results, detection_thresh, labels)
	
	print ('Evaluating ' + str(len(image_names)) + ' images...')

	# Compute recall, precision, AP
	ap = {}
	precision = {}
	recall = {}
	for label in labels:
		# Check whether some ground truth labels were not detected in the results
		if label not in detected_objects.keys():
			ap[label] = 0
			recall[label] = 0
			precision[label] = 0
			continue

		# Get ground truth images that fit the label 
		gt_label_images = ground_truth_objects[label]
		# Get number of positives for this label
		num_positives = class_num_positives[label]

		# Images detected with this label
		detected_images = detected_objects[label]['images']	
		# Get detection scores
		detected_scores = np.array(detected_objects[label]['scores'])
		# Get detected bounding boxes
		detected_bboxes = np.array(detected_objects[label]['bboxes'])

		# Sort detected objects by confidence (descending order)
		sorted_index = np.argsort(detected_scores)[::-1]
		detected_bboxes = detected_bboxes[sorted_index, :]
		detected_images = [detected_images[x] for x in sorted_index]
		
		# Mark TPs (True Positive) and FPs (False Positive)
		num_detected_images = len(detected_images)	
		true_positive = np.zeros(num_detected_images)
		false_positive = np.zeros(num_detected_images) 
		for idx in range(num_detected_images):
			# Mark detection as false positive if the image should not contain an object with this label
			if detected_images[idx] not in gt_label_images.keys():
               			false_positive[idx] = 1
                		continue
			# Get ground truth bboxes for the image
			gt_bboxes = np.array(gt_label_images[detected_images[idx]]['bboxes'], dtype=np.float32)
			# Force boxes array to be 2-Dimensional
			gt_bboxes = np.atleast_2d(gt_bboxes)
			# Get the detected bbox data
			detected_bbox = detected_bboxes[idx, :].astype(float)
			# Reset max overlap
			overlaps_max = -np.inf

			# Compute overlap
			if gt_bboxes.size > 0:
				overlaps = compute_iou(gt_bboxes, detected_bbox)           
				overlaps_max = np.max(overlaps)

			# Check whether the union is above the IoU ratio
			if overlaps_max > overlap_thresh:
				true_positive[idx] = 1.
			else:
				false_positive[idx] = 1.

		# Sum FP and TP
		false_positive = np.cumsum(false_positive)
		true_positive = np.cumsum(true_positive)

		# Compute Recall
		recall[label] = true_positive / float(num_positives)
		# Compute Precision
		precision[label] = true_positive / np.maximum(true_positive + false_positive, np.finfo(np.float64).eps)

		# Compute AP score
		ap[label] = voc_ap(recall[label], precision[label])

	return recall, precision, ap


###############################################################################################################

def display_score(ap, iou_ratio):
	"""
	Display the AP (Average Precision) score for each class, and then the mAP score (mean Average Precision) 
	given the IoU (Intersection over Union) 'iou_ratio'.
	"""
	for class_name in ap.keys():
		print('----------------------')
		print ('Class ' + class_name + ' - AP: ' + str(round(ap[class_name], 2)))
	print('######################')	
	print ('mAP@IoU' + str(round(iou_ratio * 100)) + ' score: ' + str(round((float(sum(ap.values()))) / max(1, len(ap)) * 100, 1)) + '%')
	print('######################')


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-r',  '--results_file',       	type=str,   	default='fake_results.txt',         help="Inference results file in space-separated text format. Each row is: 'image_id label score x_top_left y_top_left width height'. Default is 'fake_results.txt'.")
	parser.add_argument('-g',  '--gt_file', 		type=str,   	default='fake_ground_truth.txt',    help="Ground truth file in space-separated text format. Each row is: 'image_id label x_top_left y_top_left width height'. Default is 'fake_ground_truth.txt'.")
	parser.add_argument('-d',  '--detection_thresh',    	type=float, 	default=0.005,                      help="Threshold of confidence score for calculating evaluation metric. Default is '0.005'.")
	parser.add_argument('-i',  '--iou_thresh',   		type=float,   	default=0.5,                        help="Threshold of IOU ratio to determine a match bbox. Default is '0.5'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --results_file:',	args.results_file)
	print(' --gt_file:', 		args.gt_file)
	print(' --detection_thresh:', 	args.detection_thresh)
	print(' --iou_thresh:', 	args.iou_thresh)
	print('------------------------------------\n')

	# Get content of the results file
	f_results, results_lines = read_file(args.results_file)

	# Get content of the ground truth file  
	f_gt, gt_lines = read_file(args.gt_file)

	# Close files
	close_files([f_results, f_gt])

	# Evaluate detection results
	recall, precision, ap = compute_detection_ap(results_lines, gt_lines, args.detection_thresh, args.iou_thresh)

	# Display mAP score (Mean Average Precision)
	display_score(ap, args.iou_thresh)


###############################################################################################################

if __name__ == '__main__':
	main()
