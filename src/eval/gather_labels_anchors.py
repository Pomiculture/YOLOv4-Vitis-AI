import os
import argparse

###############################################################################################################

def gather_data(dataset, image_format, labels_anchors_file) :
	"""
	Create a file 'labels_anchors_file' for which each line follows the template 'image_name.image_format label anchor'.
	The images are taken from the dataset 'dataset'.
	"""

	# Create output file
	label_file = open(labels_anchors_file, 'w')

	# Get labels and anchors filenames from samples in dataset
	files = [f for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f)) and f.endswith(".txt")]

	# Write data to the output file
	for data_file in files :
		# Read label/anchor content of image
		label_anchor = open(os.path.join(dataset, data_file), 'r')
		# Read anchors
		anchors = list(filter(None, [item.strip() for item in label_anchor.readlines()]))
		# Concatenate image name
		for anchor in anchors:
			line_content = data_file.replace("txt", image_format) + ' ' + anchor
			# Write data
			label_file.write(line_content + '\n')

	# Close output file
	label_file.close()	

	print('The file', labels_anchors_file, 'contains the label and anchor(s) of each image from the dataset', dataset)


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-d',  '--dataset',      type=str,   default='./data/eval_set',     help="Path to the dataset folder to read from. Default is './data/eval_set'.")
	parser.add_argument('-f',  '--image_format', type=str,   default='jpg',                 help="Format of the images to process. Default is 'jpg'.")
	parser.add_argument('-o',  '--output_file',  type=str,   default='labels_anchors.txt',  help="TensorBoard port number. Default is 'labels_anchors.txt'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --dataset:', args.dataset)
	print(' --image_format:', args.image_format)
	print(' --output_file:', args.output_file)
	print('------------------------------------\n')

	# Gather labels and anchors of the dataset samples (format : 'image_name label bbox_anchor')
	gather_data(args.dataset, args.image_format, args.output_file)
	

###############################################################################################################

if __name__ == '__main__':
    main()

