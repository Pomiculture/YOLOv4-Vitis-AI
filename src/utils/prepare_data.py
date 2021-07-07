import os
import argparse
import cv2
import glob
import shutil

###############################################################################################################

def create_empty_folder(folder_name):
	""" 
	Create new output folder with name 'folder_name'.
	"""
	if not os.path.isdir(folder_name):
		# Create output folder
		os.makedirs(folder_name)
		print("Created folder", folder_name)
	else :
		# Delete output folder content
		sub_folders_list = glob.glob(folder_name)
		for sub_folder in sub_folders_list:
			shutil.rmtree(sub_folder)
		# Create output folder
		os.makedirs(folder_name)
		print("Folder", folder_name, "already exists. Resetting content...")


def save_image(image, path, file_name) :
	""" 
	Save image 'image' as file 'file_name' to folder 'path'.
	"""  
	print('------------------------------------')
	try:
		iret = cv2.imwrite(os.path.join(path, file_name), image)
		print ('Image', file_name , 'successfully saved in folder', path)   
	except:
		print('ERROR : Failed to save', file_name) 


def copy_files(input_folder, output_folder, image_names):
	"""
	Copy images of name 'image_names' from folder 'input_folder' to folder 'output_folder'.
	"""
	# Write out images
	for image_name in image_names:
		# Get image content from filename
		image = cv2.imread(os.path.join(input_folder, image_name))
		# Save image
		save_image(image, output_folder, image_name)
	print('------------------------------------')


###############################################################################################################

def create_test_data(image_dir, output_folder, image_format, image_list):

	# Get image filenames from original dataset
	image_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith("." + image_format)]

	print('Number of images to copy :', len(image_names))

	# Create/reset output folder
	create_empty_folder(output_folder)

	# Copy selected images to output folder
	copy_files(image_dir, output_folder, image_names)

	# Create/reset file to list the calibration images 
	if image_list != '':
		image_list_file = open(image_list, 'w')

	# Get image filenames from calibration dataset
	files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.endswith("." + image_format)]

	# Write filenames in the calibration file
	for img_file in files :
		image_list_file.write(output_folder + '/' + img_file + '\n')

	# Close list file
	if image_list != '':
		image_list_file.close()
	return


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-i',  '--image_dir',     type=str,   default='./data/dataset_apples',       		help="Path to the original dataset. Default is './data/dataset_apples'.")
	parser.add_argument('-o',  '--output_folder', type=str,   default='./data/test_images', 		help="Name of the test dataset to create. Default is './data/calib_dataset'.")
	parser.add_argument('-f',  '--image_format',  type=str,   default='jpg',  choices=['png','jpg','bmp'], 	help="Format of sample images. Default is 'jpg'.")
	parser.add_argument('-l',  '--image_list',    type=str,   default='./data/test_list.txt',              help="Path to the image list to produce. Default is './data/test_list.txt'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --image_dir:', args.image_dir)
	print(' --output_folder:', args.output_folder)
	print(' --image_format:', args.image_format)
	print(' --image_list:', args.image_list)
	print('------------------------------------\n')

	# Create the test set and the list of image filenames
	create_test_data(args.image_dir, args.output_folder,  args.image_format, args.image_list)


###############################################################################################################

if __name__ == '__main__':
	main()
