/*****************************************************************************
** Includes
*****************************************************************************/

// Standard
#include <sys/stat.h>
#include <filesystem>
#include <chrono>
#include <fstream>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Vitis AI
#include <vitis/ai/yolov3.hpp>  

/*****************************************************************************
** Macros
*****************************************************************************/

#define RECT_THICKNESS 	5
#define FONT_SCALE 	0.4
#define FONT_THICKNESS	0.6
#define PIXEL_OFFSET	10
#define WINDOW_WIDTH    500
#define WINDOW_HEIGHT   350

/*****************************************************************************
** Namespaces
*****************************************************************************/

using namespace std;
using namespace cv;

/*****************************************************************************
** Function signatures
*****************************************************************************/

// Folder management
bool createFolder(const char* folder_name);
bool getFilesFromFolder(string folder_name, vector<string> &files);

// Image management
bool saveImage(Mat img, string folder, string image_name);
void displayImage(Mat img, const char* window_name);

// Label processing
Scalar getColor(int label, int num_labels);

// Terminal actions 
string exec_cmd(const char* cmd);

/*****************************************************************************
** Main
*****************************************************************************/

/**
 * @brief Entry for running the inference phase of the Yolo v3 model                                                                
 *
 */

int main(int argc, char** argv) {

	// Check args
	if (argc != 7) {
		// Display help
		cerr << "usage: " << argv[0] << " model_name input_folder output_folder log_file classes alveo_model" << endl; 
		return EXIT_FAILURE;
	}	

	// Display args
	cout << "- Model name : " << argv[1] << "\n- Input folder : " << argv[2] << "\n- Output folder : " << argv[3] << "\n- Result log file : " << argv[4] << "\n- Classes : " << argv[5] << "\n- Alveo card : " << argv[6] << "\n" << endl; 
	cout << "\n-----------------------------------------" << endl; 

	// Get model name
	const string model_name = argv[1];

	// Get folder name to read data from
	const string input_folder = argv[2];

	// Get folder name to put output images in
	const char* output_folder = argv[3];

	// Create/clean output folder
	if (! createFolder(output_folder)) {
		return EXIT_FAILURE;
	}
	cout << "-----------------------------------------" << endl; 

	// Get image names from input folder
	vector<string> image_names;
	if (! getFilesFromFolder(input_folder, image_names)) {
		return EXIT_FAILURE;
	}
	
	// Check the number of images
	auto nb_images = image_names.size();
	if(nb_images == 0) {
		cerr << "No images to process from specified input data folder " << input_folder << ". Exiting program..." << endl;
		return EXIT_FAILURE;
	}

	// Get classes names
	const char* classes_filename = argv[5];
	vector<string> classes;
	fstream classes_file;
	// Open the file object to read the Yolo classes
	classes_file.open(classes_filename, ios::in); 
	// Read the content line by line 
	if (classes_file.is_open()){   
		string class_name;
		while (getline(classes_file, class_name)){ 
			classes.push_back(class_name);
		}
		// Close the file object.
		classes_file.close(); 
	}
	const unsigned int nb_classes = classes.size();
	if(nb_classes == 0) {
		cerr << "No classes set. Exiting program..." << endl;
		return EXIT_FAILURE;
	}

	string classes_list = "";
	for(int i = 0; i < nb_classes; i++) {
		classes_list += classes[i] + ((i == nb_classes-1) ? "" : ", ");
	}

	// Get Alveo card model
	const char* alveo_card = argv[6];

	// Set output file for logs 
	ofstream out_file(argv[4]);

	// Get datetime
    	time_t rawtime;
  	struct tm * timeinfo;
  	time (&rawtime);
  	timeinfo = localtime(&rawtime);

	// Log settings
	cout << "Yolo v3 inference phase with Alveo card " << alveo_card << endl;
	cout << asctime(timeinfo) << "Timezone - " << exec_cmd("cat /etc/timezone") << endl;
	cout << "-----------------------------------------" << endl;
	cout << "Model name : " << argv[1] << endl;
	cout << "Output images folder : " << argv[3] << "\n" << endl; 
	cout << "Classes : " << classes_list << endl;
	cout << "Evaluating " << nb_images << " images." << endl;
	cout << "\n-----------------------------------------" << endl; 
	
	// Create instance of YOLOv3 and preprocess input image by normalizing it
    	auto yolo = vitis::ai::YOLOv3::create(model_name, true);		// Need preprocess =  true	

	// DPU time
	float dpu_time = 0.0;

	// Bboxes attributes
	float xmin, ymin, xmax, ymax, confidence;

	float textX, textY;
	
	// Get start time for performance evaluation
    	auto t_start = chrono::high_resolution_clock::now();

	// Image processing - inference phase
	for (auto name : image_names) {
		// Read image as 3 channel BGR color image
	    	Mat image = imread(input_folder + "/" + name, IMREAD_COLOR);
		cout << "Image " << name << " :" << endl;
		//out_file << "Image " << name << " :" << endl;

		// Check whether image has been successfully loaded
		if (image.empty()) {
			cout << "Failed to load image " << name << endl; 
			// Proceed to next image
			break;	
		}	

// Get end DPU time
	auto t_dpu_1 = chrono::high_resolution_clock::now();
	
		// Run inference
	    	auto results = yolo->run(image);

// Get end DPU time
	auto t_dpu_2 = chrono::high_resolution_clock::now();
	
	// Get DPU delta execution
	dpu_time += (float) chrono::duration_cast<chrono::milliseconds>(t_dpu_2 - t_dpu_1).count();
	
		if(results.bboxes.empty()) {
			cout << "No object detected among known classes {" + classes_list + "}." << endl;
			// Prepare centered text
			Size textsize = getTextSize("No object detected", FONT_HERSHEY_DUPLEX, FONT_SCALE, FONT_THICKNESS, 0);
			textX = (image.cols - textsize.width) / 2;
			textY = (image.rows - textsize.height) / 2;
			// Write indication
			rectangle(image, Point(textX - PIXEL_OFFSET, textY + PIXEL_OFFSET), Point(textX + textsize.width + PIXEL_OFFSET, textY - textsize.height - PIXEL_OFFSET), CV_RGB(255, 255, 255), FILLED);
			putText(image,"No object detected", Point(textX, textY), FONT_HERSHEY_DUPLEX, FONT_SCALE, CV_RGB(255, 0, 0), FONT_THICKNESS, LINE_AA);
						
			// Report result in log file
			//out_file << "No object detected." << endl;
		}

		// Process output by tracing the boxes around the detected apple(s)
		for (auto& box : results.bboxes) {
			
			// Get label
			int label = box.label; 
								
			if(label >= nb_classes) {
				cerr << "Invalid label index.\n" << endl;
				break;
			}

			// Get box shape
			xmin = box.x * image.cols;
			ymin = box.y * image.rows;

			xmax = (box.x + box.width) * image.cols;      
			ymax = (box.y + box.height) * image.rows;

			// Check boundaries
			if (xmin < 0.) xmin = 0.;
			if (ymin < 0.) ymin = 0.;
			// Check boundaries
			if (xmax > image.cols) xmax = image.cols;
			if (ymax > image.rows) ymax = image.rows;     

			// Get score (with a 2-decimals precision)
			confidence = roundf(box.score * 100.0) / 100.0;
			// Display confidence
			cout << "Object detected : " << classes[label] << "\t" << xmin << "\t" << ymin << "\t" << xmax << "\t" << ymax << "\t" << confidence << endl;  

			// Report result to log file (image_name label confidence top_left_x, top_left_y width height)
			out_file << name << " " << label << " " << confidence << " " << box.x << " " << box.y << " " << box.width << " " << box.height << endl;

			// Draw box
			rectangle(image, Point(xmin, ymin), Point(xmax, ymax), getColor(label, nb_classes), RECT_THICKNESS, FILLED, 0);
			// Write label
			string score = to_string(confidence);

			// Write indication
			putText(image, classes[label] + " (" + score.substr(0, score.find(".") + 3) + ")", Point(xmin, ymin - 10), FONT_HERSHEY_DUPLEX, FONT_SCALE, getColor(label, nb_classes), FONT_THICKNESS, LINE_AA);			
		}
		
		// Set output image name
		auto pos = name.find(".");
		name.insert(pos, "_detected");

		// Show image
		//displayImage(image, name.c_str());

		// Write out modified image
		saveImage(image, output_folder, name);
		//out_file << "Image saved as " << name << ".\n" << endl; 
	}

	// Get end time for performance evaluation
	auto t_end = chrono::high_resolution_clock::now();

	// Calculate execution time
	auto duration = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();
	cout << "\n-----------------------------------------" << endl;  
	cout << "Processing time : " << duration << " ms" << endl;  
	
	// Display FPS
	float perfs =  roundf((nb_images / (float) (dpu_time / 1000.0)) * 100.0) / 100.0;
	cout << "Performances : " << perfs << " fps" << endl;       
	cout << "Execution time on DPU : " << dpu_time << " ms" << endl;

	// Update logs
	//out_file << "-----------------------------------------" << endl;  
	//out_file << "Processing time : " << duration << " ms" << "\nPerformances : " << perfs << " fps" << endl;
	//out_file << "Execution time on DPU : " << dpu_time << " ms" << endl;
	
	// Close log file
	out_file.flush();
	out_file.close();	

	return EXIT_SUCCESS;
}

/*****************************************************************************
** Functions
*****************************************************************************/

/**
 *
 * Create folder. If the folder already exists, remove and recreate it.
 *
 * @param folder_name - Folder's name.
 *
 * @return bool - Is folder successfully created.
 *
 */

bool createFolder(const char* folder_name) {
	int status = mkdir(folder_name, S_IRWXU | S_IRWXG | S_IRWXO);   
	if (status < 0) {
		if(errno == EEXIST) {
			// Folder already exists
			cout << "Replacing previous result folder..."  << endl;
			// Delete recursively existing folder 
			string rm_cmd = "rm -rf " + string(folder_name);
    			system(rm_cmd.c_str());
			// Re-create folder
			mkdir(folder_name, S_IRWXU | S_IRWXG | S_IRWXO);
			cout << "Folder " << folder_name << " successfully reset"  << endl;
		}
		else {
			cerr << "Failed to create folder " << folder_name << endl;
			return false;
		}
	}
	else {
		cout << "Folder " << folder_name << " successfully created" << endl;
	}
	return true;
}


/**
 *
 * Get file names from folder, if it exists.
 *
 * @param folder_name - Folder's name.
 * @param files - List of file names to fill in.
 *
 * @return bool - Is folder successfully read.
 *
 */

bool getFilesFromFolder(string folder_name, vector<string> &files) {

	// Reset list content
	files.clear();

	// Check whether folder exists
	if(filesystem::exists(folder_name)) {
		// Process folder content
		for (const auto & entry : filesystem::directory_iterator(folder_name)) {
			// Append file name to list
			files.push_back(entry.path().filename().string());
		}
		return true;
	}
	else {
		cout << "The specified folder " <<  folder_name << " does not exist." << endl;
		return false;
	}		
}


/**
 *
 * Save image into a given folder.
 *
 * @param img - Image content to save.
 * @param image_name - Name to give to the file.
 * @param folder - Folder's name.
 *
 * @return bool - Is image successfully saved.
 *
 */

bool saveImage(Mat img, string folder, string image_name) {
	bool res = false;

	try {
		res = imwrite(string(folder) + "/" + image_name, img);
	}
	catch (const Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
	}

	if (res) {
		cout << "Image '" << image_name << "' has been successfully placed in folder " << folder << "\n" << endl;
	}
	else {
		cerr << "Failed to upload output image. \n" << endl;
	}

	return res;
}


/**
 *
 * Display an image.
 *
 * @param img - Image content to display.
 * @param window_name - Name to give to the display window.
 *
 */

void displayImage(Mat img, const char* window_name) {
	// Set window name
	namedWindow(window_name, WINDOW_NORMAL);
	// Resize window
	resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT);
	// Display window
	imshow(window_name, img);
	// Wait for user to press any key
	waitKey(0);
	// Destroy window
	destroyWindow(window_name);
}


/**
 *
 * Associate a color with a given label.
 *
 * @param label - Index of label.
 * @param num_labels - Total number of labels.
 *
 * @return Scalar - The corresponding RGB code.
 *
 */

Scalar getColor(int label, int num_labels) {                      				
	int c[3];
	int scaling_factor = round(254/(num_labels-1));
	// Set RGB channels
	c[0] =  254 - label * scaling_factor;
	c[1] =  label * scaling_factor;
	c[2] =  label * scaling_factor;
	// Prevent grey color
	if(c[0] == c[1] && c[1] == c[2]) {
		c[1] =  0;
	}
	return Scalar(c[2], c[1], c[0]);
}


/**
 *
 * Execute a bash command and retrieve its output.
 *
 * @param cmd - Command to run.
 *
 * @return string - Command output.
 *
 */

string exec_cmd(const char* cmd) {
	array<char, 128> buffer;
	string result;
	unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
	if (!pipe) {
		throw runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += buffer.data();
	}
	return result;
}



