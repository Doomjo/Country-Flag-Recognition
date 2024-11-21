// Check out how to do the basic morphology operations
#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/core.hpp>
#include	<opencv2/imgproc.hpp>
#include	<opencv2/ml.hpp>
#include	<iostream>
#include	"Supp.h"
#include	<filesystem>

//#include	<fstream>
//#include	<sstream>
//#include	<algorithm> 
//#include	<string> 
//#include	<vector> 


using namespace cv;
using namespace std;
using namespace cv::ml;

namespace fs = filesystem;

// sort file string
bool compareStrings(const string&, const string&);
// get classes folder path
vector<string> get_classes(string);
// read flag image
void readImages(int image_length, int image_width, vector<string> classes, vector<Mat>& images, vector<int>& labels, vector<Mat>&);

//extract edge features
vector<float> extractEdge(Mat image);
//extract hsv colour histogram
Mat computeColorHistogram(const Mat& image, int hbins = 30, int sbins = 32);
//extract hu moments
vector<float> calculateHuMoments(Mat image);
//extract hog features
void computeHOGFeatures(Mat, vector<float>&);

// get the longest contours index
int getLongestContourIndex(const vector<vector<Point>>& contours);
//get random colour
Scalar getRandomColor(RNG& rng);
//get center of contours
Point2i getContourCenter(vector<Point>	curContour);
// get bounding box based on contours
Rect getBoundingBox(Mat image);
// crop image out of bounding box
vector<Mat> extractCroppedImages(vector<Mat>);
// add bounding box to original image
vector<Mat> addBoundingBoxes(vector<Mat> images);

// test the accuracy of the input image by comparing with training data (svm, imagesize, imagesize, predict image, training image, check if predict for setD, bool for displayResult)
void testAccuracy(SVM* svm, int image_length, int image_width, vector<Mat>, vector<Mat>, bool setD = false, bool displayResult = false);
// display largeWin (list of flag images, largeWin title, fileName to save the image(null = not saving))
void displayImages(vector<Mat> images, string title, string fileName = "null");
// display predicted image with actual image
void displayPredictionResult(vector<Mat> actual, vector<Mat> predictImg, vector<int> predicted, vector<bool>);
// test the accuracy for training data
void testTrainingAccuracy(SVM* svm, vector<Mat> flags, vector<int> labels);


int main(int argc, char** argv) {
	system("cls");

	vector<Mat> displayFlags; //store clear image of each class(flags)
	string setBPath = "setB/"; // Training folder name (E.g setB/) (.png)
	string setAPath = "setA/"; // Testing folder (.jpg)
	string setCPath = "setC/"; // Testing folder (.jpg)
	string setDPath = "setD/"; // Testing folder (.jpg)


	//###################################### Step 1: Read images ######################################//
	cout << "Reading the Training Data" << endl;
	int image_length = 150, image_width = 100;
	vector<Mat> flags;
	vector<int> labels;
	vector<string> classes;

	//get directory to each flag type folder
	classes = get_classes(setBPath);

	//sort the name of file in ascending order
	sort(classes.begin(), classes.end(), compareStrings);

	// read all the flags into vector images and the corresponding label into vector labels
	readImages(image_length, image_width, classes, flags, labels, displayFlags);
	//######################################## End Read images ########################################//



	//###################################### Step 2: Extract features ######################################//
	cout << "Begin data pre-processing and feature extraction" << endl;
	vector<vector<float>> features(flags.size());
	Mat	blurred;
	vector<float> hogFeatures;

	// Color Feature Histogram
	Mat hist;
	vector<float> huFeatures;
	vector<float> edges;

	for (int i = 0; i < flags.size(); i++) {

		// Apply blurring to the image
		GaussianBlur(flags[i], blurred, Size(9, 9), 0);

		// Compute edge of image
		edges = extractEdge(blurred);

		// Compute color histograms
		hist = computeColorHistogram(blurred);

		// Convert to gray color
		cvtColor(blurred, blurred, COLOR_BGR2GRAY);

		// Compute HOG features
		computeHOGFeatures(blurred, hogFeatures);

		// Calculate the Hu moments
		huFeatures = calculateHuMoments(blurred);

		//features[i].insert(features[i].end(), edges.begin(), edges.end());
		features[i].insert(features[i].end(), hogFeatures.begin(), hogFeatures.end());
		features[i].insert(features[i].end(), hist.begin<float>(), hist.end<float>());
		//features[i].insert(features[i].end(), huFeatures.begin(), huFeatures.end());
	}
	//######################################## End Extract features ########################################//



	//###################################### Step 3: Features Pre-processing ######################################//
	// Extract feature into Mat variable for training
	Mat train_Labels(labels.size(), 1, CV_32SC1);
	for (int i = 0; i < labels.size(); i++) {
		train_Labels.at<int>(i, 0) = labels[i];
	}

	// Determine the number of rows and columns in the feature matrix
	int numImages = features.size();
	int numPixels = features[0].size();
	// Create a Mat object to hold the feature matrix
	Mat featureMat(numImages, numPixels, CV_32FC1);
	// Copy the data from the 2D vector to the Mat object
	for (int i = 0; i < numImages; i++) {
		for (int j = 0; j < numPixels; j++) {
			featureMat.at<float>(i, j) = features[i][j];
		}
	}

	// Free memory
	//flags.clear();
	//labels.clear();
	features.clear();
	hogFeatures.clear();
	hist.release();
	huFeatures.clear();
	//######################################## End Features Pre-processing ########################################//



	//######################################### Step 4: Model Training #########################################//
	// Setup the SVM parameters
	cout << "\nBegin Training SVM Model" << endl;
	bool saveModel = false;
	string saveChoice;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	try {
		svm->train(featureMat, ROW_SAMPLE, train_Labels);
		//svm = SVM::load("saved_model.xml");
		cout << "Train Succesfully\n\n" << endl;
	}
	catch (const cv::Exception& e) {
		cout << "OpenCV exception caught: " << e.what() << std::endl;
		cout << "Error code: " << e.code << std::endl;
		cout << "Error msg: " << e.msg << std::endl;
	}

	cout << "Saving the model will take some times.\nDo You Want To Save The Model ? (Y = Yes || Other Key = No):";
	cin >> saveChoice;
	if (saveChoice == "Y" || saveChoice == "y") {
		saveModel = true;
	}

	if (saveModel == true) {
		cout << "Saving the model ...";
		svm->save("saved_model.xml");
		cout << "Model Saved as saved_model.xml" << endl;
	}

	// Free memory
	featureMat.release();
	train_Labels.release();
	//######################################### End Model Training #########################################//





	//######################################### Test Set Accuracy (Set A, C and D) #########################################//
	cout << "Entering Testing Section (Press Enter to Continue)..." << endl;
	system("pause");
	system("cls");

	while (true) {
		string testSelect;
		cout << "Select Test Set That You Wish To Test: \nSet A (1)\nSet B (2) <- Test on Training Data\nSet C (3)\nSet D (4)\nQuit  (5)\nChoice (1, 2, 3, 4, 5):";
		cin >> testSelect;


		//######################################### Test Set Accuracy (Set A and D) #########################################//
		if (testSelect == "1") {
			cout << "\n================================== SET A ==================================\n";
			vector<Mat> setAImages;
			for (int i = 1; i <= 32; i++) {
				string imagePath = setAPath + to_string(i) + ".jpg";
				Mat img = imread(imagePath);
				if (!img.empty()) {
					setAImages.push_back(img);
				}
			}
			testAccuracy(svm, image_length, image_width, setAImages, displayFlags, false, true);
			system("pause");
		}

		else if (testSelect == "4") {
			cout << "\n================================== SET D ==================================\n";
			vector<Mat> setDImages;
			for (int i = 1; i <= 32; i++) {
				string imagePath = setDPath + to_string(i) + ".jpg";
				Mat img = imread(imagePath);
				if (!img.empty()) {
					setDImages.push_back(img);
				}
			}
			testAccuracy(svm, image_length, image_width, setDImages, displayFlags, true, true);
			system("pause");
		}


		//######################################### Test Set Accuracy (Set C) #########################################//
		else if (testSelect == "3") {
			cout << "\n================================== SET C ==================================\nWill take time some time to load images...\nAll images is being segmented...\n";
			vector<Mat> setCImages;

			for (int i = 1; i <= 32; i++) {
				string imagePath = setCPath + to_string(i) + ".jpg";
				Mat img = imread(imagePath);
				if (!img.empty()) {
					setCImages.push_back(img);
				}

			}
			vector<Mat> croppedSetC = extractCroppedImages(setCImages);


			//================= Display Flag Detection =================//
			bool displayBoundingBox;
			string choice;
			cout << "Do you want to display the flag detection for set C (Flags with background) ? (Y = Yes || Other key = No) : ";
			cin >> choice;
			if (choice == "Y" || choice == "y") {
				displayBoundingBox = true;
			}
			else {
				displayBoundingBox = false;
			}

			if (displayBoundingBox) {
				// original set C
				displayImages(setCImages, "Original Set C Images", "setC original");
				// bounding box set C
				vector<Mat> boundedSetC = addBoundingBoxes(setCImages);
				displayImages(boundedSetC, "Bounding Box on Set C Images", "setC with bounding box");
				// cropped set C
				displayImages(croppedSetC, "Cropped Set C Images", "setC cropped");
				waitKey(0);
				destroyAllWindows();
			}
			//===========================================================//
			//######################################### Test Set Accuracy (Set C) #########################################//
			testAccuracy(svm, image_length, image_width, croppedSetC, displayFlags, false, true);
			system("pause");
		}

		else if (testSelect == "2") {
			testTrainingAccuracy(svm, flags, labels);
			system("pause");
		}

		else if (testSelect == "5") {
			cout << "\nExiting System" << endl;
			system("pause");
			break;
		}

		else {
			cout << "Please enter correct value" << endl;
			system("pause");
		}

		system("cls");

	}


	return 0;
}






bool compareStrings(const std::string& str1, const std::string& str2) {
	int value1 = stoi(str1.substr(str1.find_last_of('/') + 1));
	int value2 = stoi(str2.substr(str2.find_last_of('/') + 1));
	return value1 < value2;
}

vector<string> get_classes(string folder) {
	vector<string> classes;
	for (const auto& entry : fs::directory_iterator(folder)) {
		if (entry.is_directory()) {
			classes.push_back(entry.path().string());
		}
	}
	return classes;
}

void readImages(int image_length, int image_width, vector<string> classes, vector<Mat>& images, vector<int>& labels, vector<Mat>& displayFlags) {
	int label_value = 1;
	for (int i = 0; i < classes.size(); i++) {
		bool first_image = true;
		for (const auto& entry : fs::directory_iterator(classes[i])) {
			if (entry.is_regular_file() && entry.path().extension() == ".png") {
				Mat image = imread(entry.path().string());

				if (image.empty()) {
					cout << "cannot open image for reading" << endl;
					return;
				}

				//Resize the Image
				resize(image, image, Size(image_length, image_width));

				//Push into vector
				images.push_back(image);
				labels.push_back(label_value);

				if (first_image) {
					displayFlags.push_back(image);
					first_image = false;
				}

				// Release memory for image data
				image.release();
			}
		}
		label_value++;
	}
}

vector<float> extractEdge(Mat image) {
	Mat blurred, region, edge;
	vector<float> features;

	cvtColor(image, blurred, COLOR_BGR2GRAY);

	// Apply thresholding to create binary image
	threshold(blurred, region, 108, 255, THRESH_OTSU);

	// Perform closing to fill small gaps
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(region, region, MORPH_CLOSE, element);

	// Perform Canny edge detection
	Canny(region, edge, 30, 90, 3);

	// Flatten the image and insert into an array
	int numPixels = edge.rows * edge.cols;

	Mat flatImage = edge.reshape(1, 1);  // Reshape to a row vector
	for (int j = 0; j < numPixels; j++) {
		features.push_back(flatImage.at<uchar>(0, j) / 255.0);  // Normalize to [0, 1]
	}

	return features;
}


Mat computeColorHistogram(const Mat& image, int hbins, int sbins) {
	// Define histogram parameters
	int histSize[] = { hbins, sbins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	int channels[] = { 0, 1 };

	// Convert image to HSV color space
	Mat hsvImage;
	cvtColor(image, hsvImage, COLOR_BGR2HSV);

	// Compute color histogram
	Mat hist;
	calcHist(&hsvImage, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

	// Return the computed histogram
	return hist;
}

vector<float> calculateHuMoments(Mat image) {
	// Calculate the moments of the image
	Moments moment = moments(image, true);

	// Calculate the Hu moments from the moments
	Mat huMoments;
	HuMoments(moment, huMoments);
	huMoments = huMoments / huMoments.at<double>(0, 0);

	vector<float> huFeatures;
	if (huMoments.type() == CV_32FC1) {
		for (int i = 0; i < 7; i++) {
			huFeatures.push_back(huMoments.at<float>(i, 0));
		}
	}
	else {
		Mat huMoments32f;
		huMoments.convertTo(huMoments32f, CV_32FC1);
		for (int i = 0; i < 7; i++) {
			huFeatures.push_back(huMoments32f.at<float>(i, 0));
		}
	}

	return huFeatures;
}

void computeHOGFeatures(Mat inputImage, vector<float>& hogFeatures) {
	// Create HOG descriptor
	Size winSize = Size(64, 64);
	Size blockSize = Size(64, 64);
	Size blockStride = Size(16, 16);
	Size cellSize = Size(16, 16);
	int nbins = 8;

	HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);

	hog.compute(inputImage, hogFeatures);
}

int getLongestContourIndex(const vector<vector<Point>>& contours) {
	int index = 0, max = 0;
	for (int i = 0; i < contours.size(); i++) {
		if (max < contours[i].size()) {
			max = contours[i].size();
			index = i;
		}
	}
	return index;
}

Scalar getRandomColor(RNG& rng) {
	int t1, t2, t3, t4;
	for (;;) {
		// get random colors that are not too dim
		t1 = rng.uniform(0, 255); // blue
		t2 = rng.uniform(0, 255); // green
		t3 = rng.uniform(0, 255); // red
		t4 = t1 + t2 + t3;
		if (t4 > 255) break;
	}
	return Scalar(t1, t2, t3);
}

Point2i getContourCenter(vector<Point>	curContour) {
	Point2i p;
	p.x = p.y = 0;
	for (int j = 0; j < curContour.size(); j++)
		p += curContour[j];
	p.x /= curContour.size();
	p.y /= curContour.size();
	return p;
}

Rect getBoundingBox(Mat image) {
	Mat cannyEdge, edge, tmp, longest;
	int ratio = 3, kernelSize = 3;

	// Perform edge detection
	Canny(image, cannyEdge, 60, 60 * ratio, kernelSize);
	cvtColor(cannyEdge, edge, COLOR_GRAY2BGR);
	dilate(edge, edge, getStructuringElement(MORPH_RECT, Size(3, 3)));
	erode(edge, edge, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1));

	// Find longest contour and fill it
	vector<vector<Point> > contours;
	cvtColor(edge, tmp, COLOR_BGR2GRAY);
	findContours(tmp, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	cvtColor(tmp, tmp, COLOR_GRAY2BGR);
	int index = getLongestContourIndex(contours);

	// Draw longest contours
	tmp = Scalar(0, 0, 0);
	drawContours(tmp, contours, index, Scalar(255, 255, 255));
	tmp.copyTo(longest);

	// Fill the countours
	vector<Point> curContour = contours[index];
	Point2i p = getContourCenter(curContour);
	floodFill(longest, p, Scalar(255, 255, 255));

	// Remove noisy data
	erode(longest, tmp, getStructuringElement(MORPH_RECT, Size(15, 15)), Point(-1, -1));

	// Extract cropped image
	cvtColor(tmp, tmp, COLOR_BGR2GRAY);
	findContours(tmp, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	cvtColor(tmp, tmp, COLOR_GRAY2BGR);

	// Get bounding box
	index = getLongestContourIndex(contours);
	Rect boundingBox = boundingRect(contours[index]);

	return boundingBox;
}

vector<Mat> extractCroppedImages(vector<Mat> images) {
	Mat original;
	vector<Mat> croppedImages;

	for (int i = 0; i < images.size(); i++) {
		images[i].copyTo(original);
		resize(original, original, Size(200, 150));

		// Get Image Bounding Box
		Rect boundingBox = getBoundingBox(original);

		// Crop Image
		Mat cropped = original(boundingBox);
		resize(cropped, cropped, Size(150, 100));

		croppedImages.push_back(cropped);
	}

	return croppedImages;
}

void testAccuracy(SVM* svm, int image_length, int image_width, vector<Mat> images, vector<Mat> actualImg, bool setD, bool displayResult) {

	Mat blur2, hist2, img;
	vector<float> descriptors2;
	vector<float> huFeatures1;
	vector<float> concatenatedFeatures;
	vector<int> predictedLabels;
	vector<bool> hitMiss;
	vector<int> actualLabels;
	vector<float> edges;
	int correctCount = 0, imageIndex = 0;

	if (setD) {
		for (int i = 33; i <= 64; i++) {
			actualLabels.push_back(i);
		}
	}
	else {
		for (int i = 1; i <= 32; i++) {
			actualLabels.push_back(i);
		}
	}

	for (int i = 0; i < images.size(); i++) {

		images[imageIndex++].copyTo(img);
		resize(img, img, Size(image_length, image_width));

		// Preprocess test image
		GaussianBlur(img, blur2, Size(9, 9), 0);

		// get edges
		edges = extractEdge(blur2);

		//get color histogram
		hist2 = computeColorHistogram(blur2, 30, 32);

		// convert to gray
		cvtColor(blur2, blur2, COLOR_BGR2GRAY);

		// Extract HOG features from test image
		computeHOGFeatures(blur2, descriptors2);

		// Calculate the moments of the image
		huFeatures1 = calculateHuMoments(blur2);

		// Concatenate the three feature vectors
		//concatenatedFeatures.insert(concatenatedFeatures.end(), edges.begin(), edges.end());
		concatenatedFeatures.insert(concatenatedFeatures.end(), descriptors2.begin(), descriptors2.end());
		concatenatedFeatures.insert(concatenatedFeatures.end(), hist2.begin<float>(), hist2.end<float>());
		//concatenatedFeatures.insert(concatenatedFeatures.end(), huFeatures1.begin(), huFeatures1.end());

		// Copy the concatenated feature vector to the featureMat2 matrix
		Mat featureMat2(1, concatenatedFeatures.size(), CV_32F);
		memcpy(featureMat2.ptr<float>(0), concatenatedFeatures.data(), concatenatedFeatures.size() * sizeof(float));

		// Predict label for test image
		int predictedLabel = svm->predict(featureMat2);

		// Print predicted label
		cout << "Actual label: " << actualLabels[i] << endl;
		cout << "Predicted label: " << predictedLabel << endl;
		predictedLabels.push_back(predictedLabel);

		// Check if prediction is correct
		if (predictedLabel == actualLabels[i]) {
			correctCount++;
			hitMiss.push_back(true);
		}
		else {
			hitMiss.push_back(false);
		}

		blur2.release();
		hist2.release();
		descriptors2.clear();
		huFeatures1.clear();
		concatenatedFeatures.clear();
	}

	// Calculate accuracy
	float accuracy = static_cast<float>(correctCount) / images.size() * 100.0;
	cout << "Accuracy: " << accuracy << "%" << endl;

	if (displayResult == true) {
		displayPredictionResult(actualImg, images, predictedLabels, hitMiss);
	}
}

void displayImages(vector<Mat> images, string title, string fileName) {
	int const	noOfImagePerCol = 4, noOfImagePerRow = 8;
	Mat			largeWin, win[noOfImagePerRow * noOfImagePerCol],
		legend[noOfImagePerRow * noOfImagePerCol];
	int			winI = 0;
	string saveName;

	resize(images[0], images[0], Size(120, 80));
	createWindowPartition(images[0], largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);

	for (int i = 0; i < images.size(); i++) {
		resize(images[i], images[i], Size(120, 80));

		images[i].copyTo(win[winI++]);
	}
	if (fileName != "null") {
		saveName = fileName + ".jpg";
		imwrite(saveName, largeWin);
	}
	imshow(title, largeWin);
}

vector<Mat> addBoundingBoxes(vector<Mat> images) {
	vector<Mat> imagesWithBoundingBoxes;
	Mat original;
	for (int i = 0; i < images.size(); i++) {
		images[i].copyTo(original);
		resize(original, original, Size(200, 150));

		// Get Image Bounding Box
		Rect boundingBox = getBoundingBox(original);
		rectangle(original, boundingBox, Scalar(0, 255, 0), 2);

		imagesWithBoundingBoxes.push_back(original);
	}
	return imagesWithBoundingBoxes;
}

void displayPredictionResult(vector<Mat> actual64Flags, vector<Mat> predictImg, vector<int> predicted, vector<bool> hitMiss) {
	int const	noOfImagePerCol = 4, noOfImagePerRow = 8;
	Mat			largeWin, win[noOfImagePerRow * noOfImagePerCol],
		legend[noOfImagePerRow * noOfImagePerCol];
	int			winI = 0;


	Mat tmp, tmp2, result;
	bool creation = true;
	string resultText;
	Scalar color;
	if (predicted.size() < 32) {
		cout << "Unable to display result as test image is less than 32." << endl;
		system("pause");
	}
	else {
		for (int i = 0; i < noOfImagePerRow * noOfImagePerCol; i++) {
			int index = predicted[i] - 1; //label start from 1 but vector array start index from 0

			//actual then predict
			resize(predictImg[i], tmp, Size(200, 150)); //test set
			resize(actual64Flags[index], tmp2, Size(200, 150));

			vconcat(tmp, tmp2, result);
			resize(result, result, Size(120, 100));

			if (creation) {
				createWindowPartition(result, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);
				creation = false;
			}

			if (hitMiss[i]) {
				resultText = "hit";
				color = Scalar(0, 250, 0); //green
			}
			else {
				resultText = "miss";
				color = Scalar(0, 0, 250); //red
			}
			result.copyTo(win[winI]);
			putText(legend[winI++], resultText, Point(5, 11), 1, 1, color, 1);
		}

		imshow("Prediction (Top: Actual, Bottom: Predicted)", largeWin);
		waitKey(0);
		destroyAllWindows();
	}

}


void testTrainingAccuracy(SVM* svm, vector<Mat> flags, vector<int> labels) {

	Mat blur2, hist2, img;
	vector<float> descriptors2;
	vector<float> huFeatures1;
	vector<float> concatenatedFeatures;
	vector<float> edges;
	int correctCount = 0;


	for (int i = 0; i < flags.size(); i++) {

		flags[i].copyTo(img);

		// Preprocess test image
		GaussianBlur(img, blur2, Size(9, 9), 0);

		// get edges
		edges = extractEdge(blur2);

		//get color histogram
		hist2 = computeColorHistogram(blur2, 30, 32);

		// convert to gray
		cvtColor(blur2, blur2, COLOR_BGR2GRAY);

		// Extract HOG features from test image
		computeHOGFeatures(blur2, descriptors2);

		// Calculate the moments of the image
		huFeatures1 = calculateHuMoments(blur2);

		// Concatenate the three feature vectors
		//concatenatedFeatures.insert(concatenatedFeatures.end(), edges.begin(), edges.end());
		concatenatedFeatures.insert(concatenatedFeatures.end(), descriptors2.begin(), descriptors2.end());
		concatenatedFeatures.insert(concatenatedFeatures.end(), hist2.begin<float>(), hist2.end<float>());
		//concatenatedFeatures.insert(concatenatedFeatures.end(), huFeatures1.begin(), huFeatures1.end());

		// Copy the concatenated feature vector to the featureMat2 matrix
		Mat featureMat2(1, concatenatedFeatures.size(), CV_32F);
		memcpy(featureMat2.ptr<float>(0), concatenatedFeatures.data(), concatenatedFeatures.size() * sizeof(float));

		// Predict label for test image
		int predictedLabel = svm->predict(featureMat2);

		// Print predicted label
		cout << "Prediction " + to_string(i + 1) << endl;
		cout << "Actual label: " << labels[i] << endl;
		cout << "Predicted label: " << predictedLabel << endl << endl;


		// Check if prediction is correct
		if (predictedLabel == labels[i]) {
			correctCount++;
		}

		blur2.release();
		hist2.release();
		descriptors2.clear();
		huFeatures1.clear();
		concatenatedFeatures.clear();
	}

	// Calculate accuracy
	float accuracy = static_cast<float>(correctCount) / flags.size() * 100.0;
	cout << "Training Accuracy: " << accuracy << "%" << endl;

}
