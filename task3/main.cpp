#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include "util.h"
#include "NeuralNetwork.h"
#include "MNIST_reader.h"

/* VALIDATION_MODE = 1 => run and print results of validation on each
 * 						  for the validation set
 * VALIDATION_MODE = 0 => construct the network and print the results of
 *						  testing on the testing set
 */ 
#define VALIDATION_MODE 1

/* How many outputs? If one, make sure to set ONE_OUTPUT to 1 in 
 * NeuralNetwork.cpp. */
#define NUM_OUTPUTS 1

/* Example slicing constants */
#define NUM_EXAMPLES 6000
#define NUM_TRAINING (int) ( (2.0/3.0) * NUM_EXAMPLES )
#define NUM_VALIDATION (int) ( (1.0/3.0) * NUM_EXAMPLES )

/* Hyperparameters */
#define HIDDEN_LAYERS 3
#define HIDDEN_LAYER_SIZE 32
#define ALPHA 8e-3
#define SEED rd()
#define EPOCHS 700

#define PRECISION 4

using Out = double;

using namespace std;

int main()
{
	cout << fixed;

	string filename = "../MNIST/train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< int> > training_images;
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cout << "Image size: " << training_images[0].size() << endl;


	filename = "../MNIST/train-labels.idx1-ubyte";

	//load MNIST labels
	vector<int> training_labels;
	loadMnistLabels(filename, training_labels);
	cout << "Number of labels: " << training_labels.size() << endl;

	// slice and normalize
	vector<vector<int>> image_slice(training_images.begin(), training_images.begin() + NUM_EXAMPLES);
	vector<Out> label_slice(training_labels.begin(), training_labels.begin() + NUM_EXAMPLES);
	auto normalized_images = normalizeImages<int>(image_slice);

	// slice training and validation
	vector<vector<double>> training_image_slice(normalized_images.begin(), normalized_images.begin() + NUM_TRAINING);
	vector<Out> training_label_slice(label_slice.begin(), label_slice.begin() + NUM_TRAINING);
	
	vector<vector<double>> validation_image_slice(normalized_images.begin() + NUM_TRAINING, normalized_images.begin() + NUM_TRAINING + NUM_VALIDATION);
	vector<Out> validation_label_slice(label_slice.begin() + NUM_TRAINING, label_slice.begin() + NUM_TRAINING + NUM_VALIDATION);

	// print info for debug
	cout << "num training images: " << training_image_slice.size() << endl
		 << "num validation images: " << validation_image_slice.size() << endl
		 << "num training labels: " << training_label_slice.size() << endl
		 << "num validation labels: " << validation_label_slice.size() << endl;

	random_device rd;
	unsigned seed = SEED;
	cout << "seed: " << seed << endl;

	// initialize, train, and validate neural network
	NeuralNetwork nn(training_images[0].size(), HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_OUTPUTS);
	nn.initialize
		( ALPHA
		, seed
		, training_image_slice // training images
		, training_label_slice // training labels
		, validation_image_slice // validation images
		, validation_label_slice // validation labels
		, EPOCHS);

#if VALIDATION_MODE
	/* trainAndValidate prints the training accuracy, training loss, validation
	 * accuracy, and validation loss at each epoch.
	 */
	nn.trainAndValidate(PRECISION);
#else
	/* Given the determined hyperparameters and seed, use this to validate
	 * them
	 */
	nn.train();
	cout << "training (epoch " << EPOCHS << "): " <<endl;
	nn.showTrainingResult();
	nn.validate();
	cout << "validation (epoch " << EPOCHS << "): " <<endl;
	nn.showValidationResult(PRECISION);


	/* testing */
	filename = "../MNIST/t10k-images-idx3-ubyte";
	vector <vector< int> > testing_images;
	loadMnistImages(filename, testing_images);
	normalized_images = normalizeImages<int>(testing_images);
	cout << "number of testing images: " << testing_images.size() << endl
		 << "size of image: " << testing_images[0].size() << endl;

	filename = "../MNIST/t10k-labels-idx1-ubyte";
	vector<int> testing_labels;
	loadMnistLabels(filename, testing_labels);
	cout << "number of testing labels: " << testing_labels.size() << endl;

	nn.initialize
		( ALPHA
		, seed
		, vector<vector<double>>() // empty
		, vector<double>() // empty
		, normalized_images // validation images
		, vector<double>(testing_labels.begin(), testing_labels.end()) // validation labels
		, EPOCHS
		, false // don't reinitialize the weights
		);

	nn.validate();
	cout << "Testing result: " << endl;
	nn.showValidationResult(PRECISION);
#endif

	return 0;
}