#pragma once
#include <vector>

using namespace std;

class NeuralNetwork {
public:
	NeuralNetwork(int nInputs, int nHiddenLayers, int hiddenLayerSize, int nOutputs);

	~NeuralNetwork();

	void initialize
			( double alpha
			, time_t seed
			, const vector<vector<double>> & exampleInputs
			, const vector<double> & exampleOutputs
			, const vector<vector<double>> & validationInputs
			, const vector<double> & validationOutputs 
			, unsigned epochs
			, bool shouldInitWeights = true );

	void trainAndValidate();
	void train();
	void validate();

	void showTrainingResult();
	void showValidationResult();

	//
	struct Node {
		// the number of the node, starting at 0 for beginning of input layer
		// debug purposes atm
		static unsigned number_count; 
		unsigned number;
		vector<Node>::size_type layerIndex;

		double activation;
		double error;
		vector<double> weights;

		Node(vector<Node>::size_type layerIndex);
	};

	typedef vector<Node> Layer;

private:
	double alpha;
	int nInputs;
	int nHiddenLayers;
	int hiddenLayerSize;
	int nOutputs;
	unsigned epochs;
	int seed;

	const vector<double> * currentInput;
	const double * currentOutput;

	vector<vector<double>> exampleInputs;
	vector<double> exampleOutputs;

	vector<vector<double>> validationInputs;
	vector<double> validationOutputs;

	vector<Layer> network;

	// loss variables
	unsigned totalTrainSamples;
	unsigned totalCorrectTrainSamples;
	double totalTrainLoss;

	unsigned totalValSamples;
	unsigned totalCorrectValSamples;
	double totalValLoss;

	double trainAccuracy;
	double trainLoss;
	double valAccuracy;
	double valLoss;

	void trainSingleEpoch();

	double g(double x);
	double gprime(double y);

	static double in(vector<Layer>::iterator layerIterator, const Node & currLayerNode);

	void printOutput();

	double y(int i);

	Layer & inputLayer();
	vector<Layer>::iterator hiddenLayerBegin();
	vector<Layer>::iterator hiddenLayerEnd();
	Layer & outputLayer();

	// calculate a random nonzero weight between -1.0 and 1.0
	double randWeight();

	// assign random weights to nodes
	void initWeights();

	void forwardPropagate();
	void backwardPropagate();
	void updateWeights();

	int getPredictedLabel();
	void calcTotals();
	double getLoss();
};