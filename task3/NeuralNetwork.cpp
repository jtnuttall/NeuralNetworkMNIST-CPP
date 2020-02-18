#include "NeuralNetwork.h"
#include <random>
#include <algorithm>
#include <exception>
#include <iostream>
#include <iomanip>

/* Should we reduce labels to ranges for one output? */
#define ONE_OUTPUT 1

/* Use tanh for activation function? */
#define USE_TANH 0

#if ONE_OUTPUT
const static vector<double> ranges
	{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
#endif

using Node = NeuralNetwork::Node;
using Layer = NeuralNetwork::Layer;

unsigned Node::number_count = 0;

constexpr double WEIGHT_LOWER_BOUND = -0.5;
constexpr double WEIGHT_UPPER_BOUND = 0.5;

Node::Node(Layer::size_type layerIndex) 
	: layerIndex(layerIndex)
	, activation(0.0)
	, error(0.0)
{
	number = ++number_count;
}

Node::Node(double activation, Layer::size_type layerIndex)
	: layerIndex(layerIndex)
	, activation(activation)
	, error(0.0)
{
	number = ++number_count;
}

NeuralNetwork::NeuralNetwork(int nInputs, int nHiddenLayers, int hiddenLayerSize, int nOutputs) 
	: nOutputs(nOutputs)
	, distribution(WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND)
{
	// network = { input, hidden1, hidden2, ..., hiddenN, output }

	Layer inputLayer;
	for (int n = 0; n < nInputs; ++n) {
		inputLayer.push_back(Node(n));
	}
	network.push_back(inputLayer);

	for (int l = 0; l < nHiddenLayers; ++l) {
		Layer hiddenLayer;
		for (int n = 0; n < hiddenLayerSize; ++n) {
			hiddenLayer.push_back(Node(n));
		}
		network.push_back(hiddenLayer);
	}

	Layer outputLayer;
	for (int n = 0; n < nOutputs; ++n) {
		outputLayer.push_back(Node(n));
	}
	network.push_back(outputLayer);

	for (vector<Layer>::size_type l = 1; l < network.size(); ++l) {
		for (Layer::size_type n = 0; n < network[l].size(); ++n) {
			network[l][n].weights.resize(network[l-1].size());
		}
	}
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::initialize
		( double alpha
		, unsigned seed
		, const vector<vector<double>> & exampleInputs
		, const vector<double> & exampleOutputs
		, const vector<vector<double>> & validationInputs
		, const vector<double> & validationOutputs 
		, unsigned epochs
		, bool shouldInitWeights // defaults to true
		)
{
	this->alpha = alpha;
	this->exampleInputs = exampleInputs;
	this->exampleOutputs = exampleOutputs;
	this->validationInputs = validationInputs;
	this->validationOutputs = validationOutputs;
	this->epochs = epochs;

	generator.seed(seed);

	if (shouldInitWeights) {
		initWeights();
	}
}

void NeuralNetwork::showTrainingResult(int precision) {
	calcTotals();
	cout << setprecision(precision)
		 << "Training (epoch " << currentEpoch << "):" << endl
		 << "\tAccuracy: " << trainAccuracy
		 << "\tLoss: " << trainLoss
		 << endl;
}

void NeuralNetwork::showValidationResult(int precision) {
	calcTotals();
	cout << setprecision(precision) 
		 << "Training (epoch " << currentEpoch << "):" << endl
		 << "\tAccuracy: " << valAccuracy
		 << "\tLoss: " << valLoss
		 << endl;
}

void NeuralNetwork::trainAndValidate(int precision) {
	currentEpoch = 0;

	do {
		++currentEpoch;
		trainSingleEpoch();
		validate();

		calcTotals();

		cout << setprecision(precision)
			 << "epoch: " << currentEpoch << ", "
			 << "trainAccuracy: " << trainAccuracy << ", "
			 << "trainLoss: " << trainLoss << ", "
			 << "valAccuracy: " << valAccuracy << ", "
			 << "valLoss: " << valLoss
			 << endl;
	} while (currentEpoch < epochs);
}

void NeuralNetwork::train() {
	currentEpoch = 0;

	do {
		++currentEpoch;
		trainSingleEpoch();
	} while (currentEpoch < epochs);
}

void NeuralNetwork::trainSingleEpoch() {
	totalTrainSamples = 0;
	totalCorrectTrainSamples = 0;
	totalTrainLoss = 0.0;

	for (vector<vector<double>>::size_type i = 0; i < exampleInputs.size(); ++i) {
		currentInput = &exampleInputs[i];
		currentOutput = &exampleOutputs[i];

		forwardPropagate();
		totalTrainLoss += getLoss();

		backwardPropagate();
		updateWeights();

		if (getPredictedLabel() == (int)*currentOutput) {
			++totalCorrectTrainSamples;
		}
		++totalTrainSamples;
	}
}

void NeuralNetwork::validate() {
    totalValSamples = 0;
    totalCorrectValSamples = 0;
    totalValLoss = 0.0;

	for (vector<vector<double>>::size_type i = 0; i < validationInputs.size(); ++i) {
		currentInput = &validationInputs[i];
		currentOutput = &validationOutputs[i];

		forwardPropagate();

		totalValLoss += getLoss();
		if ((int)*currentOutput == getPredictedLabel()) {
			++totalCorrectValSamples;
		}
		++totalValSamples;
	}
}

double NeuralNetwork::in(vector<Layer>::iterator currLayerIterator, const Node & currLayerNode) {
	if (currLayerNode.weights.empty()) {
		return 0.0;
	}

	double in = 0.0;
	// j.weights[i] is the weight from node i to node j, so j.weights[i] = w_i_j
	// i = prevLayerNode
	// j = currLayerNode
	for (const Node & prevLayerNode : *(currLayerIterator-1)) {
		in += currLayerNode.weights[prevLayerNode.layerIndex] * prevLayerNode.activation;
	}

	return in;
}

void NeuralNetwork::forwardPropagate() {
	for (Node & n : inputLayer()) {
		n.activation = currentInput->at(n.layerIndex);
	}

	for (auto layerIterator = hiddenLayerBegin(); layerIterator != network.end(); ++layerIterator) {
		for (Node & currLayerNode : *layerIterator) {
			currLayerNode.activation = g(in(layerIterator, currLayerNode));
		}
	}
}

void NeuralNetwork::backwardPropagate() {
	for (Node & node : outputLayer()) {
		node.error = gprime(node.activation) * (y(node.layerIndex) - node.activation);
	}

	// output layer at rbegin() => rbegin()+1 = first hidden layer
	// input layer at rend()-1
	for (auto layerIterator = network.rbegin()+1; layerIterator != network.rend(); ++layerIterator) {
		for (Node & currLayerNode : *layerIterator) {
			double sumPrevError = 0.0;
			// Node.weights[n] is the weight from node n in the previous layer to this Node
			// 		if you let n be previous node and m be this, then it's w_n_m = m.weights[n]
			// i = currLayerNode
			// j = prevLayerNode
			// we want w_i_j, so that's prevLayerNode.weights[currLayerNode]
			for (const Node & prevLayerNode : *(layerIterator-1)) {
				sumPrevError += prevLayerNode.weights[currLayerNode.layerIndex] * prevLayerNode.error;
			}

			currLayerNode.error = gprime(currLayerNode.activation) * sumPrevError;
		}
	}
}

void NeuralNetwork::updateWeights() {
	for (auto layerIterator = hiddenLayerBegin(); layerIterator != network.end(); ++layerIterator) {
		for (Node & currLayerNode : *layerIterator) {
			for (const Node & prevLayerNode : *(layerIterator-1)) {
				currLayerNode.weights[prevLayerNode.layerIndex] += alpha * prevLayerNode.activation * currLayerNode.error;
			}
		}
	}
}

// calculate a random nonzero weight between -0.05 and 0.05
double NeuralNetwork::randWeight() { 
	double r;
	do {
		r = distribution(generator);
	} while (r == 0.0);
	return r;
}

// assign random weights to nodes
void NeuralNetwork::initWeights() {
	// Node.weights[i] is the weight from node i in the previous layer to this Node
	// therefore, network[0] (input layer) has weights.empty()

	for (Layer & layer : network) {
		for (Node & node : layer) {
			for (double & weight : node.weights) {
				weight = randWeight();
			}
		}
	}
}

#if USE_TANH
double NeuralNetwork::g(double x) 
	{ return (exp(x) - exp(-x))/(exp(x) + exp(-x)); }
double NeuralNetwork::gprime(double y) { return 1 - (y*y); }

#else
double NeuralNetwork::g(double x) { return 1.0 / (1.0 + exp(-x)); }
double NeuralNetwork::gprime(double y) { return y * (1 - y); }
#endif

void NeuralNetwork::printOutput(int precision) {
	cout << "{ ";
	for (const Node & node : outputLayer()) {
		cout << setprecision(precision)
			 << node.layerIndex << " => " << node.activation << ", ";
	}
	cout << " }" << endl;
}

#if ONE_OUTPUT
double NeuralNetwork::y(int i) {
	return (*currentOutput)/10.0;
}
#else
double NeuralNetwork::y(int i) {
	return double(i == (int)*currentOutput);
}
#endif

Layer & NeuralNetwork::inputLayer() {
	return *network.begin();
}

vector<Layer>::iterator NeuralNetwork::hiddenLayerBegin() {
	return network.begin()+1;
}

vector<Layer>::iterator NeuralNetwork::hiddenLayerEnd() {
	return network.end()-1;
}

Layer & NeuralNetwork::outputLayer() {
	return *network.rbegin();
}

#if ONE_OUTPUT
int NeuralNetwork::getPredictedLabel() {
	if (nOutputs != 1) {
		throw logic_error("nOutputs != 1");
	}

	int result = 0;
	for (double rangeMax : ranges) {
		if ((*outputLayer().begin()).activation <= rangeMax) {
			break;
		}
		++result;
	}
	if (result > 9) {
		throw logic_error("invalid output result");
	}

	return result;
}

#else
int NeuralNetwork::getPredictedLabel() {
	auto maxIter = max_element(outputLayer().begin(), outputLayer().end(), 
		[](const Node & n1, const Node & n2) { 
			return n1.activation < n2.activation; 
		}
	);
	return maxIter - outputLayer().begin();
}
#endif

void NeuralNetwork::calcTotals() {
    trainAccuracy = totalCorrectTrainSamples / (double)totalTrainSamples;
    trainLoss = totalTrainLoss / (double)totalTrainSamples;
    valAccuracy = totalCorrectValSamples / (double)totalValSamples;
    valLoss = totalValLoss / (double)totalValSamples;
}

double NeuralNetwork::getLoss() {
	// SUM forall i OF (y_i - o_i) ^ 2
	// y_i == desired output
	// o_i == actual output
	double loss = 0.0;
	double yioi;

	for (const Node & node : outputLayer()) { 
		yioi = y(node.layerIndex) - node.activation;
		loss += 100.0 * yioi * yioi;
	}

	return loss;
}