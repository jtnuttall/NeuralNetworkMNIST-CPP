#include "NeuralNetwork.h"
#include <random>
#include <cstdio>
#include <algorithm>
#include <exception>
#include <cmath>

/* Should we reduce labels to ranges for one output? */
#define ONE_OUTPUT 0
#if ONE_OUTPUT
const static vector<double> ranges
	{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
#endif

/* Use tanh for activation function? */
#define USE_TANH 1

typedef NeuralNetwork::Node Node;
typedef NeuralNetwork::Layer Layer;

unsigned Node::number_count = 0;

Node::Node(Layer::size_type layerIndex) 
	: layerIndex(layerIndex)
{
	number = ++number_count;
}

NeuralNetwork::NeuralNetwork(int nInputs, int nHiddenLayers, int hiddenLayerSize, int nOutputs) 
	: nInputs(nInputs)
	, nHiddenLayers(nHiddenLayers)
	, hiddenLayerSize(hiddenLayerSize)
	, nOutputs(nOutputs)
{
	// network = { input, hidden1, hidden2, ..., hiddenN, output }

	Layer inputLayer;
	for (int n = 0; n < nInputs; ++n) {
		inputLayer.push_back(Node(n));
	}
	network.push_back(inputLayer);

	for (int l = 0; l < nHiddenLayers; ++l) {
		Layer hiddenLayer;
		for (int n = 0; n < hiddenLayerSize; ++n)
			hiddenLayer.push_back(Node(n));
		network.push_back(hiddenLayer);
	}

	Layer outputLayer;
	for (int n = 0; n < nOutputs; ++n) {
		outputLayer.push_back(Node(n));
	}
	network.push_back(outputLayer);

	// output + hidden - input
	for (vector<Layer>::size_type l = 1; l < network.size(); ++l) {
		// layer = network[l]
		for (Layer::size_type n = 0; n < network[l].size(); ++n) {
			network[l][n].weights.resize(network[l-1].size());
		}
	}
}

NeuralNetwork::~NeuralNetwork() {
}

void NeuralNetwork::initialize
		( double alpha
		, time_t seed
		, const vector<vector<double>> & exampleInputs
		, const vector<double> & exampleOutputs
		, const vector<vector<double>> & validationInputs
		, const vector<double> & validationOutputs 
		, unsigned epochs
		, bool shouldInitWeights )
{
	this->alpha = alpha;
	this->seed = seed;
	this->exampleInputs = exampleInputs;
	this->exampleOutputs = exampleOutputs;
	this->validationInputs = validationInputs;
	this->validationOutputs = validationOutputs;
	this->epochs = epochs;

	if (shouldInitWeights) {
		initWeights();
	}
}

void NeuralNetwork::showTrainingResult() {
	calcTotals();
	printf("accuracy: %0.3lf, loss: %0.3lf\n", trainAccuracy, trainLoss);
}

void NeuralNetwork::showValidationResult() {
	calcTotals();
	printf("accuracy: %0.3lf, loss: %0.3lf\n", valAccuracy, valLoss);
}

void NeuralNetwork::trainAndValidate() {
	unsigned current_epoch = 0;

	do {
		++current_epoch;
		trainSingleEpoch();
		validate();

		calcTotals();

	    printf("epoch: %u, trainAccuracy: %0.3lf, trainLoss: %0.3lf, valAccuracy: %0.3lf, valLoss: %0.3lf\n",
	    	current_epoch, trainAccuracy, trainLoss, valAccuracy, valLoss);
	} while (current_epoch < epochs);
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

void NeuralNetwork::train() {
	unsigned current_epoch = 0;

	do {
		++current_epoch;
		trainSingleEpoch();
	} while (current_epoch < epochs);
}

inline void NeuralNetwork::trainSingleEpoch() {
	totalTrainSamples = 0;
	totalCorrectTrainSamples = 0;
	totalTrainLoss = 0.0;

	for (vector<vector<double>>::size_type i = 0; i < exampleInputs.size(); ++i) {
		currentInput = &exampleInputs[i];
		currentOutput = &exampleOutputs[i];
		// printf("epoch: %u  target: %0.1lf\n", current_epoch, exampleOutputs[currentExample]);

		// do the thing
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

inline double NeuralNetwork::in(vector<Layer>::iterator currLayerIterator, const Node & currLayerNode) {
	if (currLayerNode.weights.empty()) return 0.0;

	double in = 0.0;
	// j.weights[i] is the weight from node i to node j, so j.weights[i] = w_i_j
	// i = prevLayerNode
	// j = currLayerNode
	for (const Node & prevLayerNode : *(currLayerIterator-1)) {
		in += currLayerNode.weights[prevLayerNode.layerIndex] * prevLayerNode.activation;
	}

	return in;
}

inline void NeuralNetwork::forwardPropagate() {
	for (Node & n : inputLayer()) {
		n.activation = currentInput->at(n.layerIndex);
	}

	for (auto layerIterator = hiddenLayerBegin(); layerIterator != network.end(); ++layerIterator) {
		for (Node & currLayerNode : *layerIterator) {
			currLayerNode.activation = g(in(layerIterator, currLayerNode));
		}
	}
}

inline void NeuralNetwork::backwardPropagate() {
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

			// currLayerNode.error = gprime(g(in(layerIterator.base() - 1, currLayerNode))) * sumPrevError;
			currLayerNode.error = gprime(currLayerNode.activation) * sumPrevError;
		}
	}
}

inline void NeuralNetwork::updateWeights() {
	for (auto layerIterator = hiddenLayerBegin(); layerIterator != network.end(); ++layerIterator) {
		for (Node & currLayerNode : *layerIterator) {
			for (const Node & prevLayerNode : *(layerIterator-1)) {
				currLayerNode.weights[prevLayerNode.layerIndex] += alpha * prevLayerNode.activation * currLayerNode.error;
			}
		}
	}
}

// calculate a random nonzero weight between -0.05 and 0.05
inline double NeuralNetwork::randWeight() { 
	int r;
	do {
		r = (rand() % 200) - 99;
	} while (r == 0);
	return r * 5.0 / 1000;
}

// assign random weights to nodes
inline void NeuralNetwork::initWeights() {
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
inline double NeuralNetwork::g(double x) 
	{ return (exp(x) - exp(-x))/(exp(x) + exp(-x)); }
inline double NeuralNetwork::gprime(double y) { return 1 - (y*y); }

#else
inline double NeuralNetwork::g(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double NeuralNetwork::gprime(double y) { return y * (1 - y); }
#endif

inline void NeuralNetwork::printOutput() {
	printf("\t\t{ ");
	for (const Node & node : outputLayer()) {
		printf("%lu => %0.2lf, ", node.layerIndex, node.activation);
	}
	printf("}\n");
}

#if ONE_OUTPUT
inline double NeuralNetwork::y(int i) {
	return (*currentOutput)/10.0;
}
#else
inline double NeuralNetwork::y(int i) {
	return double(i == (int)*currentOutput);
}
#endif

inline Layer & NeuralNetwork::inputLayer() {
	return *network.begin();
}

inline vector<Layer>::iterator NeuralNetwork::hiddenLayerBegin() {
	return network.begin()+1;
}

inline vector<Layer>::iterator NeuralNetwork::hiddenLayerEnd() {
	return network.end()-1;
}

inline Layer & NeuralNetwork::outputLayer() {
	return *network.rbegin();
}

#if ONE_OUTPUT
inline int NeuralNetwork::getPredictedLabel() {
	if (nOutputs != 1) 
		throw logic_error("nOutputs != 1");

	int result = 0;
	for (double rangeMax : ranges) {
		if ((*outputLayer().begin()).activation <= rangeMax) break;
		++result;
	}
	if (result > 9) 
		throw logic_error("invalid output result");

	return result;
}

#else
inline int NeuralNetwork::getPredictedLabel() {
	Layer::iterator maxIter = max_element(outputLayer().begin(), outputLayer().end(), 
		[](const Node & n1, const Node & n2) { 
			return n1.activation < n2.activation; 
		}
	);
	return maxIter - outputLayer().begin();
}
#endif

inline void NeuralNetwork::calcTotals() {
    trainAccuracy = totalCorrectTrainSamples / (double)totalTrainSamples;
    trainLoss = totalTrainLoss / (double)totalTrainSamples;
    valAccuracy = totalCorrectValSamples / (double)totalValSamples;
    valLoss = totalValLoss / (double)totalValSamples;
}

inline double NeuralNetwork::getLoss() {
	// SUM forall i OF (y_i - o_i) ^ 2
	// y_i == desired output
	// o_i == actual output
	double loss = 0.0;
	double yioi;

	for (const Node & node : outputLayer()) { 
		yioi = y(node.layerIndex) - node.activation;
		loss += yioi * yioi;
	}

	return loss;
}