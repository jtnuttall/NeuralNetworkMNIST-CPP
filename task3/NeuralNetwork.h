#pragma once
#include <vector>
#include <random>

using namespace std;

/**
 * @brief      A neural network.
 */
class NeuralNetwork {
public:
	/**
	 * @brief      Constructs a new neural network. Must call #initialize before running.
	 *
	 * @param[in]  nInputs          The number of inputs
	 * @param[in]  nHiddenLayers    The number of hidden layers
	 * @param[in]  hiddenLayerSize  The number of nodes in each hidden layer
	 * @param[in]  nOutputs         The number of outputs
	 */
	NeuralNetwork(int nInputs, int nHiddenLayers, int hiddenLayerSize, int nOutputs);

	~NeuralNetwork();

	/**
	 * @brief      Initialize the neural network with the given values. 
	 *             May be called with the testing sets as the validation arguments and 
	 *             followed by validate() to test the network. Be sure to pass \p shouldInitWeights 
	 *             as false in this case.
	 *
	 * @param[in]  alpha              The learning rate
	 * @param[in]  seed               The seed for initializing the weights
	 * @param[in]  exampleInputs      The training inputs
	 * @param[in]  exampleOutputs     The training outputs
	 * @param[in]  validationInputs   The validation inputs
	 * @param[in]  validationOutputs  The validation outputs
	 * @param[in]  epochs             The number of training epochs
	 * @param[in]  shouldInitWeights  Should we initialize weights to random values?
	 */
	void initialize
			( double alpha
			, unsigned seed
			, const vector<vector<double>> & exampleInputs
			, const vector<double> & exampleOutputs
			, const vector<vector<double>> & validationInputs
			, const vector<double> & validationOutputs 
			, unsigned epochs
			, bool shouldInitWeights = true );

	/**
	 * @brief      Train the function over the training inputs and validate on each 
	 * 			   epoch. 
	 * 			   Prints the prediction accuracy and loss for both the training set 
	 * 			   and the validation set on each epoch.
	 *
	 * @param[in]  precision  The precision for printing the floating point values.
	 */
	void trainAndValidate(int precision = 3);
	/**
	 * @brief      Train the network over the given training inputs.
	 */
	void train();
	/**
	 * @brief      Validate the network. Should be called after training.
	 */
	void validate();

	/**
	 * @brief      Shows the training result.
	 *
	 * @param[in]  precision  The precision for printing the floating point values. 
	 */
	void showTrainingResult(int precision = 3);
	/**
	 * @brief      Shows the validation result.
	 *
	 * @param[in]  precision  The precision for printing the floating point values.
	 */
	void showValidationResult(int precision = 3);

	/**
	 * @brief      A Single node in the neural network.
	 */
	struct Node {
		static unsigned number_count; 
		unsigned number;

		vector<Node>::size_type layerIndex;

		double activation;
		double error;
		vector<double> weights;

		/**
		 * @brief      Create a new Node.
		 *
		 * @param[in]  layerIndex  The index of this floating point in the layer.
		 */
		Node(vector<Node>::size_type layerIndex);

		Node(double activation, vector<Node>::size_type layerIndex);
	};

	typedef vector<Node> Layer;

private:
	double alpha;
	unsigned epochs;
	unsigned currentEpoch;
	vector<Layer>::size_type nOutputs;

	vector<Layer> network;

	const vector<double> * currentInput;
	const double * currentOutput;

	vector<vector<double>> exampleInputs;
	vector<double> exampleOutputs;

	vector<vector<double>> validationInputs;
	vector<double> validationOutputs;

	mt19937 generator;
	uniform_real_distribution<double> distribution;

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

	/**
	 * @brief      Do training one time for all training examples.
	 */
	void trainSingleEpoch();

	/**
	 * @brief      The activation function.
	 *
	 * @param[in]  x     Input
	 *
	 * @return     Activation
	 */
	static double g(double x);
	static double gprime(double y);
	/**
	 * @brief      Calculate the input 
	 *
	 * @param[in]  layerIterator  Iterator to the current layer
	 * @param[in]  currLayerNode  The current node in the layer
	 *
	 * @return     The weighted input for currLayerNode
	 */
	static double in(vector<Layer>::iterator currLayerIterator, const Node & currLayerNode);

	/**
	 * @brief      Print output layer
	 *
	 * @param[in]  precision  The precision for printing the floating point values.
	 */
	void printOutput(int precision = 3);

	/**
	 * @brief      Expected value of output i.
	 *
	 * @param[in]  i     Output i
	 *
	 * @return     Expected value
	 */
	double y(int i);

	Layer & inputLayer();
	vector<Layer>::iterator hiddenLayerBegin();
	vector<Layer>::iterator hiddenLayerEnd();
	Layer & outputLayer();

	/**
	 * @brief      Calculate a random nonzero weight using #generator and #distribution, 
	 *             which has been seeded by #initialize.
	 * @return     A random weight in the range given by #distribution.
	 */
	double randWeight();

	/**
	 * @brief      Initialize every weight in the network to the value returned by
	 *             randWeight().
	 */
	void initWeights();

	/**
	 * @brief      Perform forward propagation on the network.
	 */
	void forwardPropagate();
	/**
	 * @brief      Perform backward propagation on the network.
	 */
	void backwardPropagate();
	/**
	 * @brief      Update every weight in the network using the calculated errors.
	 */
	void updateWeights();

	/**
	 * @brief      Gets the label predicted by the output node(s).
	 *
	 * @return     The predicted label.
	 */
	int getPredictedLabel();
	/**
	 * @brief      Calculates the total accuracy and loss.
	 */
	void calcTotals();
	/**
	 * @brief      Calculates the loss.
	 *
	 * @return     The loss.
	 */
	double getLoss();
};