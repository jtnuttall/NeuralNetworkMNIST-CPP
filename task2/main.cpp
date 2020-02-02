#include <vector>
#include <iostream>

#include "SimpleFeedForwardNetwork.h"


using namespace std;
int main()
{
	// hyper-paramters
	size_t inputLayerSize = 2;

	// // epoch = 9999, outputs = 0.09 0.93 0.93 0.062
	// // time = 0m0.284s
	// double alpha = 0.1;   // learning rate
	// size_t hiddenLayerSize = 5;
	// size_t numEpochs = 10000;

	// epoch = 9999, outputs = 0.047 0.97 0.97 0.022
	// time = 0m0.264s
	// double alpha = 0.3;
	// size_t hiddenLayerSize = 5;
	// size_t numEpochs = 10000;

	// //epoch = 9999, outputs = 0.025 0.98 0.98 0.024  //g2g
	// // time = 0m0.576s
	// double alpha = 0.4;
	// size_t hiddenLayerSize = 25;
	// size_t numEpochs = 10000;

	// // epoch = 49999, outputs = 0.013 0.99 0.99 0.01 // g2g
	// // time = 0m1.828s
	// double alpha = 0.3;
	// size_t hiddenLayerSize = 10;
	// size_t numEpochs = 50000;

	// epoch = 69999, outputs = 0.015 0.99 0.99 0.015 // g2g
	// time = 0m3.812s
	double alpha = 0.1;
	size_t hiddenLayerSize = 20;
	size_t numEpochs = 70000;

	int seed = 0; // random seed for the network initialization

	// input data
	vector< vector< double > > x(4);
	x[0].push_back(0);
	x[0].push_back(0);
	x[1].push_back(0);
	x[1].push_back(1);
	x[2].push_back(1);
	x[2].push_back(0);
	x[3].push_back(1);
	x[3].push_back(1);
	vector< double > y{ 0, 1, 1, 0 };


	SimpleFeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize);
	nn.initialize(seed);
	nn.train(x, y, numEpochs);
	return 0;
}
