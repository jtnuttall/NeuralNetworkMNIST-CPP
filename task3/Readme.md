# Neural Network for MNIST Digit Recognition

## Build Environment Information
### OS
	System Version: macOS 10.14 (18A391)
	Kernel Version: Darwin 18.0.0
	Processor Name: Intel Core i7
	Processor Speed: 2.9 GHz


### C++ Compiler
	Apple LLVM version 10.0.1 (clang-1001.0.46.4)
	Target: x86_64-apple-darwin18.0.0
	Thread model: posix

Note: `clang` is aliased as `gcc` and `g++` by default on Macs with the XCode toolchain 
installed, so this is the output of `g++ --version`. However, as long as the `--std=c++0x` 
flag is set, the code should compile fine with GNU `g++`.

### Flags
Aside from the given flags `-g`, `-Wall`, and `--std=c++0x`, I have added `-Wpedantic` and
`-O3`. 
Rationale:

- `-Wpedantic`: Warns on forbidden extensions and for certain extra potential error cases.
- `-O3`: Produces a signifcant performance gain. If removed, the program will still run, but 
I strongly suggest you leave it in. The high level solution is reducible to matrix math, so
`-O3` reduces it to this. Without it, it will perform a whole lot of extra operations 
specified by the `C++11` features I have used.


## Hyperperameters for Accuracy > 0.9
Hyperperameters are `#define`d at the top of `main.cpp`.

For random weights, `#define SEED time(NULL)` will work as expected.

### Testing Accuracy: `0.912`
```cpp
	#define HIDDEN_LAYERS 3
	#define HIDDEN_LAYER_SIZE 32
	#define ALPHA 8e-3
	#define SEED 1570649057
	#define EPOCHS 508
```
#### Time taken
Rewrite `main`:
```cpp
	main () {
		{ load MNIST }

		NeuralNetwork nn(training_images[0].size(), HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, 10);
		nn.initialize
			( ALPHA
			, seed
			, training_image_slice // training images
			, training_label_slice // training labels
			, validation_image_slice // validation images
			, validation_label_slice // validation labels
			, EPOCHS );

		nn.train();

		return 0;
	}
```

Then:
```
	$ time ./task3
	./task3  184.42s user 0.88s system 99% cpu 3:06.21 total
	>>> elapsed time 3m6s
```

## Using one output
Two simple changes:
1. Go to the top of `main.cpp` and change `NUM_OUTPUTS`
```cpp
	#define NUM_OUTPUTS 1
```
Make sure to change this to `10` if you switch back.

2. Go to the top of `NeuralNetwork.cpp` and change `ONE_OUTPUT`
```cpp
	#define ONE_OUTPUT 1
```


## Changing the activation function to `tanh`
Simply go to the top of `NeuralNetwork.cpp` and change `USE_TANH`
```cpp
	#define USE_TANH 1
```
