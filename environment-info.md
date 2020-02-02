# Build Environment Specifications

## OS Information
Tested on two systems:

	Distributor ID:	Ubuntu
	Description:	Ubuntu 16.04.3 LTS
	Release:	16.04
	Codename:	xenial
and

	  System Version: macOS 10.14 (18A391)
      Kernel Version: Darwin 18.0.0

## cmake
	cmake version 3.5.1

## Compiler 
For Ubuntu:

	gcc (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609
	g++ (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609

For Mac:

	Apple LLVM version 10.0.1 (clang-1001.0.46.4)
	Target: x86_64-apple-darwin18.0.0
	Thread model: posix
Note: `clang` is aliased as `gcc` and `g++` by default on Macs with the XCode toolchain 
installed, so this is the output of `g++ --version`. However, as long as the `--std=c++0x` 
flag is set, the code should compile fine with GNU `g++`.

