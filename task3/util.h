#pragma once
#include <vector>

using namespace std;

template <typename T>
vector<vector<double>> normalizeImages(vector<vector<T>> const & training_images) {
	vector<vector<double>> result;
	for (vector<T> row : training_images) {
		result.push_back(vector<double>());
		for (T val : row) {
			result.rbegin()->push_back(((double)val - 125.0) / 255.0);
		}
	}
	return result;
}