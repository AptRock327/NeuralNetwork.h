#include <iostream>
#include <iomanip>
#include <vector>
#include "neural.h"

int main()
{
    NeuralNetwork example;

    //The AI will learn that it is supposed to give back exactly what was given.
    //There is little data, and hence the AI is pretty inaccurate (especially visible with the last number), but the more data one gives, the better the performance.
    std::vector<std::vector<float>> learning_inputs{{1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
	std::vector<std::vector<float>> learning_outputs{{1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

	std::vector<std::vector<float>> inputs{{1, 0, 0}};

	example.randomize_weights(learning_inputs[0].size(), learning_inputs.size());

	std::vector<std::vector<float>> training = example.learn(10000, learning_inputs, learning_outputs);
	std::vector<std::vector<float>> results = example.obtain_results(inputs);

    std::cout << "LEARNING RESULTS" << std::endl << std::endl;;
    for(int i = 0; i < 3; i++)
    {
        std::cout << "AI: ";
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::fixed << std::setprecision(2) << training[i][j] << ' ';
        }
        std::cout << "IDEAL: ";
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::fixed << std::setprecision(2) << learning_outputs[i][j] << ' ';
        }
        std::cout << std::endl;
    }

    std::vector<std::vector<float>> correct_results{{1, 0, 0}};

    std::cout << std::endl << "TESTING RESULTS" << std::endl << std::endl;;
    for(int i = 0; i < 1; i++)
    {
        std::cout << "AI: ";
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::fixed << std::setprecision(2) << results[i][j] << ' ';
        }
        std::cout << "IDEAL: ";
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::fixed << std::setprecision(2) << correct_results[i][j] << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}