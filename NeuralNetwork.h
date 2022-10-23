//This is a simple library for neural networks without hidden layers written from scratch by Dominik Śliwiński

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

//The sigmoid function is used for normalization.
float sigmoid(float x)
{
	return 1/(1 + exp(-x));
}

//This function gives the value of the derivative of the sigmoid, given the output of the usual sigmoid function.
//This is possible because the sigmoid function satisfies the following differential equation: f'(x) = f(x)(1-f(x))
float derivative_from_sigmoid(float x)
{
	return x * (1 - x);
}

struct NeuralNetwork
{
	//The weights let the AI differentiate between important and unimportant traits.
	//"Machine learning" is really just the process of an AI doing trial and error until it finds the right weights.
	std::vector<std::vector<float>> weights;
	//Biases are here for the adjustement of the final value.
	std::vector<std::vector<float>> biases;

	//This function generates random weights between -1 and 1 that are used by the neural network.
	void randomize_weights(unsigned int input_size_a, unsigned int input_size_b)
	{
		std::srand(time(0)); //This starts the randomization process.
		std::vector<float> temp; //This is a vector we will store zeros in.
		//This loop fills weights with zeros, so it reaches its proper size, according to input_size_a and input_size_b.
		for(int i = 0; i < input_size_a; i++)
		{
			for(int j = 0; j < input_size_b; j++) temp.push_back(0);
			this->weights.push_back(temp);
		}


		//This assigns random weights.
		for(int i = 0; i < input_size_a; i++)
		{
			for(int j = 0; j < input_size_b; j++)
			{
				//rand() gives a float between 0 and 1, so its result is multiplied by 2 and 1 is subtracted to get a number between -1 and 1.
				this->weights[i][j] = (2 * static_cast<float>(rand() / static_cast<float>(RAND_MAX)) - 1);
			}
		}
	}

	//This function is responsible for machine learning (adapting the weights).
	std::vector<std::vector<float>> learn(unsigned int iterations, std::vector<std::vector<float>> data_input, std::vector<std::vector<float>> data_output)
	{
		//This section sets up some values.
		std::vector<float> temp;

		std::vector<std::vector<float>> output = data_output;
		std::vector<std::vector<float>> cost;
		for(int i = 0; i < data_input.size(); i++)
		{
			for(int j = 0; j < data_input[0].size(); j++) temp.push_back(0);
			cost.push_back(temp);
		}
		temp.clear();

		for(int i = 0; i < data_input.size(); i++)
		{
			for(int j = 0; j < data_input[0].size(); j++) temp.push_back(0);
			biases.push_back(temp);
		}

		//This is the main learning loop. The amount of iterations is given as an argument of this function.

		//Too little iterations will give little results, too much iterations will make the model too used to the training data itself (overfitting).
		for(int i = 0; i < iterations; i++)
		{
			//clear cost
			cost.clear();
			temp.clear();
			for(int i = 0; i < data_input.size(); i++)
			{
				for(int j = 0; j < data_input[0].size(); j++) temp.push_back(0);
				cost.push_back(temp);
			}
			//clear output
			output.clear();
			temp.clear();
			for(int i = 0; i < data_input.size(); i++)
			{
				for(int j = 0; j < data_input[0].size(); j++) temp.push_back(0);
				output.push_back(temp);
			}

			for(int j = 0; j < data_input.size(); j++)
			{
				//A specific output is set to whatever is the data at the point times a given weight (a weight that will be adapted).
				for(int x = 0; x < data_output[0].size(); x++)
				{
					for(int k = 0; k < data_input[j].size(); k++)
					{
						output[j][x] += data_input[j][k] * this->weights[x][k];
					}
				}
				//The biases are added, and then, each output is normalized.
				for(int x = 0; x < data_output[0].size(); x++)
				{
					output[j][x] = sigmoid(output[j][x] + this->biases[j][x]);
				}
			}

			//The cost is calculated, which tells us about how much the weights have to be adapted. It's a measure of how bad the answer is.
			for(int j = 0; j < data_output.size(); j++)
			{
				for(int x = 0; x < data_output[0].size(); x++)
				{
					cost[j][x] = data_output[j][x] - output[j][x];
				}
			}

			//This adapts the weights and biases to the cost.
			for(int j = 0; j < data_input[0].size(); j++)
			{
				for(int k = 0; k < data_input.size(); k++)
				{
					for(int x = 0; x < data_output[0].size(); x++) 
					{
						this->weights[x][j] += data_input[k][j] * cost[k][x] * derivative_from_sigmoid(output[k][x]);
						this->biases[x][j] += cost[k][x] * derivative_from_sigmoid(output[k][x]);
					}
				}
			}

		}
		return output; 
	}

	//This function is very similar, except that the learning elements are removed.
	std::vector<std::vector<float>> obtain_results(std::vector<std::vector<float>> data_input)
	{
		std::vector<float> temp;

		std::vector<std::vector<float>> output;

		for(int i = 0; i < data_input.size(); i++)
		{
			for(int j = 0; j < data_input[0].size(); j++) temp.push_back(0);
			output.push_back(temp);
		}

			for(int j = 0; j < data_input.size(); j++)
			{
				for(int x = 0; x < data_input[0].size(); x++)
				{
					for(int k = 0; k < data_input[j].size(); k++)
					{
						output[j][x] += data_input[j][k] * this->weights[x][k];
					}
				}
				for(int x = 0; x < data_input[0].size(); x++)
				{
					output[j][x] = sigmoid(output[j][x]);
				}
			}


		return output; 
	}
};
