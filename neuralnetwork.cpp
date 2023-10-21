#include "neuralnetwork.h"

#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <random>
#include <thread>
#include <future>
#include <chrono>

long double ReturnRandomValueFrom(long double lowerBound, long double upperBound)
{
	return ((long double)rnd() / rnd.max()) * (upperBound - lowerBound) + lowerBound;
}

long double TakeRandomFromTwoAndAddMutation(long double firstParameter, long double secondParameter, long double mutationChance, long double mutationMax)
{
	return ((ReturnRandomValueFrom(0, 1) < 0.5) ? firstParameter : secondParameter) + ((ReturnRandomValueFrom(0, 1) < mutationChance) ? (ReturnRandomValueFrom(0, 1) * mutationMax) : 0);
}

long double TakeRandomFromTwo(long double firstParameter, long double secondParameter)
{
	return TakeRandomFromTwoAndAddMutation(firstParameter, secondParameter, 0, 0);
}

network::network()
{
	network(DefaultSize, DefaultSize, DefaultSize, DefaultSize);
}

network::network(int numberOfInputNeurons, int numberOfLayers, int numberOfLayerNeurons, int numberOfEndNeurons)
{
	this->numberOfInputNeurons = numberOfInputNeurons;
	this->numberOfLayers = numberOfLayers;
	this->numberOfLayerNeurons = numberOfLayerNeurons;
	this->numberOfEndNeurons = numberOfEndNeurons;

	inputNeurons = new neuron[numberOfInputNeurons];
	outputNeurons = new neuron[numberOfEndNeurons];
	layerNeurons = new neuron * [numberOfLayers];
	answer = new long double[numberOfEndNeurons];

	for (int i = 0; i < numberOfLayers; i++)
	{
		layerNeurons[i] = new neuron[numberOfLayerNeurons];
	}

	inputSynapses = new long double* [numberOfInputNeurons];
	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		inputSynapses[i] = new long double[numberOfLayerNeurons];
	}

	layersSynapses = new long double** [numberOfLayers];
	for (int i = 0; i < numberOfLayers - 1; i++)
	{
		layersSynapses[i] = new long double* [numberOfLayerNeurons];

		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			layersSynapses[i][j] = new long double[numberOfLayerNeurons];
		}
	}

	layersSynapses[numberOfLayers - 1] = new long double* [numberOfLayerNeurons];
	for (int j = 0; j < numberOfLayerNeurons; j++)
	{
		layersSynapses[numberOfLayers - 1][j] = new long double[numberOfEndNeurons];
	}
}

void network::SetWeightsRandom()
{
	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		inputNeurons[i].b = ((long double)rnd() / rnd.max()) * MaxOffsetForB - MaxOffsetForB / 2;
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			inputSynapses[i][j] = ((long double)rnd() / rnd.max()) * MaxOffsetForSynapses - MaxOffsetForSynapses / 2;
		}
	}

	for (int i = 0; i < numberOfLayers - 1; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			layerNeurons[i][j].b = ((long double)rnd() / rnd.max()) * MaxOffsetForB - MaxOffsetForB / 2;

			for (int k = 0; k < numberOfLayerNeurons; k++)
			{
				layersSynapses[i][j][k] = ((long double)rnd() / rnd.max()) * MaxOffsetForSynapses - MaxOffsetForSynapses / 2;
			}
		}
	}

	for (int j = 0; j < numberOfLayerNeurons; j++)
	{
		layerNeurons[numberOfLayers - 1][j].b = ((long double)rnd() / rnd.max()) * MaxOffsetForB - MaxOffsetForB / 2;

		for (int k = 0; k < numberOfEndNeurons; k++)
		{
			layersSynapses[numberOfLayers - 1][j][k] = ((long double)rnd() / rnd.max()) * MaxOffsetForSynapses - MaxOffsetForSynapses / 2;
		}
	}

	for (int j = 0; j < numberOfEndNeurons; j++)
	{
		outputNeurons[j].b = ((long double)rnd() / rnd.max()) * MaxOffsetForB - MaxOffsetForB / 2;
	}
}

void network::SetWeightsFromParents(network firstParent, network secondParent, long double mutationChance, long double mutationMax)
{
	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		inputNeurons[i].b = TakeRandomFromTwoAndAddMutation(firstParent.inputNeurons[i].b, secondParent.inputNeurons[i].b, mutationChance, mutationMax);
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			inputSynapses[i][j] = TakeRandomFromTwoAndAddMutation(firstParent.inputSynapses[i][j], secondParent.inputSynapses[i][j], mutationChance, mutationMax);
		}
	}

	for (int i = 0; i < numberOfLayers - 1; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			layerNeurons[i][j].b = TakeRandomFromTwoAndAddMutation(firstParent.layerNeurons[i][j].b, secondParent.layerNeurons[i][j].b, mutationChance, mutationMax);

			for (int k = 0; k < numberOfLayerNeurons; k++)
			{
				layersSynapses[i][j][k] = TakeRandomFromTwoAndAddMutation(firstParent.layersSynapses[i][j][k], secondParent.layersSynapses[i][j][k], mutationChance, mutationMax);
			}
		}
	}

	for (int j = 0; j < numberOfLayerNeurons; j++)
	{
		layerNeurons[numberOfLayers - 1][j].b = TakeRandomFromTwoAndAddMutation(firstParent.layerNeurons[numberOfLayers - 1][j].b, secondParent.layerNeurons[numberOfLayers - 1][j].b, mutationChance, mutationMax);

		for (int k = 0; k < numberOfEndNeurons; k++)
		{
			layersSynapses[numberOfLayers - 1][j][k] = TakeRandomFromTwoAndAddMutation(firstParent.layersSynapses[numberOfLayers - 1][j][k], secondParent.layersSynapses[numberOfLayers - 1][j][k], mutationChance, mutationMax);
		}
	}

	/*for (int j = 0; j < numberOfEndNeurons; j++)
	{
		outputNeurons[j].b = TakeRandomFromTwoAndAddMutation(firstParent.outputNeurons[j].b, secondParent.outputNeurons[j].b, mutationChance, mutationMax);
	}
	*/
}

void network::SetWeightsFromNetwork(network currNetwork)
{
	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		inputNeurons[i].b = currNetwork.inputNeurons[i].b;
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			inputSynapses[i][j] = currNetwork.inputSynapses[i][j];
		}
	}

	for (int i = 0; i < numberOfLayers - 1; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			layerNeurons[i][j].b = currNetwork.layerNeurons[i][j].b;

			for (int k = 0; k < numberOfLayerNeurons; k++)
			{
				layersSynapses[i][j][k] = currNetwork.layersSynapses[i][j][k];
			}
		}
	}

	for (int j = 0; j < numberOfLayerNeurons; j++)
	{
		layerNeurons[numberOfLayers - 1][j].b = currNetwork.layerNeurons[numberOfLayers - 1][j].b;

		for (int k = 0; k < numberOfEndNeurons; k++)
		{
			layersSynapses[numberOfLayers - 1][j][k] = currNetwork.layersSynapses[numberOfLayers - 1][j][k];
		}
	}

	for (int j = 0; j < numberOfEndNeurons; j++)
	{
		outputNeurons[j].b = currNetwork.outputNeurons[j].b;
	}
}

void network::SetWeightsFromFile(std::string file)
{
	std::ifstream fin(file);

	fin >> numberOfInputNeurons >> numberOfLayers >> numberOfLayerNeurons >> numberOfEndNeurons;

	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		fin >> inputNeurons[i].b;
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			fin >> inputSynapses[i][j];
		}
	}

	for (int i = 0; i < numberOfLayers - 1; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			fin >> layerNeurons[i][j].b;

			for (int k = 0; k < numberOfLayerNeurons; k++)
			{
				fin >> layersSynapses[i][j][k];
			}
		}
	}

	for (int j = 0; j < numberOfLayerNeurons; j++)
	{
		fin >> layerNeurons[numberOfLayers - 1][j].b;

		for (int k = 0; k < numberOfEndNeurons; k++)
		{
			fin >> layersSynapses[numberOfLayers - 1][j][k];
		}
	}

	for (int j = 0; j < numberOfEndNeurons; j++)
	{
		fin >> outputNeurons[j].b;
	}

	fin.close();
}

void network::SetWeightsIntoFile(std::string file)
{
	std::ofstream fout(file);

	fout << numberOfInputNeurons << ' ' << numberOfLayers << ' ' << numberOfLayerNeurons << ' ' << numberOfEndNeurons << '\n';

	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		fout << inputNeurons[i].b << '\n';
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			fout << inputSynapses[i][j] << ' ';
		}

		fout << '\n';
	}

	for (int i = 0; i < numberOfLayers - 1; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			fout << layerNeurons[i][j].b << '\n';

			for (int k = 0; k < numberOfLayerNeurons; k++)
			{
				fout << layersSynapses[i][j][k] << ' ';
			}
		}
	}

	for (int j = 0; j < numberOfLayerNeurons; j++)
	{
		fout << layerNeurons[numberOfLayers - 1][j].b << '\n';

		for (int k = 0; k < numberOfEndNeurons; k++)
		{
			fout << layersSynapses[numberOfLayers - 1][j][k] << ' ';
		}
	}

	for (int j = 0; j < numberOfEndNeurons; j++)
	{
		fout << outputNeurons[j].b << ' ';
	}

	fout.flush();
	fout.close();
}

void network::SetInput(long double* input, int sizeOfInput)
{
	if (sizeOfInput != numberOfInputNeurons)
	{
		return;
	}

	for (int i = 0; i < sizeOfInput; i++) inputNeurons[i].x = input[i];
}

long double* network::GetOutput()
{
	for (int i = 0; i < numberOfEndNeurons; i++) answer[i] = outputNeurons[i].x;

	return answer;
}

void network::Process()
{
	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		inputNeurons[i].x = 0;
	}
	for (int i = 0; i < numberOfLayers; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			layerNeurons[i][j].x = 0;
		}
	}
	for (int i = 0; i < numberOfEndNeurons; i++)
	{
		outputNeurons[i].x = 0;
	}

	for (int i = 0; i < numberOfInputNeurons; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			layerNeurons[0][j].x += inputSynapses[i][j] * inputNeurons[i].x;
		}
	}

	for (int i = 0; i < numberOfLayers - 1; i++)
	{
		for (int j = 0; j < numberOfLayerNeurons; j++)
		{
			layerNeurons[i][j].x = 1. / (1. + exp(-layerNeurons[i][j].x + layerNeurons[i][j].b));

			for (int k = 0; k < numberOfLayerNeurons; k++)
			{
				layerNeurons[i + 1][k].x += layersSynapses[i][j][k] * layerNeurons[i][j].x;
			}
		}
	}

	for (int i = 0; i < numberOfLayerNeurons; i++)
	{
		layerNeurons[numberOfLayers - 1][i].x = 1. / (1. + exp(-layerNeurons[numberOfLayers - 1][i].x + layerNeurons[numberOfLayers - 1][i].b));

		for (int j = 0; j < numberOfEndNeurons; j++)
		{
			outputNeurons[j].x += layersSynapses[numberOfLayers - 1][i][j] * layerNeurons[numberOfLayers - 1][i].x;
		}
	}

	for (int j = 0; j < numberOfEndNeurons; j++)
	{
		outputNeurons[j].x = 1. / (1. + exp(-outputNeurons[j].b + outputNeurons[j].x));
	}
}

genetic::genetic()
{
	numberOfNeuralNetworks = 0;
	numberOfInputNeurons = 0;
	numberOfLayers = 0;
	numberOfLayerNeurons = 0;
	numberOfEndNeurons = 0;
	numberToReproduce = 2;
	threadsCount = std::thread::hardware_concurrency();
}

genetic::genetic(int numberOfNeuralNetworks, int numberOfInputNeurons, int numberOfLayers, int numberOfLayerNeurons, int numberOfEndNeurons)
{
	this->numberOfNeuralNetworks = numberOfNeuralNetworks;
	this->numberOfInputNeurons = numberOfInputNeurons;
	this->numberOfLayers = numberOfLayers;
	this->numberOfLayerNeurons = numberOfLayerNeurons;
	this->numberOfEndNeurons = numberOfEndNeurons;
	this->threadsCount = std::thread::hardware_concurrency();

	networks.resize(numberOfNeuralNetworks);
	newNetworks.resize(numberOfNeuralNetworks);
	threads.resize(threadsCount);

	for (int i = 0; i < numberOfNeuralNetworks; i++)
	{
		networks[i] = network(numberOfInputNeurons, numberOfLayers, numberOfLayerNeurons, numberOfEndNeurons);
		newNetworks[i] = network(numberOfInputNeurons, numberOfLayers, numberOfLayerNeurons, numberOfEndNeurons);
		networks[i].SetWeightsRandom();
	}
}

void genetic::SetRandomWeights()
{
	for (int i = 0; i < numberOfNeuralNetworks; i++)
	{
		networks[i].SetWeightsRandom();
	}
}

void genetic::LearnAtSample(long double* input, int estimatedResult, int inputSize, int randomNetworksCount, long double mutationChance, long double mutationMax, long double evaluate(long double*, int))
{
	if (inputSize != numberOfInputNeurons || numberOfNeuralNetworks < 2 + randomNetworksCount) return;

	std::vector<pair> all;

	long long currentThread = 0;

	for (int i = 0; i < numberOfNeuralNetworks; i++)
	{
		networks[i].SetInput(input, inputSize);

		while (threads[currentThread].joinable())
		{
			currentThread++;
			if (currentThread == threadsCount) currentThread = 0;
		}

		threads[currentThread] = std::thread(&network::Process, this->networks[i]);
		threads[currentThread].detach();
	}

	currentThread = 0;
	while (currentThread < threadsCount)
	{
		if (!threads[currentThread].joinable())
		{
			currentThread++;
		}
	}

	for (int i = 0; i < numberOfNeuralNetworks; i++)
	{
		all.push_back({ evaluate(networks[i].GetOutput(), estimatedResult), i });
	}

	std::sort(all.begin(), all.end(), [](pair a, pair b) {return a.left > b.left; });

	while (numberToReproduce * (numberToReproduce + 1) / 2 + randomNetworksCount < numberOfNeuralNetworks)
	{
		numberToReproduce++;
	}

	int currIterator = 0;

	currentThread = 0;
	for (int i = 0; i < numberToReproduce; i++)
	{
		for (int j = i + 1; j < numberToReproduce; j++)
		{
			while (threads[currentThread].joinable())
			{
				currentThread++;
				if (currentThread == threadsCount) currentThread = 0;
			}

			if (j != numberToReproduce - 1) threads[currentThread] = std::thread(&network::SetWeightsFromParents, this->newNetworks[currIterator], networks[all[i].right], networks[all[j].right], mutationChance * (1 - std::min(all[i].left, all[j].left)), mutationMax * (1 - std::min(all[i].left, all[j].left)));
			else threads[currentThread] = std::thread(&network::SetWeightsFromNetwork, this->newNetworks[currIterator], networks[i]);

			threads[currentThread].detach();

			currIterator++;
		}
	}

	while (currIterator < numberOfNeuralNetworks)
	{
		newNetworks[currIterator].SetWeightsRandom();

		currIterator++;
	}

	currentThread = 0;
	while (currentThread < threadsCount)
	{
		if (!threads[currentThread].joinable())
		{
			currentThread++;
		}
	}

	for (int i = 0; i < numberOfNeuralNetworks; i++) networks[i].SetWeightsFromNetwork(newNetworks[i]);
}

network* genetic::ReturnBestNetwork()
{
	if (numberOfNeuralNetworks != 0) return &networks[0];
	else return new network();
}
