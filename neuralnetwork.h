/// written by aonbel
///
/// 04.10.2023

#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <random>
#include <thread>
#include <future>
#include <chrono>

static std::mt19937 rnd(time(nullptr));
static const int DefaultSize = 16;
static const int MaxOffsetForB = 2;
static const int MaxOffsetForSynapses = 2;

long double ReturnRandomValueFrom(long double lowerBound, long double upperBound);

long double TakeRandomFromTwoAndAddMutation(long double firstParameter, long double secondParameter, long double mutationChance, long double mutationMax);

long double TakeRandomFromTwo(long double firstParameter, long double secondParameter);

class neuron
{
public:
	long double x = 0;
	long double b = 0;
};

class pair
{
public:
	long double left = 0;
	int right = 0;
};

class network
{
public:
	int numberOfInputNeurons;
	int numberOfLayers;
	int numberOfLayerNeurons;
	int numberOfEndNeurons;
	neuron* inputNeurons;
	neuron* outputNeurons;
	neuron** layerNeurons;
	long double** inputSynapses;
	long double*** layersSynapses;
	long double* answer;

	network();

	network(int numberOfInputNeurons, int numberOfLayers, int numberOfLayerNeurons, int numberOfEndNeurons);

	void SetWeightsRandom();

	void SetWeightsFromParents(network firstParent, network secondParent, long double mutationChance, long double mutationMax);

	void SetWeightsFromNetwork(network currNetwork);

	void SetWeightsFromFile(std::string file);

	void SetWeightsIntoFile(std::string file);

	void SetInput(long double* input, int sizeOfInput);

	long double* GetOutput();

	void Process();
};

class genetic
{
private:
	int numberOfNeuralNetworks;
	int numberOfInputNeurons;
	int numberOfLayers;
	int numberOfLayerNeurons;
	int numberOfEndNeurons;
	int numberToReproduce;
	int threadsCount;
	std::vector<network> networks;
	std::vector<network> newNetworks;
	std::vector<std::thread> threads;
public:
	genetic();
	
	genetic(int numberOfNeuralNetworks, int numberOfInputNeurons, int numberOfLayers, int numberOfLayerNeurons, int numberOfEndNeurons);

	void SetRandomWeights();

	void LearnAtSample(long double* input, int estimatedResult, int inputSize, int randomNetworksCount, long double mutationChance, long double mutationMax, long double evaluate(long double*, int));

	network* ReturnBestNetwork();
};

