//#######################################
//# Copyright (C) 2019-2020 Otmar Ertl. #
//# All rights reserved.                #
//#######################################

#include "minhash.hpp"
#include "bitstream_random.hpp"

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

using namespace std;

struct ExtractFunction {
    uint64_t operator()(const uint64_t& d) const {
        return d;
    }
    uint64_t operator()(const std::tuple<uint64_t,double>& d) const {
        return std::get<0>(d);
    }
};

class RNGFunction {
    const uint64_t seed;
public:

    RNGFunction(uint64_t seed) : seed(seed) {}
    
    WyrandBitStream operator()(uint64_t x) const {
        return WyrandBitStream(x, seed);
    }
};

class RNGFunctionForSignatureComponents {
    const uint64_t seed;
public:

    RNGFunctionForSignatureComponents(uint64_t seed) : seed(seed) {}
    
    WyrandBitStream operator()(uint32_t x) const {
        return WyrandBitStream(x, seed);
    }
};

struct WeightFunction {
    double operator()(const std::tuple<uint64_t,double>& d) const {
        return std::get<1>(d);
    }
};

template<typename H, typename D>
void testCase(H& h, uint64_t dataSize, uint32_t hashSize, uint64_t numCycles, const D& testData, const string& algorithmLabel, const string& distributionLabel) {

    assert(numCycles = testData.size());

    uint64_t consumer = 0;
    chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
    for (const auto& data :  testData) {
        vector<uint64_t> result = h(data);
        for(uint64_t r : result) consumer ^= r;
    }
    chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
    double avgHashTime = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart).count() / numCycles;

    cout << setprecision(numeric_limits< double >::max_digits10) << scientific;
    cout << algorithmLabel << ";";
    cout << numCycles << ";";
    cout << hashSize << ";";
    cout << dataSize << ";";
    cout << avgHashTime << ";";
    cout << distributionLabel << ";";
    cout << consumer << endl << flush;
}

template<template<typename, typename, typename, typename> typename H, typename D, typename GEN>
void testWeightedCase(GEN& rng, uint64_t dataSize, uint32_t hashSize, uint64_t numCycles, const D& testData, const string& algorithmLabel, const string& distributionLabel) {
    H<uint64_t, ExtractFunction, RNGFunction, WeightFunction> h(hashSize, ExtractFunction(), RNGFunction(rng()));
    testCase(h, dataSize, hashSize, numCycles, testData, algorithmLabel, distributionLabel);
}

template<template<typename, typename, typename> typename H, typename D, typename GEN>
void testUnweightedCase(GEN& rng, uint64_t dataSize, uint32_t hashSize, uint64_t numCycles, const D& testData, const string& algorithmLabel) {
    H<uint64_t, ExtractFunction, RNGFunction> h(hashSize, ExtractFunction(), RNGFunction(rng()));
    testCase(h, dataSize, hashSize, numCycles, testData, algorithmLabel, "unweighted");
}

template<typename D, typename GEN>
void testUnweightedCaseOnePermutationHashingWithOptimalDensification(GEN& rng, uint64_t dataSize, uint32_t hashSize, uint64_t numCycles, const D& testData, const string& algorithmLabel) {
    OnePermutationHashingWithOptimalDensification<uint64_t, ExtractFunction, RNGFunction, RNGFunctionForSignatureComponents> h(hashSize, ExtractFunction(), RNGFunction(rng()), RNGFunctionForSignatureComponents(rng()));
    testCase(h, dataSize, hashSize, numCycles, testData, algorithmLabel, "unweighted");
}

 
template <typename GEN> double generatePareto(GEN& rng, double scale, double shape) {
    std::uniform_real_distribution<double> distributionUniform(0., 1.); 
    return scale * pow(1 - distributionUniform(rng), -1./shape);
}

template <typename GEN> void test(GEN& rng, uint32_t hashSize,uint64_t dataSize, uint64_t numCycles) {

    {
        const string distributionLabel = "exp(1)";
        
        // generate test data
        std::exponential_distribution<double> distribution(1.);
        vector<vector<tuple<uint64_t, double>>> testData(numCycles);
        for (uint64_t i = 0; i < numCycles; ++i) {
            vector<tuple<uint64_t, double>> d(dataSize);
            for (uint64_t j = 0; j < dataSize; ++j) {
                uint64_t data = rng();
                double weight = distribution(rng);
                d[j] = make_tuple(data, weight);
            }
            testData[i] = d;
        }

        testWeightedCase<PMinHash>(rng, dataSize, hashSize, numCycles, testData, "P-MinHash", distributionLabel);
        testWeightedCase<ProbMinHash1>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1", distributionLabel);
        testWeightedCase<ProbMinHash1a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1a", distributionLabel);
        testWeightedCase<ProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash2", distributionLabel);
        testWeightedCase<NonStreamingProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash2", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash3>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash3a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3a", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash4", distributionLabel);
        if(hashSize > 1) testWeightedCase<NonStreamingProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash4", distributionLabel);
    }
    {
        const string distributionLabel = "pareto(1,0.5)";

        // generate test data
        vector<vector<tuple<uint64_t, double>>> testData(numCycles);
        for (uint64_t i = 0; i < numCycles; ++i) {
            vector<tuple<uint64_t, double>> d(dataSize);
            for (uint64_t j = 0; j < dataSize; ++j) {
                uint64_t data = rng();
                double weight = generatePareto(rng, 1, 0.5);
                d[j] = make_tuple(data, weight);
            }
            testData[i] = d;
        }

        testWeightedCase<PMinHash>(rng, dataSize, hashSize, numCycles, testData, "P-MinHash", distributionLabel);
        testWeightedCase<ProbMinHash1>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1", distributionLabel);
        testWeightedCase<ProbMinHash1a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1a", distributionLabel);
        testWeightedCase<ProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash2", distributionLabel);
        testWeightedCase<NonStreamingProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash2", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash3>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash3a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3a", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash4", distributionLabel);
        if(hashSize > 1) testWeightedCase<NonStreamingProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash4", distributionLabel);
    }
    {
        const string distributionLabel = "pareto(1,2)";
        
        // generate test data
        vector<vector<tuple<uint64_t, double>>> testData(numCycles);
        for (uint64_t i = 0; i < numCycles; ++i) {
            vector<tuple<uint64_t, double>> d(dataSize);
            for (uint64_t j = 0; j < dataSize; ++j) {
                uint64_t data = rng();
                double weight = generatePareto(rng, 1, 2);
                d[j] = make_tuple(data, weight);
            }
            testData[i] = d;
        }

        testWeightedCase<PMinHash>(rng, dataSize, hashSize, numCycles, testData, "P-MinHash", distributionLabel);
        testWeightedCase<ProbMinHash1>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1", distributionLabel);
        testWeightedCase<ProbMinHash1a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1a", distributionLabel);
        testWeightedCase<ProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash2", distributionLabel);
        testWeightedCase<NonStreamingProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash2", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash3>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash3a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3a", distributionLabel);
        if(hashSize > 1) testWeightedCase<ProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash4", distributionLabel);
        if(hashSize > 1) testWeightedCase<NonStreamingProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash4", distributionLabel);
    }
    { // unweighted case (weights constant 1)

        // generate test data
        vector<vector<uint64_t>> testData(numCycles);
        for (uint64_t i = 0; i < numCycles; ++i) {
            vector<uint64_t> d(dataSize);
            for (uint64_t j = 0; j < dataSize; ++j) {
                uint64_t data = rng();
                d[j] = data;
            }
            testData[i] = d;
        }

        testUnweightedCase<PMinHash>(rng, dataSize, hashSize, numCycles, testData, "P-MinHash");
        testUnweightedCase<ProbMinHash1>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1");
        testUnweightedCase<ProbMinHash1a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash1a");
        testUnweightedCase<ProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash2");
        testUnweightedCase<NonStreamingProbMinHash2>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash2");
        if(hashSize > 1) testUnweightedCase<ProbMinHash3>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3");
        if(hashSize > 1) testUnweightedCase<ProbMinHash3a>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash3a");
        if(hashSize > 1) testUnweightedCase<ProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "ProbMinHash4");
        if(hashSize > 1) testUnweightedCase<NonStreamingProbMinHash4>(rng, dataSize, hashSize, numCycles, testData, "NonStreamingProbMinHash4");
        testUnweightedCase<SuperMinHash>(rng, dataSize, hashSize, numCycles, testData, "SuperMinHash");
        testUnweightedCase<MinHash>(rng, dataSize, hashSize, numCycles, testData, "MinHash");
        testUnweightedCaseOnePermutationHashingWithOptimalDensification(rng, dataSize, hashSize, numCycles, testData, "OPH");
    }
}

int main(int argc, char* argv[]) {

    uint64_t numCycles = 100;

    assert(argc==4);
    uint64_t seed = atol(argv[1]);
    uint32_t hashSize = atoi(argv[2]);
    uint64_t dataSize = atol(argv[3]);

    mt19937_64 rng(seed);

    test(rng, hashSize, dataSize, numCycles);

    return 0;
}
