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
#include <cmath>

using namespace std;

struct ExtractFunction {
    uint64_t operator()(const std::tuple<uint64_t,double>& d) const {
        return std::get<0>(d);
    }

    uint64_t operator()(const uint64_t& d) const {
        return d;
    }
    
};

class RNGFunction {
    const uint64_t seed;
public:

    RNGFunction(uint64_t seed) : seed(seed) {}
    
    WyrandBitStream operator()(const uint64_t& x) const {
        return WyrandBitStream(x, seed);
    }
};
struct WeightFunction {
    double operator()(const std::tuple<uint64_t,double>& d) const {
        return std::get<1>(d);
    }
};

int main(int argc, char* argv[]) {

    uint64_t numCycles = 10000;
    uint64_t seedSize = 256;

    uint64_t numExamples = 8;

    double paretoShape2 = 2;
    double paretoShape0_5 = 0.5;

    assert(argc==4);
    uint64_t initialSeed = atol(argv[1]);
    uint32_t hashSize = atoi(argv[2]);
    uint64_t dataSize = atol(argv[3]);

    cout << numCycles << endl;
    cout << dataSize << endl;
    cout << "probMinHash1a" << ";";
    cout << "probMinHash1aWeightedExp" << ";";
    cout << "probMinHash1aWeightedPareto0_5" << ";";
    cout << "probMinHash1aWeightedPareto2" << ";";
    cout << "probMinHash3a" << ";";
    cout << "probMinHash3aWeightedExp" << ";";
    cout << "probMinHash3aWeightedPareto0_5" << ";";
    cout << "probMinHash3aWeightedPareto2" << endl << flush;

    mt19937 initialRng(initialSeed);
    vector<uint32_t> seeds(numCycles * seedSize);
    generate(seeds.begin(), seeds.end(), initialRng);
    vector<uint64_t> result(numExamples*numCycles, 0);

    #pragma omp parallel
    {
        auto probMinHash1a = ProbMinHash1a<uint64_t, ExtractFunction, RNGFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0x27e425d2009e3d13)));
        auto probMinHash1aWeightedExp = ProbMinHash1a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0x1d2563ce211c3bcb)), WeightFunction());
        auto probMinHash1aWeightedPareto0_5 = ProbMinHash1a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0x82b7ed987541049d)), WeightFunction());
        auto probMinHash1aWeightedPareto2 = ProbMinHash1a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0x5d449ffbfd9f1d8b)), WeightFunction());
        auto probMinHash3a = ProbMinHash3a<uint64_t, ExtractFunction, RNGFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0x70bf4e71d0f1683f)));
        auto probMinHash3aWeightedExp = ProbMinHash3a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0xac6dd080115e960d)), WeightFunction());
        auto probMinHash3aWeightedPareto0_5 = ProbMinHash3a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0x43c1b9412b264fab)), WeightFunction());
        auto probMinHash3aWeightedPareto2 = ProbMinHash3a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(hashSize, ExtractFunction(), RNGFunction(UINT64_C(0x808f01127ff99416)), WeightFunction());

        std::exponential_distribution<double> distributionExp(1.); 
        std::uniform_real_distribution<double> distributionUniform(0., 1.); 

        vector<uint64_t> data;
        vector<tuple<uint64_t, double>> dataExp;
        vector<tuple<uint64_t, double>> dataPareto0_5;
        vector<tuple<uint64_t, double>> dataPareto2;
        #pragma omp for
        for (uint64_t i = 0; i < numCycles; ++i) {
            
            seed_seq seedSequence(seeds.begin() + i * seedSize, seeds.begin() + (i + 1) * seedSize);
            mt19937_64 rng(seedSequence);
            
            data.clear();
            dataExp.clear();
            dataPareto0_5.clear();
            dataPareto2.clear();
            
            for (uint64_t j = 0; j < dataSize; ++j) data.push_back(rng());
            for (uint64_t j = 0; j < dataSize; ++j) dataExp.emplace_back(rng(), distributionExp(rng));
            for (uint64_t j = 0; j < dataSize; ++j) dataPareto0_5.emplace_back(rng(), pow(1 - distributionUniform(rng), -1./paretoShape0_5));
            for (uint64_t j = 0; j < dataSize; ++j) dataPareto2.emplace_back(rng(), pow(1 - distributionUniform(rng), -1./paretoShape2));

            probMinHash1a(data);
            probMinHash1aWeightedExp(dataExp);
            probMinHash1aWeightedPareto0_5(dataPareto0_5);
            probMinHash1aWeightedPareto2(dataPareto2);
            result[numExamples*i+0] = probMinHash1a.getMaxBufferSize();
            result[numExamples*i+1] = probMinHash1aWeightedExp.getMaxBufferSize();
            result[numExamples*i+2] = probMinHash1aWeightedPareto0_5.getMaxBufferSize();
            result[numExamples*i+3] = probMinHash1aWeightedPareto2.getMaxBufferSize();
            if (hashSize > 1) {
                probMinHash3a(data);
                probMinHash3aWeightedExp(dataExp);
                probMinHash3aWeightedPareto0_5(dataPareto0_5);
                probMinHash3aWeightedPareto2(dataPareto2);
                result[numExamples*i+4] = probMinHash3a.getMaxBufferSize();
                result[numExamples*i+5] = probMinHash3aWeightedExp.getMaxBufferSize();
                result[numExamples*i+6] = probMinHash3aWeightedPareto0_5.getMaxBufferSize();
                result[numExamples*i+7] = probMinHash3aWeightedPareto2.getMaxBufferSize();
            }
        }
    }

    for (uint64_t i = 0; i < numCycles; ++i) {
        for(uint64_t j = 0; j < numExamples - 1; ++j) cout << result[numExamples * i + j] << ";";
        cout << result[numExamples*i+numExamples - 1] << endl;
    }

    return 0;
}
