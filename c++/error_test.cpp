//#######################################
//# Copyright (C) 2019-2020 Otmar Ertl. #
//# All rights reserved.                #
//#######################################

#include "bitstream_random.hpp"
#include "minhash.hpp"
#include "data_generation.hpp"

#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

struct ExtractFunction {
    uint64_t operator()(const std::tuple<uint64_t,double>& d) const {
        return std::get<0>(d);
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

template<typename S>
void testCase(const Weights& w, const string& algorithmDescription, uint32_t m, uint64_t numIterations, S&& hashFunctionSupplier) {

    const auto dataSizes = w.getSizes();

    uint64_t seedSize = 256;

    // values from random.org
    seed_seq initialSeedSequence{
        UINT32_C(0xc9c5e41d), UINT32_C(0x14b77d0b), UINT32_C(0x78ff862e), UINT32_C(0x51d8975e),
        UINT32_C(0xe6dc72f6), UINT32_C(0x5f64d296), UINT32_C(0xe2946980), UINT32_C(0xea8615eb)};

    mt19937 initialRng(initialSeedSequence);
    vector<uint32_t> seeds(numIterations * seedSize);
    generate(seeds.begin(), seeds.end(), initialRng);

    vector<uint32_t> numEquals(m+1);

    #pragma omp parallel
    {
        auto h = hashFunctionSupplier(m);

        #pragma omp for
        for (uint64_t i = 0; i < numIterations; ++i) {

            seed_seq seedSequence(seeds.begin() + i * seedSize, seeds.begin() + (i + 1) * seedSize);
            mt19937_64 rng(seedSequence);

            const tuple<vector<tuple<uint64_t,double>>,vector<tuple<uint64_t,double>>> data = generateData(rng, w);

            const vector<tuple<uint64_t,double>>& d1 = get<0>(data);
            const vector<tuple<uint64_t,double>>& d2 = get<1>(data);

            assert(get<0>(dataSizes) == d1.size());
            assert(get<1>(dataSizes) == d2.size());

            std::vector<uint64_t> h1 = h(d1);
            std::vector<uint64_t> h2 = h(d2);
            
            uint32_t numEqual = 0;
            for (uint32_t j = 0; j < m; ++j)  {
                if (h1[j] == h2[j]) {
                    numEqual += 1;
                }
            }

            #pragma omp atomic
            numEquals[numEqual] += 1;
        }
    }

    cout << setprecision(numeric_limits< double >::max_digits10) << scientific;
    cout << w.getJw() << ";";
    cout << w.getJn() << ";";
    cout << w.getJp() << ";";
    cout << algorithmDescription << ";";
    cout << w.getLatexDescription() << ";";
    cout << numIterations << ";";
    cout << m << ";";
    cout << w.getId() << ";";
    cout << w.allWeightsZeroOrOne() << ";";
    cout << std::get<0>(dataSizes) << ";";
    cout << std::get<1>(dataSizes) << ";";
    cout << std::get<2>(dataSizes) << endl;

    bool first = true;
    for(uint32_t x : numEquals) {
        if (!first) {
            cout << ";";
        } else {
            first = false;
        }
        cout << x;
    }
    cout << endl;
    cout << flush;
}

void testCase(const Weights& w, uint32_t hashSize, uint64_t numIterations) {
    if (w.allWeightsZeroOrOne()) {
        testCase(w, "P-MinHash", hashSize, numIterations, [](uint32_t m){return PMinHash<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0xe48abf31a570087f));});
        testCase(w, "ProbMinHash1", hashSize, numIterations, [](uint32_t m){return ProbMinHash1<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x7845da8c2cb11b39));});
        testCase(w, "ProbMinHash1a", hashSize, numIterations, [](uint32_t m){return ProbMinHash1a<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x7845da8c2cb11b39));});
        testCase(w, "ProbMinHash2", hashSize, numIterations, [](uint32_t m){return ProbMinHash2<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0xc8d7cc411fc10f3d));});
        testCase(w, "NonStreamingProbMinHash2", hashSize, numIterations, [](uint32_t m){return NonStreamingProbMinHash2<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0xc8d7cc411fc10f3d));});
        if(hashSize > 1) testCase(w, "ProbMinHash3", hashSize, numIterations, [](uint32_t m){return ProbMinHash3<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x4e4b27c3824bd1e8));});
        if(hashSize > 1) testCase(w, "ProbMinHash3a", hashSize, numIterations, [](uint32_t m){return ProbMinHash3a<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x4e4b27c3824bd1e8));});
        if(hashSize > 1) testCase(w, "ProbMinHash4", hashSize, numIterations, [](uint32_t m){return ProbMinHash4<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x7f6ad5700f5c4cf0));});
        if(hashSize > 1) testCase(w, "NonStreamingProbMinHash4", hashSize, numIterations, [](uint32_t m){return NonStreamingProbMinHash4<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x7f6ad5700f5c4cf0));});
        testCase(w, "MinHash", hashSize, numIterations, [](uint32_t m){return MinHash<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x955e28fcedd50e0f));});
        testCase(w, "SuperMinHash", hashSize, numIterations, [](uint32_t m){return SuperMinHash<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0xd647a0e043225db3));});
        testCase(w, "OPH", hashSize, numIterations, [](uint32_t m){return OnePermutationHashingWithOptimalDensification<uint64_t, ExtractFunction, RNGFunction, RNGFunctionForSignatureComponents>(m, ExtractFunction(), RNGFunction(0x61af5b0a94498d39), RNGFunctionForSignatureComponents(0x17e56631727aac60));});
        //testCase(w, "HistoSketch", hashSize, numIterations, [](uint32_t m){return HistoSketch<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0xa9b7a09103ca956b));});
        //testCase(w, "0-bit Engineered", hashSize, numIterations, [](uint32_t m){return ZeroBitEngineered<uint64_t, ExtractFunction, RNGFunction>(m, ExtractFunction(), RNGFunction(0x77294d7eba3f85c1));});
    } else {
        testCase(w, "P-MinHash", hashSize, numIterations, [](uint32_t m){return PMinHash<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x60979c513666ee4b));});
        testCase(w, "ProbMinHash1", hashSize, numIterations, [](uint32_t m){return ProbMinHash1<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x897d7798a7629409));});
        testCase(w, "ProbMinHash1a", hashSize, numIterations, [](uint32_t m){return ProbMinHash1a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x897d7798a7629409));});
        testCase(w, "ProbMinHash2", hashSize, numIterations, [](uint32_t m){return ProbMinHash2<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x930bb771b6666420));});
        testCase(w, "NonStreamingProbMinHash2", hashSize, numIterations, [](uint32_t m){return NonStreamingProbMinHash2<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x930bb771b6666420));});
        if(hashSize > 1) testCase(w, "ProbMinHash3", hashSize, numIterations, [](uint32_t m){return ProbMinHash3<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0xa03cf8d39d2a03b8));});
        if(hashSize > 1) testCase(w, "ProbMinHash3a", hashSize, numIterations, [](uint32_t m){return ProbMinHash3a<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0xa03cf8d39d2a03b8));});
        if(hashSize > 1) testCase(w, "ProbMinHash4", hashSize, numIterations, [](uint32_t m){return ProbMinHash4<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x02f7adcab92fdcbb));});
        if(hashSize > 1) testCase(w, "NonStreamingProbMinHash4", hashSize, numIterations, [](uint32_t m){return NonStreamingProbMinHash4<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x02f7adcab92fdcbb));});
        //testCase(w, "HistoSketch", hashSize, numIterations, [](uint32_t m){return HistoSketch<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0x672a7d286a025f11));});
        //testCase(w, "0-bit Engineered", hashSize, numIterations, [](uint32_t m){return ZeroBitEngineered<uint64_t, ExtractFunction, RNGFunction, WeightFunction>(m, ExtractFunction(), RNGFunction(0xc8f40b415d4cf25f));});
    }
}

int main(int argc, char* argv[]) {
    cout << "Jw" << ";";
    cout << "Jn" << ";";
    cout << "Jp" << ";";
    cout << "algorithmDescription" << ";";
    cout << "caseDescription" << ";";
    cout << "numIterations" << ";";
    cout << "hashSize" << ";";
    cout << "caseId" << ";";
    cout << "isUnweighted" << ";";
    cout << "dataSizeA" << ";";
    cout << "dataSizeB" << ";";
    cout << "dataSizeAB";
    cout << endl;
    cout << "histogramEqualSignatureComponents";
    cout << endl;
    cout << flush;

    uint32_t hashSizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    uint64_t numIterations = 10000;

    vector<Weights> cases = {
        getWeightsCase_075be894225e78f7(),
        getWeightsCase_0a92d95c38b0bec5(), 
        getWeightsCase_29baac0d70950228(), 
        getWeightsCase_4e8536ff3d0c07af(), 
        getWeightsCase_52d5eb9e59e690e7(), 
        getWeightsCase_83f19a65b7f42e88(), 
        getWeightsCase_ae7f50b05c6ea2dd(),
        getWeightsCase_dae81d77e5c7e0c3(),
        getWeightsCase_a9415c152258dac1(),
        getWeightsCase_431c7f212064fc5d(),
        getWeightsCase_8d6bb210472266c3(),
        getWeightsCase_8a224349623eeb24()
        };

    for (uint32_t hashSize : hashSizes) {
        for (const Weights w : cases) {
            testCase(w, hashSize, numIterations);
        }
    }

    return 0;
}
