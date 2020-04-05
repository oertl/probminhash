//#######################################
//# Copyright (C) 2019-2020 Otmar Ertl. #
//# All rights reserved.                #
//#######################################

#include "minhash.hpp"

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

using namespace std;

struct HashFunction {
    uint64_t operator()(uint64_t d) const {
        return d;
    }
};

class RngFunction {
    uint64_t seed;
public:

    RngFunction(uint64_t seed) : seed(seed) {}

    WyrandBitStream operator()(uint64_t d, uint64_t idx) const {
        return WyrandBitStream(d, idx, seed);
    }
};

class HashCombiner {
    uint64_t seed;
public:

    HashCombiner(uint64_t seed) : seed(seed) {}

    uint64_t operator()(const void* hashBuffer, uint64_t hashBufferSizeBytes) const {
        return wyhash(hashBuffer, hashBufferSizeBytes, seed);
    }
};

template<typename D, typename H>
void testCase(H&& h, uint64_t dataSize, uint32_t hashSize, uint8_t l, uint64_t numCycles, const D& testData, const string& algorithmLabel) {

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
    cout << static_cast<uint32_t>(l) << ";";
    cout << dataSize << ";";
    cout << avgHashTime << ";";
    cout << consumer << endl << flush;
}

template <typename GEN> void test(GEN& rng, uint32_t m, uint8_t l, uint64_t dataSize, uint64_t numCycles) {

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

    OrderMinHash orderMinHash(m, l, HashFunction(), RngFunction(UINT64_C(0xa4a90a84e7b77e99)), HashCombiner(UINT64_C(0x784a6768e4396b0f)));
    FastOrderMinHash1 fastOrderMinHash1(m, l, HashFunction(), RngFunction(UINT64_C(0xc1c3b5ab39077fea)), HashCombiner(UINT64_C(0xd29d6633ca07d772)));   // use same seed for FastOrderMinHash1 and 
    FastOrderMinHash1a fastOrderMinHash1a(m, l, HashFunction(), RngFunction(UINT64_C(0xc1c3b5ab39077fea)), HashCombiner(UINT64_C(0xd29d6633ca07d772))); // FastOrderMinHash1a to obtain identical results, because they are equivalent
    FastOrderMinHash2 fastOrderMinHash2(m, l, HashFunction(), RngFunction(UINT64_C(0xbff0c0d6ca84b838)), HashCombiner(UINT64_C(0xaea4b4aa74a50ad4)));

    testCase(orderMinHash, dataSize, m, l, numCycles, testData, "OrderMinHash");
    testCase(fastOrderMinHash1, dataSize, m, l, numCycles, testData, "FastOrderMinHash1");
    testCase(fastOrderMinHash1a, dataSize, m, l, numCycles, testData, "FastOrderMinHash1a");
    testCase(fastOrderMinHash2, dataSize, m, l, numCycles, testData, "FastOrderMinHash2");
}

int main(int argc, char* argv[]) {

    uint64_t numCycles = 100;    

    assert(argc==5);
    uint64_t seed = atol(argv[1]);
    uint32_t hashSize = atoi(argv[2]);
    uint64_t dataSize = atol(argv[3]);
    uint32_t l = atoi(argv[4]);

    assert(dataSize >= l);

    mt19937_64 rng(seed);

    test(rng, hashSize, l, dataSize, numCycles);

    return 0;
}
