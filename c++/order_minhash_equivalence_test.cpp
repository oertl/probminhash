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
#include <unordered_map>

using namespace std;

template<typename R>
pair<vector<uint64_t>, vector<uint64_t>> generate(const pair<vector<uint64_t>, vector<uint64_t>>& pattern, R& rng) {
    unordered_map<uint64_t, uint64_t> mapping;
    for(uint64_t x : pattern.first) {
        mapping[x] = rng();
    }
    for(uint64_t x : pattern.second) {
        mapping[x] = rng();
    }

    pair<vector<uint64_t>, vector<uint64_t>> result(pattern.first.size(), pattern.second.size());
    transform(pattern.first.cbegin(), pattern.first.cend(), result.first.begin(), [&mapping](uint64_t x) {return mapping[x];});
    transform(pattern.second.cbegin(), pattern.second.cend(), result.second.begin(), [&mapping](uint64_t x) {return mapping[x];});
    return result;
}

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

template<typename H, typename R>
void test(uint32_t m, H& hashAlgorithm, R&& rng, const pair<vector<uint64_t>, vector<uint64_t>>& pattern, uint64_t numIterations) {
    
    std::vector<uint32_t> histogram(m+1, UINT32_C(0));
    for(uint64_t i = 0; i < numIterations; ++i) {
        
        auto data = generate(pattern, rng);

        auto hash1 = hashAlgorithm(data.first);
        auto hash2 = hashAlgorithm(data.second);

        uint32_t numEqual = 0;
        for (uint32_t j = 0; j < m; ++j)  {
            if (hash1[j] == hash2[j]) {
                numEqual += 1;
            }
        }
        histogram[numEqual] += 1;
    }

    for(uint32_t i = 0; i < m; ++i) {
        cout << histogram[i] << " ";
    }
    cout << histogram[m] << endl << flush;
}

void test(uint32_t m, uint32_t l, const pair<vector<uint64_t>, vector<uint64_t>>& pattern, uint64_t numIterations) {

    OrderMinHash orderMinHash(m, l, HashFunction(), RngFunction(UINT64_C(0x38fc6dba5f56f705)), HashCombiner(UINT64_C(0xbbae2e9361244083)));
    FastOrderMinHash1 fastOrderMinHash1(m, l, HashFunction(), RngFunction(UINT64_C(0xf2f18c08823d6851)), HashCombiner(UINT64_C(0x8fcacb27aa2ef430)));   // use same seeds for FastOrderMinHash1 and 
    FastOrderMinHash1a fastOrderMinHash1a(m, l, HashFunction(), RngFunction(UINT64_C(0xf2f18c08823d6851)), HashCombiner(UINT64_C(0x8fcacb27aa2ef430))); // FastOrderMinHash1a to obtain identical results, because they are equivalent
    FastOrderMinHash2 fastOrderMinHash2(m, l, HashFunction(), RngFunction(UINT64_C(0xcf7355744a6e8145)), HashCombiner(UINT64_C(0xde10c59a9218c94a)));

    test(m, orderMinHash, mt19937_64(UINT64_C(0xe9d0a2ff7ade68cf)), pattern, numIterations);
    test(m, fastOrderMinHash1, mt19937_64(UINT64_C(0x90ca243e41d4c111)), pattern, numIterations);  // use same seed for FastOrderMinHash1 and 
    test(m, fastOrderMinHash1a, mt19937_64(UINT64_C(0x90ca243e41d4c111)), pattern, numIterations); // FastOrderMinHash1a to obtain identical results, because they are equivalent
    test(m, fastOrderMinHash2, mt19937_64(UINT64_C(0x251de5d8b2b3c529)), pattern, numIterations);
}

int main(int argc, char* argv[]) {
        
    pair<vector<uint64_t>, vector<uint64_t>> pattern1 {
        {0, 0, 1, 2}, 
        {0, 1, 1, 2}};

    pair<vector<uint64_t>, vector<uint64_t>> pattern2 {
        {0, 1, 2, 3, 4, 0, 1, 2, 3, 2, 4, 5}, 
        {0, 1, 2, 6, 4, 0, 7, 1, 2, 3, 2, 4, 5}};

    pair<vector<uint64_t>, vector<uint64_t>> pattern3 {
        {0, 1, 2, 3, 4, 0, 1, 2, 3, 2, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 6, 2, 4, 5}, 
        {0, 1, 2, 6, 4, 0, 7, 1, 2, 3, 2, 4, 5}};

    pair<vector<uint64_t>, vector<uint64_t>> pattern4;
    for(uint64_t i = 0; i < 100; ++i) {
        pattern4.first.push_back(i);
        pattern4.second.push_back(i + 30);
    }

    pair<vector<uint64_t>, vector<uint64_t>> pattern5;
    for(uint64_t i = 0; i < 25; ++i) {
        pattern5.first.push_back(i);
        pattern5.second.push_back(i + 5);
    }

    uint64_t numIterations = 100000;

    test(1024, 3, pattern1, numIterations);
    test(1024, 3, pattern2, numIterations);
    test(1024, 3, pattern3, numIterations);
    test(1024, 3, pattern4, numIterations);

    test(1024, 5, pattern2, numIterations);
    test(1024, 5, pattern3, numIterations);
    test(1024, 5, pattern4, numIterations);
    
    test(32, 3, pattern1, numIterations);
    test(32, 3, pattern2, numIterations);
    test(32, 3, pattern3, numIterations);
    test(32, 3, pattern4, numIterations);

    test(32, 5, pattern2, numIterations);
    test(32, 5, pattern3, numIterations);
    test(32, 5, pattern4, numIterations);
    
    test(4, 2, pattern5, numIterations);

    return 0;
}
