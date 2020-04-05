//#######################################
//# Copyright (C) 2019-2020 Otmar Ertl. #
//# All rights reserved.                #
//#######################################

#include "bitstream_random.hpp"

#include <random>

using namespace std;


int main(int argc, char* argv[]) {

    mt19937_64 rng(UINT64_C(0x356fc7675f6cce28));

    uniform_int_distribution<uint8_t> dist(1, 64);

    WyrandBitStream s1(UINT64_C(0xc2881c5d6c802af7), (0x26a2b296e4474346));
    WyrandBitStream s2(UINT64_C(0xc2881c5d6c802af7), (0x26a2b296e4474346));

    uint64_t numIterations = 1000000;

    for(uint64_t i = 0; i < numIterations; ++i) {

        uint8_t numBits = dist(rng);

        uint64_t v1 = s1(numBits);
        uint64_t v2 = 0;
        for (uint8_t k = 0; k < numBits; ++k) {
            v2 <<= 1;
            v2 |= s2();
        }

        assert(v1 == v2);

    }
    
}