//#######################################
//# Copyright (C) 2019-2020 Otmar Ertl. #
//# All rights reserved.                #
//#######################################

#ifndef _MINHASH_HPP_
#define _MINHASH_HPP_

#include "bitstream_random.hpp"
#include "exponential_distribution.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <numeric>

template <typename T>
class MaxValueTracker {
    const uint32_t m;
    const uint32_t lastIndex;
    const std::unique_ptr<T[]> values; 

public:
    MaxValueTracker(uint32_t m) : m(m), lastIndex((m << 1) - 2), values(new T[lastIndex+1]) {}

    void reset(const T& infinity) {
        std::fill_n(values.get(), lastIndex + 1, infinity);
    }

    bool update(uint32_t idx, T value) {
        assert(idx < m);
        if (value < values[idx]) {
            while(true) {
                values[idx] = value;
                const uint32_t parentIdx = m + (idx >> 1);
                if (parentIdx > lastIndex) break;
                const uint32_t siblingIdx = idx ^ UINT32_C(1);
                const T siblingValue = values[siblingIdx];
                if (!(siblingValue < values[parentIdx])) break;
                if (value < siblingValue) value = siblingValue;
                idx = parentIdx;
            }
            return true;
        }
        else {
            return false;
        }
    }

    bool isUpdatePossible(T value) const {
        return value < values[lastIndex];
    }
};

struct UnaryWeightFunction {
    template<typename X>
    constexpr double operator()(X) const {
        return 1;
    }
};

// Yang, Dingqi, et al. "HistoSketch: Fast Similarity-Preserving Sketching of Streaming Histograms with Concept Drift." 
// Proceedings of the IEEE International Conference on Data Mining (ICDM’17). 2017.
// https://exascale.info/assets/pdf/icdm2017_HistoSketch.pdf
template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class HistoSketch {
    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;
    const std::unique_ptr<double[]> values; 

    void reset() {
        std::fill_n(values.get(), m, std::numeric_limits<double>::infinity());
    }

public:

    HistoSketch(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : 
    m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), values(new double[m])  {}

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {

            double w = weightFunction(x);
            if (!( w > 0)) continue;
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            
            for (uint32_t k = 0; k < m; ++k) {
                double c = getGamma21(rng);
                double r = getGamma21(rng);
                double beta = getUniformDouble(rng);
                double y = std::exp(std::log(w) - r * beta);
                double a = c / (y * std::exp(r));
                if (a < values[k]) {
                    values[k] = a;
                    result[k] = d;
                }
            }
        }
        return result;
    }
};

// Equivalent, but much simpler version of HistoSketch
// also described as "P-MinHash" in
// 
//   Ryan Moulton, Yunjiang Jiang, 
//   "Maximally Consistent Sampling and the Jaccard Index of Probability Distributions", 2018
//     see https://openreview.net/forum?id=BkOswnc5z (published March 29, 2018)
//     see https://arxiv.org/abs/1809.04052 (published September 11, 2018)
//     see https://doi.org/10.1109/ICDM.2018.00050
//
//  and as "D^2 HistoSketch" in 
// 
//   D. Yang, B. Li, L. Rettig and P. Cudré-Mauroux, 
//   "D2 HistoSketch: Discriminative and Dynamic Similarity-Preserving Sketching of Streaming Histograms," 
//   in IEEE Transactions on Knowledge and Data Engineering, 2018, doi: 10.1109/TKDE.2018.2867468
//     see https://doi.ieeecomputersociety.org/10.1109/TKDE.2018.2867468
//
template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class PMinHash {
    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    const std::unique_ptr<double[]> values; 

    void reset() {
        std::fill_n(values.get(), m, std::numeric_limits<double>::infinity());
    }

public:

    PMinHash(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), values(new double[m])  {}

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {

            double w = weightFunction(x);
            if (!( w > 0)) continue;
            double wInv = 1. / w;
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);

            for (uint32_t k = 0; k < m; ++k) {
                double a = ziggurat::getExponential(rng) * wInv;
                if (a < values[k]) {
                    values[k] = a;
                    result[k] = d;
                }
            }
        }
        return result;
    }

};


// Raff, Edward, Jared Sylvester, and Charles Nicholas. "Engineering a Simplified 0-Bit Consistent Weighted Sampling." 
// Proceedings of the 27th ACM International Conference on Information and Knowledge Management, p. 1203-1212, (2018).
// https://doi.org/10.1145/3269206.3271690
// Attempt to simplify 0-bit minwise hashing. Due to a mistake they end up with something similar to HistoSketch.
// The resulting signatures can neither be used to estimate the weighted Jaccard index nor to estimate the Jaccard index for probability distributions J_p
template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class ZeroBitEngineered {
    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    const std::unique_ptr<double[]> values; 

    void reset() {
        std::fill_n(values.get(), m, std::numeric_limits<double>::infinity());
    }

public:

    ZeroBitEngineered(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), values(new double[m])  {}

    template<typename X>
    std::vector<D> operator()(const X& data) {
    
        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {

            double w = weightFunction(x);
            if (!( w > 0)) continue;
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            
            for (uint32_t k = 0; k < m; ++k) {
                double c = getGamma21(rng);
                double r = getGamma21(rng);
                double a = c / (w * std::exp(r));
                if (a < values[k]) {
                    values[k] = a;
                    result[k] = d;
                }            
            }
        }
        return result;
    }
};

template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class ProbMinHash1 {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;

    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    MaxValueTracker<double> q;

    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
    }

public:

    ProbMinHash1(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), q(m)  {}

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {
            #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
            double wInv;
            if constexpr(isWeighted) {
                double w = weightFunction(x);
                if (!( w > 0)) continue;
                wInv = 1. / w;
            }
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            
            double h;
            if constexpr(isWeighted) h = wInv * ziggurat::getExponential(rng); else h = ziggurat::getExponential(rng);
            while(q.isUpdatePossible(h)) {
                uint32_t k = getUniformLemire(m, rng);
                if (q.update(k, h)) {
                    result[k] = d;
                    if (!q.isUpdatePossible(h)) break;
                }
                if constexpr(isWeighted) h += wInv * ziggurat::getExponential(rng); else h += ziggurat::getExponential(rng);
            }
        }

        return result;
    }
};


template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class ProbMinHash1a {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;
    typedef typename std::conditional<isWeighted, std::tuple<D, double, typename std::result_of<R(D)>::type, double>, std::tuple<D, double, typename std::result_of<R(D)>::type>>::type bufferType;
    
    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    MaxValueTracker<double> q;
    std::vector<bufferType> buffer;
    uint64_t maxBufferSize;
    
    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
        buffer.clear();
    }

public:

    ProbMinHash1a(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), q(m)  {}

    uint64_t getMaxBufferSize() const {
        return maxBufferSize;
    }

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);
        
        for(const auto& x : data) {
            #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
            double wInv;
            if constexpr(isWeighted) {
                double w = weightFunction(x);
                if (!( w > 0)) continue;
                wInv = 1. / w;
            }
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            double h;
            if constexpr(isWeighted) h = wInv * ziggurat::getExponential(rng); else h = ziggurat::getExponential(rng);
            if (!q.isUpdatePossible(h)) continue;
            uint32_t k = getUniformLemire(m, rng);
            if (q.update(k, h)) {
                result[k] = d;
                if (!q.isUpdatePossible(h)) continue;
            }
            if constexpr(isWeighted) buffer.emplace_back(d, h, std::move(rng), wInv); else buffer.emplace_back(d, h, std::move(rng));

        }

        maxBufferSize = buffer.size();

        while(!buffer.empty()) {
            auto writeIt = buffer.begin();
            for(auto readIt = buffer.begin(); readIt != buffer.end(); ++readIt) {
                const auto& d = std::get<0>(*readIt);
                double h = std::get<1>(*readIt);
                auto& rng = std::get<2>(*readIt);
                #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
                double wInv;
                if constexpr(isWeighted) wInv = std::get<3>(*readIt);
                if (!q.isUpdatePossible(h)) continue;
                if constexpr(isWeighted) h += wInv * ziggurat::getExponential(rng); else h += ziggurat::getExponential(rng);
                if (!q.isUpdatePossible(h)) continue;
                uint32_t k = getUniformLemire(m, rng);
                if (q.update(k, h)) {
                    result[k] = d;
                    if (!q.isUpdatePossible(h)) continue;
                }
                if constexpr(isWeighted) *writeIt = std::make_tuple(d, h, std::move(rng), wInv); else *writeIt = std::make_tuple(d, h, std::move(rng));
                ++writeIt;
            }
            buffer.erase(writeIt, buffer.end());
        }

        return result;
    }
};

template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class ProbMinHash2 {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;

    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    MaxValueTracker<double> q;
    PermutationStream permutationStream;
    const std::unique_ptr<double[]> g;

    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
    }

public:

    ProbMinHash2(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), q(m), permutationStream(m), g(new double[m-1])  {
        for(uint32_t i = 1; i < m; ++i) {
            g[i - 1] = static_cast<double>(m) / static_cast<double>(m - i);
        }
    }

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {

            #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
            double wInv;
            if constexpr(isWeighted) {
                double w = weightFunction(x);
                if (!( w > 0)) continue;
                wInv = 1. / w;
            }
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            permutationStream.reset();
            
            double h;
            if constexpr(isWeighted) h = wInv * ziggurat::getExponential(rng); else h = ziggurat::getExponential(rng);
            uint32_t i = 0;
            while(q.isUpdatePossible(h)) {
                uint32_t k = permutationStream.next(rng);
                if (q.update(k, h)) {
                    result[k] = d;
                    if (!q.isUpdatePossible(h)) break;
                }
                if constexpr(isWeighted) h += (wInv * g[i]) * ziggurat::getExponential(rng); else h += g[i] * ziggurat::getExponential(rng);
                i += 1;
                assert(i < m);
            }
        }

        return result;
    }
};


template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class NonStreamingProbMinHash2 {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;

    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    PermutationStream permutationStream;
    const std::unique_ptr<double[]> g;
    const double initialLimitFactor;

public:

    NonStreamingProbMinHash2(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        extractFunction(extractFunction), 
        rngFunction(rngFunction), 
        weightFunction(weightFunction), 
        permutationStream(m), 
        g(new double[m-1]),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m)
    {
        for(uint32_t i = 1; i < m; ++i) {
            g[i - 1] = static_cast<double>(m) / static_cast<double>(m - i);
        }
    }

    template<typename X>
    std::vector<D> operator()(const X& data, uint64_t* iterationCounter = nullptr) {

        double weightSum;
        if constexpr(isWeighted) {    
            weightSum = std::accumulate(data.begin(), data.end(), 0., [this](double x, const auto& d) {return x + weightFunction(d);});
        } else {
            weightSum = data.size();
        }

        std::vector<D> result(m);

        const double limitIncrement = initialLimitFactor / weightSum;

        double limit = limitIncrement;

        std::vector<double> hashValues(m, limit);

        if (iterationCounter != nullptr) *iterationCounter = 1;

        while(true) {

            for(const auto& x : data) {

                #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
                double wInv;
                if constexpr(isWeighted) {
                    double w = weightFunction(x);
                    if (!( w > 0)) continue;
                    wInv = 1. / w;
                }
                const D& d = extractFunction(x);
                auto rng = rngFunction(d);
                permutationStream.reset();
                
                double h;
                if constexpr(isWeighted) h = wInv * ziggurat::getExponential(rng); else h = ziggurat::getExponential(rng);
                uint32_t i = 0;
                while(h < limit) {
                    uint32_t k = permutationStream.next(rng);
                    if (h < hashValues[k]) {
                        hashValues[k] = h;
                        result[k] = d;
                    }
                    if constexpr(isWeighted) h += (wInv * g[i]) * ziggurat::getExponential(rng); else h += g[i] * ziggurat::getExponential(rng);
                    i += 1;
                    if (i == m) break;
                }
            }

            bool success = std::none_of(hashValues.begin(), hashValues.end(), [limit](const auto& r){return r == limit;});

            if (success) return result;

            if (iterationCounter != nullptr) (*iterationCounter) += 1;
            double oldLimit = limit;
            limit += limitIncrement;
            std::for_each(hashValues.begin(), hashValues.end(), [oldLimit, limit](auto& d) {if (d == oldLimit) d = limit;});
        }
    }
};


template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class ProbMinHash3 {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;

    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    MaxValueTracker<double> q;
    TruncatedExponentialDistribution truncatedExponentialDistribution;

    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
    }

public:

    ProbMinHash3(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), q(m), truncatedExponentialDistribution(log1p(1./static_cast<double>(m-1))) {
        assert(m > 1);
    }

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {
            #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
            double wInv;
            if constexpr(isWeighted) {
                double w = weightFunction(x);
                if (!( w > 0)) continue;
                wInv = 1. / w;
            }
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);

            double h;
            if constexpr(isWeighted) h = wInv * truncatedExponentialDistribution(rng); else h = getUniformDouble(rng);
            uint32_t i = 1;
            while(q.isUpdatePossible(h)) {
                uint32_t k = getUniformLemire(m, rng);
                if (q.update(k, h)) result[k] = d;
                if constexpr(isWeighted) h = wInv * i; else h = i;
                if (!q.isUpdatePossible(h)) break;
                if constexpr(isWeighted) h += wInv * truncatedExponentialDistribution(rng); else h += getUniformDouble(rng);
                i += 1;
            }
        }

        return result;
    }
};

template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class ProbMinHash3a {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;
    typedef typename std::conditional<isWeighted, std::tuple<D, typename std::result_of<R(D)>::type, double>, std::tuple<D, typename std::result_of<R(D)>::type>>::type bufferType;

    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    MaxValueTracker<double> q;
    std::vector<bufferType> buffer;
    TruncatedExponentialDistribution truncatedExponentialDistribution;
    uint64_t maxBufferSize;
    
    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
        buffer.clear();
    }

public:

    ProbMinHash3a(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), weightFunction(weightFunction), q(m), truncatedExponentialDistribution(log1p(1./static_cast<double>(m-1)))  {
        assert(m > 1);
    }

    uint64_t getMaxBufferSize() const {
        return maxBufferSize;
    }

    template<typename X>
    typename std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);
        
        for(const auto& x : data) {
            #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
            double wInv;
            if constexpr(isWeighted) {
                double w = weightFunction(x);
                if (!( w > 0)) continue;
                wInv = 1. / w;
            }
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            double h;
            if constexpr(isWeighted) h = wInv * truncatedExponentialDistribution(rng); else h = getUniformDouble(rng);
            if (!q.isUpdatePossible(h)) continue;
            uint32_t k = getUniformLemire(m, rng);
            if (q.update(k, h)) result[k] = d;
            if constexpr(isWeighted) {
                if (!q.isUpdatePossible(wInv)) continue;
                buffer.emplace_back(d, std::move(rng), wInv);
            }
            else {
                if (!q.isUpdatePossible(1)) continue;
                buffer.emplace_back(d, std::move(rng));
            }
        }

        maxBufferSize = buffer.size();

        uint64_t i = 1;
        while(!buffer.empty()) {
            auto writeIt = buffer.begin();
            for(auto readIt = buffer.begin(); readIt != buffer.end(); ++readIt) {
                const auto& d = std::get<0>(*readIt);
                auto& rng = std::get<1>(*readIt);
                double wInv;
                double h;
                if constexpr(isWeighted) {
                    wInv = std::get<2>(*readIt);
                    h = i * wInv;
                }
                else {
                    h = i;
                }
                if (!q.isUpdatePossible(h)) continue;
                if constexpr(isWeighted) h += wInv * truncatedExponentialDistribution(rng); else h += getUniformDouble(rng);
                if (!q.isUpdatePossible(h)) continue;
                uint32_t k = getUniformLemire(m, rng);
                if (q.update(k, h)) result[k] = d;
                if constexpr(isWeighted) {
                    if (!q.isUpdatePossible((i + 1) * wInv)) continue;
                    *writeIt = std::make_tuple(d, std::move(rng), wInv);
                }
                else {
                    if (!q.isUpdatePossible(i + 1)) continue;
                    *writeIt = std::make_tuple(d, std::move(rng));
                }
                ++writeIt;
            }
            buffer.erase(writeIt, buffer.end());
            i += 1;
        }

        return result;
    }
};

template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class ProbMinHash4 {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;

    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    MaxValueTracker<double> q;
    PermutationStream permutationStream;

    const std::unique_ptr<double[]> boundaries;
    const std::unique_ptr<TruncatedExponentialDistribution[]> truncatedExponentialDistributions;
    double firstBoundaryInv;

    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
    }

public:

ProbMinHash4(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), 
            weightFunction(weightFunction), q(m), permutationStream(m), boundaries(new double[m-1]), truncatedExponentialDistributions(new TruncatedExponentialDistribution[m-1]) {
        assert(m > 1);
        const double firstBoundary = log1p(1./static_cast<double>(m-1));
        double previousBoundary = firstBoundary;
        truncatedExponentialDistributions[0] = TruncatedExponentialDistribution(firstBoundary);
        boundaries[0] = 1;
        for(uint32_t i = 1; i < m-1; ++i) {
            const double boundary = log1p(static_cast<double>(i + 1)/static_cast<double>(m - i - 1));
            boundaries[i] = boundary / firstBoundary;
            truncatedExponentialDistributions[i] = TruncatedExponentialDistribution(boundary - previousBoundary);
            previousBoundary = boundary;
        }
        firstBoundaryInv = 1. / firstBoundary;
    }

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {

            double wInv;
            if constexpr(isWeighted) {
                double w = weightFunction(x);
                if (!( w > 0)) continue;
                wInv = 1. / w;
            }
            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            permutationStream.reset();
            
            double h;
            if constexpr(isWeighted) h = wInv * truncatedExponentialDistributions[0](rng); else h = getUniformDouble(rng);
            uint32_t i = 1;
            while(q.isUpdatePossible(h)) {
                uint32_t k = permutationStream.next(rng);
                if (q.update(k, h)) result[k] = d;
                if constexpr(isWeighted) {
                    if (!q.isUpdatePossible(wInv * boundaries[i-1])) break; 
                }
                else {
                    if (!q.isUpdatePossible(i)) break; 
                }
                if (i < m - 1) {
                    if constexpr(isWeighted) h = wInv * (boundaries[i-1] + (boundaries[i] - boundaries[i-1]) * truncatedExponentialDistributions[i](rng)); else  h = i + getUniformDouble(rng);
                }
                else {
                    if constexpr(isWeighted) h = wInv * (boundaries[m-2] + firstBoundaryInv * ziggurat::getExponential(rng)); else h = (m - 1) + getUniformDouble(rng);
                    if (q.isUpdatePossible(h)) {
                        uint32_t k = permutationStream.next(rng);
                        q.update(k, h);
                        result[k] = d;
                    }
                    break;
                }
                i += 1;
            }
        }

        return result;
    }
};

template<typename D, typename E, typename R, typename W = UnaryWeightFunction>
class NonStreamingProbMinHash4 {
    constexpr static bool isWeighted = !std::is_same<W, UnaryWeightFunction>::value;

    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const W weightFunction;

    PermutationStream permutationStream;
    const std::unique_ptr<double[]> boundaries;
    const std::unique_ptr<TruncatedExponentialDistribution[]> truncatedExponentialDistributions;
    double firstBoundaryInv;
    const double initialLimitFactor;

public:

    NonStreamingProbMinHash4(const uint32_t m, E extractFunction = E(), R rngFunction = R(), W weightFunction = W(), double successProbabilityFirstRun = 0.9) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), 
            weightFunction(weightFunction), permutationStream(m), boundaries(new double[m-1]), truncatedExponentialDistributions(new TruncatedExponentialDistribution[m-1]), initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m) {
        assert(m > 1);
        const double firstBoundary = log1p(1./static_cast<double>(m-1));
        double previousBoundary = firstBoundary;
        truncatedExponentialDistributions[0] = TruncatedExponentialDistribution(firstBoundary);
        boundaries[0] = 1;
        for(uint32_t i = 1; i < m-1; ++i) {
            const double boundary = log1p(static_cast<double>(i + 1)/static_cast<double>(m - i - 1));
            boundaries[i] = boundary / firstBoundary;
            truncatedExponentialDistributions[i] = TruncatedExponentialDistribution(boundary - previousBoundary);
            previousBoundary = boundary;
        }
        firstBoundaryInv = 1. / firstBoundary;
    }

    template<typename X>
    std::vector<D> operator()(const X& data, uint64_t* iterationCounter = nullptr) {

        double weightSum;
        if constexpr(isWeighted) {    
            weightSum = std::accumulate(data.begin(), data.end(), 0., [this](double x, const auto& d) {return x + weightFunction(d);});
        } else {
            weightSum = data.size();
        }

        std::vector<D> result(m);

        const double limitIncrement = initialLimitFactor / weightSum;

        double limit = limitIncrement;

        std::vector<double> hashValues(m, limit);

        if (iterationCounter != nullptr) *iterationCounter = 1;

        while(true) {

            for(const auto& x : data) {

                double wInv;
                if constexpr(isWeighted) {
                    double w = weightFunction(x);
                    if (!( w > 0)) continue;
                    wInv = 1. / w;
                }
                const D& d = extractFunction(x);
                auto rng = rngFunction(d);
                permutationStream.reset();
                
                double h;
                if constexpr(isWeighted) h = wInv * truncatedExponentialDistributions[0](rng); else h = getUniformDouble(rng);
                uint32_t i = 1;
                while(h < limit) {
                    uint32_t k = permutationStream.next(rng);
                    if (h < hashValues[k]) {
                        hashValues[k] = h;
                        result[k] = d;
                    }
                    if constexpr(isWeighted) {
                        if (wInv * boundaries[i-1] >= limit) break; 
                    }
                    else {
                        if (i >= limit) break; 
                    }
                    if (i < m - 1) {
                        if constexpr(isWeighted) h = wInv * (boundaries[i-1] + (boundaries[i] - boundaries[i-1]) * truncatedExponentialDistributions[i](rng)); else  h = i + getUniformDouble(rng);
                    }
                    else {
                        if constexpr(isWeighted) h = wInv * (boundaries[m-2] + firstBoundaryInv * ziggurat::getExponential(rng)); else h = (m - 1) + getUniformDouble(rng);
                        if (h < limit) {
                            uint32_t k = permutationStream.next(rng);
                            if (h < hashValues[k]) {
                                hashValues[k] = h;
                                result[k] = d;
                            }
                        }
                        break;
                    }
                    i += 1;
                }
            }

            bool success = std::none_of(hashValues.begin(), hashValues.end(), [limit](const auto& r){return r == limit;});

            if (success) return result;

            if (iterationCounter != nullptr) (*iterationCounter) += 1;
            double oldLimit = limit;
            limit += limitIncrement;
            std::for_each(hashValues.begin(), hashValues.end(), [oldLimit, limit](auto& d) {if (d == oldLimit) d = limit;});
        }
    }
};


// An implementation of the original MinHash algorithm as described in
// Andrei Z. Broder. 1997. On the Resemblance and Containment of Documents. In Proc. Compression and Complexity of Sequences. 21–2 
// https://doi.org/10.1109/SEQUEN.1997.666900
template<typename D, typename E, typename R>
class MinHash {
    const uint32_t m; 
    const E extractFunction;
    const R rngFunction;

    const std::unique_ptr<uint64_t[]> values; 

    void reset() {
        std::fill_n(values.get(), m, std::numeric_limits<uint64_t>::max());
    }

public:

    MinHash(const uint32_t m, E extractFunction = E(), R rngFunction = R()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), values(new uint64_t[m])  {}

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);
        
        for(const auto& x : data) {

            const D& d = extractFunction(x);
            auto rng = rngFunction(d);

            for(uint32_t j = 0; j < m; ++j) {
                uint64_t r = getUniformPow2(64, rng);
                if (r < values[j]) {
                    values[j] = r;
                    result[j] = d;
                }
            }
        }

        return result;
    }
};


// An implementation of One Permutation Hashing with optimal densification as described in
// A. Shrivastava. Optimal densification for fast and accurate minwise
// hashing. In Proceedings of the 34th International Conference on
// Machine Learning (ICML), pages 3154–3163, 2017.
template<typename D, typename E, typename R, typename Q>
class OnePermutationHashingWithOptimalDensification {
    const uint32_t m;
    const E extractFunction;
    const R rngFunction;
    const Q rngFunctionForEmptyBinIndex;

    const std::unique_ptr<uint64_t[]> values; 

    void reset() {
        std::fill_n(values.get(), m, std::numeric_limits<uint64_t>::max());
    }

public:

    OnePermutationHashingWithOptimalDensification(const uint32_t m, E extractFunction = E(), R rngFunction = R(), Q rngFunctionForEmptyBinIndex = Q()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), rngFunctionForEmptyBinIndex(rngFunctionForEmptyBinIndex), values(new uint64_t[m])  {}

    template<typename X>
    std::vector<D> operator()(const X& data) {
        reset();
        std::vector<D> result(m);

        for(const auto& x : data) {

            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            uint32_t k = getUniformLemire(m, rng);
            uint64_t r = getUniformPow2(64, rng);
            if (r <= values[k]) {
                values[k] = r;
                result[k] = d;
            }
        }

        for(uint32_t k = 0; k < m; ++k) {
            if (values[k] == std::numeric_limits<uint64_t>::max()) {
                auto rngForEmptyBin = rngFunctionForEmptyBinIndex(k);
                uint32_t l;
                do {
                    l = getUniformLemire(m, rngForEmptyBin);
                } while (values[l] == std::numeric_limits<uint64_t>::max());
                result[k] = result[l];
            }
        }
        return result;
    }
};


// An implementation of the SuperMinHash algorithm as described in
// Otmar Ertl. "SuperMinHash - A New Minwise Hashing Algorithm for Jaccard Similarity Estimation"
// see https://arxiv.org/abs/1706.05698
template<typename D, typename E, typename R>
class SuperMinHash {
    const uint32_t m;
    const E extractFunction;
    const R rngFunction;

    const std::unique_ptr<uint64_t[]> values; 
    PermutationStream permutationStream;
    const std::unique_ptr<uint32_t[]> levels;
    const std::unique_ptr<uint32_t[]> levelHistogram;

    void reset() {
        std::fill_n(values.get(), m, std::numeric_limits<uint64_t>::max());
        std::fill_n(levels.get(), m, m-1);
        std::fill_n(levelHistogram.get(), m - 1, 0);
        levelHistogram[m-1] = m;
    }

public:

    SuperMinHash(const uint32_t m, E extractFunction = E(), R rngFunction = R()) : m(m), extractFunction(extractFunction), rngFunction(rngFunction), values(new uint64_t[m]), permutationStream(m),
        levels(new uint32_t[m]), levelHistogram(new uint32_t[m])  {}

    template<typename X>
    std::vector<D> operator()(const X& data) {

        reset();
        std::vector<D> result(m);
        uint32_t maxLevel = m-1;

        for(const auto& x : data) {

            const D& d = extractFunction(x);
            auto rng = rngFunction(d);
            permutationStream.reset();

            uint32_t j = 0;
            do {
                uint32_t k = permutationStream.next(rng);
                uint64_t r = getUniformPow2(64, rng);
                uint32_t& levelsP = levels[k];
                if (levelsP >= j) {
                    if (levelsP == j) {
                        if (r <= values[k]) {
                            values[k] = r;
                            result[k] = d;
                        }
                    }
                    else {
                        levelHistogram[levelsP] -= 1;
                        levelHistogram[j] += 1;
                        while(levelHistogram[maxLevel] == 0) maxLevel -= 1;
                        levelsP = j;
                        values[k] = r;
                        result[k] = d;
                    }
                }
                j += 1; 
            } while (j <= maxLevel);
        }

        return result;
    }
};

template<typename V>
class OrderMinhashHelper {
    const uint32_t l;
    const uint64_t ml;
    const std::unique_ptr<uint64_t[]> indices;
    const std::unique_ptr<V[]> values;
    const std::unique_ptr<uint64_t[]> hashBuffer;
    const uint64_t hashBufferSizeBytes;

public:

    OrderMinhashHelper(const uint32_t m, const uint32_t l) :         
        l(l), 
        ml(static_cast<uint64_t>(m) * static_cast<uint64_t>(l)), 
        indices(new uint64_t[ml]), 
        values(new V[ml]),
        hashBuffer(new uint64_t[l]),
        hashBufferSizeBytes(static_cast<uint64_t>(l) * sizeof(uint64_t)) {
            assert(l >= 1);
        }
    
    void update(uint64_t pos, V value, uint64_t elementIdx) {
        const uint64_t firstArrayIdx = static_cast<uint64_t>(l) * static_cast<uint64_t>(pos);
        const uint64_t lastArrayIdx = firstArrayIdx + (l - 1);

        if (value < values[lastArrayIdx]) {
            uint64_t arrayIdx = lastArrayIdx;
            while(arrayIdx > firstArrayIdx && value < values[arrayIdx - 1]) {
                values[arrayIdx] = values[arrayIdx - 1];
                indices[arrayIdx] = indices[arrayIdx - 1];
                arrayIdx -= 1;
            }
            values[arrayIdx] = value;
            indices[arrayIdx] = elementIdx;
        }
    }

    template<typename M>
    bool update(uint64_t pos, V value, uint64_t elementIdx, M& maxValueTracker) {
        const uint64_t firstArrayIdx = static_cast<uint64_t>(l) * static_cast<uint64_t>(pos);
        const uint64_t lastArrayIdx = firstArrayIdx + (l - 1);

        if (value < values[lastArrayIdx]) {
            uint64_t arrayIdx = lastArrayIdx;
            while(arrayIdx > firstArrayIdx && value < values[arrayIdx - 1]) {
                values[arrayIdx] = values[arrayIdx - 1];
                indices[arrayIdx] = indices[arrayIdx - 1];
                arrayIdx -= 1;
            }
            values[arrayIdx] = value;
            indices[arrayIdx] = elementIdx;
            return maxValueTracker.update(pos, values[lastArrayIdx]);
        } 
        else {
            return false;
        }
    }

    template<typename M>
    bool updateIfNewIndex(uint64_t pos, V value, const uint64_t elementIdx, M& maxValueTracker) {
        const uint64_t firstArrayIdx = static_cast<uint64_t>(l) * static_cast<uint64_t>(pos);
        const uint64_t lastArrayIdx = firstArrayIdx + (l - 1);

        if (value < values[lastArrayIdx]) {
            uint64_t idx = firstArrayIdx;
            while(idx < lastArrayIdx) {
                if(indices[idx] == elementIdx) return false;
                if(value < values[idx]) break;
                idx += 1;
            }

            uint64_t elementIdx2 = elementIdx;
            while(idx < lastArrayIdx) {
                std::swap(values[idx], value);
                std::swap(indices[idx], elementIdx2);
                if (elementIdx2 == elementIdx) return false;
                idx += 1;
            }
            values[lastArrayIdx] = value;
            indices[lastArrayIdx] = elementIdx2;
            return maxValueTracker.update(pos, value);
        } 
        else {
            return false;
        }
    }

    void resetValues(const V& infiniteValue) {
        std::fill_n(values.get(), ml, infiniteValue);
    }

    void resetIndices() {
        std::fill_n(indices.get(), ml, std::numeric_limits<uint64_t>::max());
    }

    template<typename X, typename H, typename C>
    std::vector<uint64_t> createSignature(const X& data, const H& hashFunction, const C& hashCombiner, uint32_t m) {
        std::vector<uint64_t> result(m);
        for(uint64_t i = 0, *start = indices.get(); i < m; i += 1, start += l) {
            std::sort(start, start + l);
            for(uint32_t j = 0; j < l; ++j) {
                hashBuffer[j] = hashFunction(data[*(start + j)]);
            }
            result[i] = hashCombiner(hashBuffer.get(), hashBufferSizeBytes);
        }
        return result;
    }

    uint32_t getMinimumDataSize() const {
        return l;
    }

};

class UniqueCounter {
    std::unordered_map<uint64_t, uint64_t> counts;
public:

    UniqueCounter(uint64_t dataSize) : counts(dataSize) {}

    uint64_t next(uint64_t e) {
        uint64_t& count = counts.insert({e, UINT64_C(0)}).first->second;
        return count++;
    }

    void reset() {
        counts.clear();
    }
};

// Marcais, Guillaume, et al. "Locality sensitive hashing for the edit distance." BioRxiv (2019): 534446.
template<typename H, typename R, typename C>
class OrderMinHash {
    const uint32_t m;
    const H hashFunction;
    const R rngFunction;
    const C hashCombiner;

    OrderMinhashHelper<uint64_t> orderMinhashHelper;

    void reset() {
        orderMinhashHelper.resetValues(std::numeric_limits<uint64_t>::max());
    }

public:

    OrderMinHash(const uint32_t m, const uint32_t l, H hashFunction = H(), R rngFunction = R(), C hashCombiner = C()) : 
        m(m),
        hashFunction(hashFunction),
        rngFunction(rngFunction),
        hashCombiner(hashCombiner),
        orderMinhashHelper(m,l) {}

    template<typename X>
    std::vector<uint64_t> operator()(const X& data) {

        uint64_t size = data.size();
        assert(size >= orderMinhashHelper.getMinimumDataSize());

        reset();

        UniqueCounter uniqueCounter(size);

        for(uint64_t idx = 0; idx < size; ++idx) {
            
            const uint64_t d = hashFunction(data[idx]);
            auto rng = rngFunction(d, uniqueCounter.next(d));

            for(uint32_t j = 0; j < m; ++j) {
                uint64_t r = getUniformPow2(64, rng);
                orderMinhashHelper.update(j, r, idx);
            }
        }

        return orderMinhashHelper.createSignature(data, hashFunction, hashCombiner, m);
    }
};



// An equivalent but faster version of OrderMinHash based on ProbMinHash1.
template<typename H, typename R, typename C>
class FastOrderMinHash1 {
    const uint32_t m;
    const H hashFunction;
    const R rngFunction;
    const C hashCombiner;

    MaxValueTracker<double> q;
    OrderMinhashHelper<double> orderMinhashHelper;

    std::vector<std::tuple<uint64_t, uint64_t, double, WyrandBitStream> > buffer;
    uint64_t maxBufferSize;
    std::unique_ptr<uint32_t[]> updatedRegisters;

    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
        orderMinhashHelper.resetValues(std::numeric_limits<double>::max());
        orderMinhashHelper.resetIndices();
        buffer.clear();
        std::fill_n(updatedRegisters.get(), m, std::numeric_limits<uint32_t>::max());
    }

public:

    FastOrderMinHash1(const uint32_t m, const uint32_t l, H hashFunction = H(), R rngFunction = R(), C hashCombiner = C()) : 
        m(m),
        hashFunction(hashFunction),
        rngFunction(rngFunction),
        hashCombiner(hashCombiner),
        q(m),
        orderMinhashHelper(m,l),
        updatedRegisters(new uint32_t[m]) {}

    uint64_t getMaxBufferSize() const {
        return maxBufferSize;
    }

    template<typename X>
    std::vector<uint64_t> operator()(const X& data) {

        uint64_t size = data.size();
        assert(size >= orderMinhashHelper.getMinimumDataSize());

        reset();

        UniqueCounter uniqueCounter(size);

        for(uint64_t idx = 0; idx < size; ++idx) {

            const uint64_t d = hashFunction(data[idx]);
            auto rng = rngFunction(d, uniqueCounter.next(d));
           
            double h = ziggurat::getExponential(rng);
            uint32_t numUpdatedRegisters = 0;
            while(q.isUpdatePossible(h)) {
                uint32_t k = getUniformLemire(m, rng);
                if (updatedRegisters[k] != idx) {
                    if(orderMinhashHelper.update(k, h, idx, q)) {
                        if (!q.isUpdatePossible(h)) break;   
                    }
                    updatedRegisters[k] = idx;
                    numUpdatedRegisters += 1;
                    if (numUpdatedRegisters == m) break;
                }
                h += ziggurat::getExponential(rng);
            }
        }

        return orderMinhashHelper.createSignature(data, hashFunction, hashCombiner, m);
    }
};

// An equivalent but faster version of OrderMinHash based on ProbMinHash1a.
template<typename H, typename R, typename C>
class FastOrderMinHash1a {
    const uint32_t m;
    const H hashFunction;
    const R rngFunction;
    const C hashCombiner;

    MaxValueTracker<double> q;
    OrderMinhashHelper<double> orderMinhashHelper;

    std::vector<std::tuple<uint64_t, double, typename std::result_of<R(uint64_t, uint64_t)>::type> > buffer;

    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
        orderMinhashHelper.resetValues(std::numeric_limits<double>::max());
        orderMinhashHelper.resetIndices();
        buffer.clear();
    }

public:

    FastOrderMinHash1a(const uint32_t m, const uint32_t l, H hashFunction = H(), R rngFunction = R(), C hashCombiner = C()) : 
        m(m),
        hashFunction(hashFunction),
        rngFunction(rngFunction),
        hashCombiner(hashCombiner),
        q(m),
        orderMinhashHelper(m,l) {}

    template<typename X>
    std::vector<uint64_t> operator()(const X& data) {

        uint64_t size = data.size();
        assert(size >= orderMinhashHelper.getMinimumDataSize());

        reset();

        UniqueCounter uniqueCounter(size);

        for(uint64_t idx = 0; idx < size; ++idx) {
            
            const uint64_t d = hashFunction(data[idx]);
            auto rng = rngFunction(d, uniqueCounter.next(d));

            double h = ziggurat::getExponential(rng);
            if (!q.isUpdatePossible(h)) continue;
            uint32_t k = getUniformLemire(m, rng);
            if (orderMinhashHelper.update(k, h, idx, q)) {
                if (!q.isUpdatePossible(h)) continue;
            }
            buffer.emplace_back(idx, h, std::move(rng));
        }

        while(!buffer.empty()) {
            auto writeIt = buffer.begin();
            for(auto readIt = buffer.begin(); readIt != buffer.end(); ++readIt) {
                uint64_t idx = std::get<0>(*readIt);
                double h = std::get<1>(*readIt);
                auto& rng = std::get<2>(*readIt);
                if (!q.isUpdatePossible(h)) continue;
                h += ziggurat::getExponential(rng);
                if (!q.isUpdatePossible(h)) continue;
                uint32_t k = getUniformLemire(m, rng);
                if (orderMinhashHelper.updateIfNewIndex(k, h, idx, q)) {
                    if (!q.isUpdatePossible(h)) continue;
                }
                *writeIt = std::make_tuple(idx, h, std::move(rng));
                ++writeIt;
            }
            buffer.erase(writeIt, buffer.end());
        }

        return orderMinhashHelper.createSignature(data, hashFunction, hashCombiner, m);
    }
};


// An equivalent but faster version of OrderMinHash based on ProbMinHash2.
template<typename H, typename R, typename C>
class FastOrderMinHash2 {
    const uint32_t m;
    const H hashFunction;
    const R rngFunction;
    const C hashCombiner;

    MaxValueTracker<double> q;
    OrderMinhashHelper<double> orderMinhashHelper;
    PermutationStream permutationStream;
    const std::unique_ptr<double[]> g;


    void reset() {
        q.reset(std::numeric_limits<double>::infinity());
        orderMinhashHelper.resetValues(std::numeric_limits<double>::max());
    }

public:

    FastOrderMinHash2(const uint32_t m, const uint32_t l, H hashFunction = H(), R rngFunction = R(), C hashCombiner = C()) : 
        m(m),
        hashFunction(hashFunction),
        rngFunction(rngFunction),
        hashCombiner(hashCombiner),
        q(m),
        orderMinhashHelper(m, l),
        permutationStream(m), 
        g(new double[m-1]) {
            for(uint32_t i = 1; i < m; ++i) {
                g[i - 1] = static_cast<double>(m) / static_cast<double>(m - i);
            }
        }

    template<typename X>
    std::vector<uint64_t> operator()(const X& data) {

        uint64_t size = data.size();
        assert(size >= orderMinhashHelper.getMinimumDataSize());

        reset();

        UniqueCounter uniqueCounter(size);

        for(uint64_t idx = 0; idx < size; ++idx) {

            const uint64_t d = hashFunction(data[idx]);
            auto rng = rngFunction(d, uniqueCounter.next(d));

            permutationStream.reset();
            
            double h = ziggurat::getExponential(rng);
            uint32_t i = 0;
            while(q.isUpdatePossible(h)) {
                uint32_t k = permutationStream.next(rng);
                if (orderMinhashHelper.update(k, h, idx, q)) {
                    if (!q.isUpdatePossible(h)) break;
                }                
                if (i + 1 >= m) break;
                h += ziggurat::getExponential(rng) * g[i];
                i += 1;
            }
        }

        return orderMinhashHelper.createSignature(data, hashFunction, hashCombiner, m);
    }
};


#endif // _MINHASH_HPP_