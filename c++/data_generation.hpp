//#######################################
//# Copyright (C) 2019-2020 Otmar Ertl. #
//# All rights reserved.                #
//#######################################

#ifndef _DATA_GENERATION_HPP_
#define _DATA_GENERATION_HPP_

#include <vector>
#include <string>
#include <tuple>
#include <algorithm>

class Weights {

    const std::vector<std::tuple<double, double>> weights;
    const std::string latexDescription;
    const std::string id;

public:

    double getJn() const {
        
        double w0 = 0;
        double w1 = 0;
        for(const auto& w : weights) {
            w0 += std::get<0>(w);
            w1 += std::get<1>(w);
        }
        
        double d = 0;
        double n = 0;
        for(const auto& w : weights) {
            d += std::max(std::get<0>(w) / w0, std::get<1>(w) / w1);
            n += std::min(std::get<0>(w) / w0, std::get<1>(w) / w1);
        }
        return n / d;
    }
    
    double getJw() const {
        
        double d = 0;
        double n = 0;
        for(const auto& w : weights) {
            d += std::max(std::get<0>(w), std::get<1>(w));
            n += std::min(std::get<0>(w), std::get<1>(w));
        }
        return n / d;
    }
    
    double getJp() const {
        double result = 0;
        for(const auto& wi : weights) {
            double w0i = std::get<0>(wi);
            double w1i = std::get<1>(wi);
            double x = 0;
            for(const auto& wj : weights) {
                double w0j = std::get<0>(wj);
                double w1j = std::get<1>(wj);
                x += std::max(w0i * w1j, w0j * w1i);
            }
            result += (w0i * w1i) / x;
        }
        return result;
    }

    std::tuple<size_t, size_t, size_t> getSizes() const {
        size_t sizeA = 0;
        size_t sizeB = 0;
        size_t sizeAB = 0;
        for (const auto& w : weights) {
            if(std::get<0>(w) > 0) sizeA += 1;
            if(std::get<1>(w) > 0) sizeB += 1;
            if(std::get<0>(w) > 0 && std::get<1>(w) > 0) sizeAB += 1;
        }
        return std::make_tuple(sizeA, sizeB, sizeAB);
    }

    const std::vector<std::tuple<double, double>> getWeights() const {
        return weights;
    }

    const std::string& getLatexDescription() const {
        return latexDescription;
    }

    const std::string& getId() const {
        return id;
    }

    Weights(const std::vector<std::tuple<double, double>>& weights, const std::string& latexDescription, const std::string& id) : weights(weights), latexDescription(latexDescription), id(id) {}

    bool allWeightsZeroOrOne() const {
        for(const auto& w : weights) {
            if (
                (std::get<0>(w) != 0 && std::get<0>(w)!=1) ||
                (std::get<1>(w) != 0 && std::get<1>(w)!=1)) return false;
        }
        return true;
    }

};

template<typename R>
std::tuple<std::vector<std::tuple<uint64_t, double>>, std::vector<std::tuple<uint64_t, double>>> generateData(R& rng, const Weights& w) {

    const auto& resultSizes = w.getSizes();

    std::vector<std::tuple<uint64_t, double>> valuesA;
    std::vector<std::tuple<uint64_t, double>> valuesB;

    valuesA.reserve(std::get<0>(resultSizes));
    valuesB.reserve(std::get<1>(resultSizes));

    for(const auto& x : w.getWeights()) {
        uint64_t data = rng();
        if(std::get<0>(x) > 0) valuesA.emplace_back(data, std::get<0>(x));
        if(std::get<1>(x) > 0) valuesB.emplace_back(data, std::get<1>(x));
    }

    std::shuffle(valuesA.begin(), valuesA.end(), rng);
    std::shuffle(valuesB.begin(), valuesB.end(), rng);

    return std::make_tuple(valuesA, valuesB);
}

Weights getWeightsCase_075be894225e78f7() {
    std::vector<std::tuple<double,double>> v;
    v.emplace_back(3,20);
    v.emplace_back(30,7);
    return Weights(v, "$\\lbrace(3,20),(30,7)\\rbrace$","075be894225e78f7");
}

Weights getWeightsCase_dae81d77e5c7e0c3() {
    std::vector<std::tuple<double,double>> v;
    v.emplace_back(0,2);
    v.emplace_back(3,4);
    v.emplace_back(6,3);
    v.emplace_back(2,4);
    return Weights(v, "$\\lbrace(0,2),(3,4),(6,3),(2,4)\\rbrace$","dae81d77e5c7e0c3");
}

Weights getWeightsCase_52d5eb9e59e690e7() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i < 15; ++i) {
        v.emplace_back(4,2);
    }
    for (uint32_t i = 0; i < 10; ++i) {
        v.emplace_back(1,4);
    }
    for (uint32_t i = 0; i < 5; ++i) {
        v.emplace_back(12,0);
    }
    return Weights(v, "$\\lbrace(4,2)^{15},(1,4)^{10},(12,0)^{5}\\rbrace$","52d5eb9e59e690e7");
}

Weights getWeightsCase_83f19a65b7f42e88() {
    std::vector<std::tuple<double,double>> v;

    for (uint32_t i = 0; i <= 1000; ++i) {
        v.emplace_back(std::pow(1.001,i), std::pow(1.002, i));
    }
    return Weights(v, "$\\bigcup_{\\symTestCaseIndex=0}^{1000} \\lbrace({1.001}^\\symTestCaseIndex, {1.002}^{\\symTestCaseIndex})\\rbrace$","83f19a65b7f42e88");
}

Weights getWeightsCase_29baac0d70950228() {
    std::vector<std::tuple<double,double>> v;
    v.emplace_back(0,1);
    v.emplace_back(1,1);
    v.emplace_back(1,0);
    return Weights(v, "$\\lbrace(0,1),(1,0),(1,1)\\rbrace$","29baac0d70950228");
}

Weights getWeightsCase_4e8536ff3d0c07af() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i < 30; ++i) {
        v.emplace_back(0,1);
    }
    for (uint32_t i = 0; i < 10; ++i) {
        v.emplace_back(1,0);
    }
    for (uint32_t i = 0; i < 160; ++i) {
        v.emplace_back(1,1);
    }
    return Weights(v, "$\\lbrace(0,1)^{30},(1,0)^{10},(1,1)^{160}\\rbrace$","4e8536ff3d0c07af");
}

Weights getWeightsCase_ae7f50b05c6ea2dd() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i < 300; ++i) {
        v.emplace_back(0,1);
    }
    for (uint32_t i = 0; i < 500; ++i) {
        v.emplace_back(1,0);
    }
    for (uint32_t i = 0; i < 1200; ++i) {
        v.emplace_back(1,1);
    }
    return Weights(v, "$\\lbrace(0,1)^{300},(1,0)^{500},(1,1)^{1200}\\rbrace$","ae7f50b05c6ea2dd");
}

Weights getWeightsCase_0a92d95c38b0bec5() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i < 300; ++i) {
        v.emplace_back(1,3);
    }
    for (uint32_t i = 0; i < 500; ++i) {
        v.emplace_back(2,1);
    }
    for (uint32_t i = 0; i < 700; ++i) {
        v.emplace_back(5,4);
    }
    return Weights(v, "$\\lbrace(1,3)^{300},(2,1)^{500},(5,4)^{700}\\rbrace$","0a92d95c38b0bec5");
}

Weights getWeightsCase_a9415c152258dac1() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i <= 1000; ++i) {
        v.emplace_back(std::pow(0.999,i), std::pow(1.001, i));
    }
    return Weights(v, "$\\bigcup_{\\symTestCaseIndex=0}^{1000} \\lbrace({0.999}^\\symTestCaseIndex, {1.001}^{\\symTestCaseIndex})\\rbrace$","a9415c152258dac1");
}

Weights getWeightsCase_431c7f212064fc5d() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i <= 1000; ++i) {
        v.emplace_back(i, 1000-i);
    }
    return Weights(v, "$\\bigcup_{\\symTestCaseIndex=0}^{1000} \\lbrace(\\symTestCaseIndex, 1000-{\\symTestCaseIndex})\\rbrace$","431c7f212064fc5d");
}

Weights getWeightsCase_8d6bb210472266c3() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i <= 500; ++i) {
        v.emplace_back(std::pow(0.98,i), std::pow(1.01, i));
    }
    return Weights(v, "$\\bigcup_{\\symTestCaseIndex=0}^{500} \\lbrace({0.98}^\\symTestCaseIndex, {1.01}^{\\symTestCaseIndex})\\rbrace$","8d6bb210472266c3");
}

Weights getWeightsCase_8a224349623eeb24() {
    std::vector<std::tuple<double,double>> v;
    for (uint32_t i = 0; i <= 300; ++i) {
        v.emplace_back(std::pow(0.96,i), std::pow(1.01, i));
    }
    return Weights(v, "$\\bigcup_{\\symTestCaseIndex=0}^{300} \\lbrace({0.96}^\\symTestCaseIndex, {1.01}^{\\symTestCaseIndex})\\rbrace$","8a224349623eeb24");
}



#endif // _DATA_GENERATION_HPP_
