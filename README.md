# ProbMinHash – A Class of Locality-Sensitive Hash Algorithms for the (Probability) Jaccard Similarity

The revision with tag [results-published-in-tkde-paper](https://github.com/oertl/probminhash/tree/results-published-in-tkde-paper) was used to generate the results presented in the final paper, which is available at https://doi.ieeecomputersociety.org/10.1109/TKDE.2020.3021176 or as arXiv-preprint at https://arxiv.org/abs/1911.00675.

In addition to the algorithms presented in the paper, [minhash.hpp](https://github.com/oertl/probminhash/blob/master/c%2B%2B/minhash.hpp) contains the algorithms `NonStreamingProbMinHash2` and `NonStreamingProbMinHash4`, which are non-streaming equivalent variants of `ProbMinHash2` and `ProbMinHash4`. In a first pass they calculate the sum of all weights, which determines the distribution of the final stop limit. This allows to estimate an appropriate stop limit upfront. For example, if the stop limit is initialized to the 90-th percentile of this distribution, the processing can be stopped early even for the first elements for which the stop limit would otherwise be infinite. However, there is a 10% probability that the stop limit was chosen too small and the algorithm therefore fails. In this case the algorithm has to be restarted in this case with a larger stop limit. Nevertheless, the [performance results](https://github.com/oertl/probminhash/blob/master/paper/speed_charts.pdf) show that this approach can reduce the expected calculation time, provided that multiple passes over the data are allowed.

ProbMinHash is a locality-sensitive hash algorithm for the [probability Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index#Probability_Jaccard_similarity_and_distance). If a hash algorithm for the [weighted Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index#Weighted_Jaccard_similarity_and_distance) is needed, we recommend the use of [TreeMinHash](https://github.com/oertl/treeminhash) or [BagMinHash](https://github.com/oertl/bagminhash).








