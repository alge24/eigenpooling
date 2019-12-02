# Graph Convolutional Networks with EigenPooling 
Pytorch implementation of [eigenpooling](https://arxiv.org/pdf/1904.13107.pdf). Some parts of the code are adapdted from the implementation of [diffpool](https://github.com/RexYing/diffpool).

For more details of the algorithm, please refer to our [paper](https://arxiv.org/pdf/1904.13107.pdf). If you find this work useful and use it in your research, please cite our paper.

```
@inproceedings{Ma:2019:GCN:3292500.3330982,
 author = {Ma, Yao and Wang, Suhang and Aggarwal, Charu C. and Tang, Jiliang},
 title = {Graph Convolutional Networks with EigenPooling},
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
 series = {KDD '19},
 year = {2019},
 isbn = {978-1-4503-6201-6},
 location = {Anchorage, AK, USA},
 pages = {723--731},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3292500.3330982},
 doi = {10.1145/3292500.3330982},
 acmid = {3330982},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {graph classification, graph convolution networks, pooling, spectral graph theory},
} 

```

#### Usage
Please check run_example.sh for an example of running the code.

#### Preprocessed datasets
You may download the preprocessed datasets [here](https://drive.google.com/open?id=1-8FrJxWFczCAnhOWVi9fq0SdwpA7pM_p) to save the time of preprocessing data.

#### Known Issue
Running on GPU may result in sub-optimal performance on some of the datasets inclduing ENZYMES, NCI1 and NCI109.