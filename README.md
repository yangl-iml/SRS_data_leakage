## Don’t Get Ahead of Yourself: A Critical Study on Data Leakage in Offline Evaluation of Sequential Recommenders
This repository provides the source code and supplementary results for our LBR paper at RecSys2025. 

### Supplementary Results
We provide six addtional tables and one additional figure in [supplementary_results.pdf](supplementary_results.pdf) to further support our findings. 
* **Tables A1** and **A2** compare the performance of sequential recommenders with temporal LOO and split-by-timepoint LOO in nDCG@10 and recall@10, respectively. 
* **Tables A3** and **A4** compare the performance of sequential recommenders using temporal LOO with **subsampled training data** and split-by-timepoint LOO in nDCG@10 and recall@10 respectively. These results, along with those from **Tables A1** and **A2**, were used to generate results presented in Table 2 in Section 4.1.
* **Tables A5** and **A6** provide the full version (i.e. with results for all sequential recommenders) of Table 3 in Section 4.2 for nDCG@10 and recall@10, respectively.
* **Figure A1** is similar to Figure 2 in Section 4.3, showing how model rankings for sequential recommenders changes between temporal LOO and split-by-timepoint LOO with recall@10.
