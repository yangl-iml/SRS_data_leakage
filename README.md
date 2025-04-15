## Temporal Consistency and Data Leakage in Offline Evaluation of Sequential Recommender Systems
This anonymous repository provides the source code and supplementary results for our submission at RecSys2025. 

### Supplementary Results
We provide seven addtional tables and one additional figure in [supplementary_results.pdf](supplementary_results.pdf) to further support our findings. 
* **Table A1** presents similar results to Table 2 in Section 5.1, but for recall@10.
* **Tables A2** and **A3** provide the full version (i.e. with results for all sequential recommenders) of Table 3 in Section 5.2 for nDCG@10 and recall@10, respectively.
* **Tables A4** and **A5** compare the performance of sequential recommenders using temporal LOO with **subsampled training data** and split-by-timepoint LOO in nDCG@10 and recall@10 respectively. These results, along with those from Table 2 and Tabel A1, were used to generate results presented in Table 4 in Section 5.2.
* **Table A6** shows the results for general recommenders with temporal LOO omitted from Section 5.3.
* **Table A7** presents the results for general recommenders with split-by-timepoint LOO for recall@10, corresponding to Table 5 in Section 5.3.
* **Figure A1** is similar to Figure 2 in Section 5.1, showing how model rankings for sequential recommenders changes between temporal LOO and split-by-timepoint LOO with recall@10.
