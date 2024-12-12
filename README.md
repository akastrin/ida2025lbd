# Make Literature-Based Discovery Great Again through Reproducible Pipelines

## Introduction

This repository provides a collection of Python notebooks that implement all the steps of the bisociative literature-based discovery (LBD) process: data acquisition, text preprocessing, discovery by term linking, and evaluation. The notebooks demonstrate traditional LBD approaches alongside our own bisociative LBD methods, which include (i) an ensemble-based approach for identifying cross-domain bridging terms, (ii) an outlier-based method that reduces the search space of linking terms to outlier documents, (iii) a link prediction approach for ranking associations, and (iv) an embedding-based technique for extracting semantic similarities between concepts/relations in a unified vector space.

## Notebooks overview

1. **Swanson's ABC closed discovery model** The notebook `01_closed_discovery.ipynb` illustrates the general principle of LBD based on Swanson's ABC closed discovery model, where A and C refer to the two domains in the A-C domain pair, and B refers to a set of potential bridging terms.

2. **Concept-based open discovery** In the notebook `02_open_discovery.ipynb`, we extend the closed discovery approach to the open discovery mode to generate new research hypotheses. We reproduce Swanson's discovery, which connects migraine headaches to magnesium. Specifically, we aim to identify novel therapeutic candidates (A) for migraine headaches (C). 

3. **Text mining-based closed discovery** CrossBee method represents documents using a bag-of-words technique, with term frequency-inverse document frequency weights used to model the importance of words and multiwords (n-grams). The method is implemented in the `03_mini_crossbee.ipynb` notebook.

4. **Outlier-based closed discovery** Outlier-based LBD methods for closed discovery narrow down the search for cross-domain bridging terms to outlier documents, thus simplifying the exploration of bridging terms. The approach is demonstrated in the `04_mini_ontogen.ipynb` notebook. 

5. **Outlier-based open discovery** The notebook `05_mini_rajolink.ipynb` illustrates the RaJoLink LBD procedure used in the open LBD setting, using Autism as the start domain of interest.

6. **Network-based closed discovery** The network-based approach extends the discovery of bridging terms to relation discovery, exploiting their potential to reveal links among knowledge concepts across different literature domains. The approach is illustrated in the `06 mini linkpred.ipynb` notebook.

## Authors

- Bojan Cestnik, Temida LLC 
- Andrej Kastrin, University of Ljubljana
- Boshko Koloski, Jožef Stefan Institute
- Nada Lavrač, Jožef Stefan Institute

## Acknowledgements

The authors acknowledge the financial support from the Slovenian Research and Innovation Agency through the Knowledge technologies (No. P2-0103), and Methodology for data analysis in medical sciences (No. P3-0154) core research projects, as well as Embeddings-based techniques for Media Monitoring Applications (No. L2-50070) research project. The young researcher grant (No. PR-12394) supported the work of Boshko Koloski.

## License

[MIT](https://choosealicense.com/licenses/mit/)
