# d3-discriminative-drift-detector-concept-drift

Ömer Gözüaçık, Alican Büyükçakır, Hamed Bonab, and Fazli Can. 2019. Unsupervised Concept Drift Detection with a Discriminative Classifier. In Proceedings of The 28th ACM International Conference on Information and Knowledge Management, Beijing, China, November 3–7, 2019, (CIKM’2019), 4 pages. [arXiv] [ACM DL] (To be published)

**Parameters:**
* w: window size
* rho: new data percentage
* tau: threshold for AUC

**Command line instructions:**

python D3.py dataset_name w rho tau (sample: python D3.py elec.csv 100 0.1 0.7)

**The code will output:** 
* Final accuracy
* Total elapsed time (from beginning of the stream to the end)
* Prequential accuracy plot (dividing data stream into 30 chunks)
