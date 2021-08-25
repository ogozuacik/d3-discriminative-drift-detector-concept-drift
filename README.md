# Unsupervised Concept Drift Detection with a Discriminative Classifier (D3)

Ömer Gözüaçık, Alican Büyükçakır, Hamed Bonab, and Fazli Can. 2019. Unsupervised Concept Drift Detection with a Discriminative Classifier. In Proceedings of The 28th ACM International Conference on Information and Knowledge Management, Beijing, China, November 3–7, 2019, (CIKM’2019), 4 pages. [ACM DL](https://dl.acm.org/citation.cfm?id=3357384.3358144)

## Implementation of D3 in river package

D3 is now supported in **river** which is a merger between creme and scikit-multiflow. It is one of the most comprehensive python library for doing machine learning on streaming data. Right now, D3 is not included in the package, but you can download and install the development version of the package to access D3. The results may differ from the original implementation as the code is refactored according to library standards.

For more details on river: https://riverml.xyz/latest/

Forked version of river having D3: https://github.com/ogozuacik/river

**Installing development version of river which includes D3.**

```
pip install git+https://github.com/ogozuacik/river --upgrade
```

**Sample run**
```python
from river import synth
from river.drift import D3
d3 = D3()
# Simulate a data stream
data_stream = synth.Hyperplane(seed=42, n_features=10, mag_change=0.5)
# Update drift detector and verify if change is detected
i = 0
for x, y in data_stream.take(250):
   in_drift, in_warning = d3.update(x)
   if in_drift:
      print(f"Change detected at index {i}")
   i += 1
```


## Legacy Code

You can still download the old code in this repository and follow the instructions given below.

**Parameters:**
* w: window size
* rho: new data percentage
* tau: threshold for AUC

**Command line instructions:**

* python D3.py dataset_name w rho tau (sample: python D3.py elec.csv 100 0.1 0.7)

* You can either put the datasets into the same directory or write dataset directory in place of dataset_name.
Datasets should be in **csv** format. You can access the datasets used in the paper and more from:
  * https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow

* You have to install scikit-multiflow in addition to commonly used python libraries. (sklearn, pandas, numpy, matplotlib)
  * https://scikit-multiflow.github.io/

**The code will output:** 
* Final accuracy
* Total elapsed time (from beginning of the stream to the end)
* Prequential accuracy plot (dividing data stream into 30 chunks)
