# Transparent and Adaptive Pruning of Hoeffding Trees

This repository contains the official code for our workshop paper.

**"Transparent and Adaptive Pruning of Hoeffding Trees"**
*Submitted to the 2nd TempXAI Workshop on Explainable AI in Time Series and Data Streams, ECML-PKDD 2025*

## Abstract
Explainable artificial intelligence has gained significant attention, with decision trees playing a key role due to their interpretability. However, incremental decision trees, namely Hoeffding trees (HT), widely used for efficient and transparent data stream processing, suffer from unbounded growth.
Existing adaptive methods address this but overlook transparency.
We introduce **Pr**uning H**oe**ffding Trees by the **I**mportance of **F**eatures (ProeFI), a novel approach that, in a transparent manner, prunes HT to mitigate unbounded growth and enhance adaptability to evolving data.
ProeFI employs incremental permutation feature importance and a self-adaptive threshold to dynamically refine its pruning process in response to drifting data distributions.
Experimental results show ProeFI achieves comparable performance to state-of-the-art methods while maintaining similar tree complexity. Our method outperforms existing techniques in balancing predictive performance and complexity.

## Implementation
- ProeFI
  - Fixed or adaptive threshold
- Incremental PFI with seeds for reproducibility
  - Only for the sampling strategies and marginal imputer
- Modified datasets used in the experiments.

The implementation is based on [river](https://github.com/online-ml/river) and [ixai](https://github.com/mmschlk/iXAI). ProeFI is implemented as a Hoeffding tree classifier and should support all river functionalities.

## Installation
ProeFI requires **Python 3.8 or above**. Installation of the requirements can be done via `pip`:
```sh
pip install -r requirements.txt 
```

## Example Code
The following code shows one way to train & evaluate ProeFI on Agrawal with an abrupt concept drift. 
```python
>>> from river import metrics

>>> from tree import ProeFI
>>> from data.datasets.experiment_datasets import DriftingAgrawal

>>> dataset = DriftingAgrawal(width=50)
>>> model = ProeFI()
>>> metric = metrics.ROCAUC()

>>> for x, y in dataset.take(10000):
>>>   y_pred = model.predict_proba_one(x)
>>>   metric.update(y, y_pred)
>>>   model.learn_one(x, y)
```

## Reproducibility
Reproduce paper results with:
```sh
bash ./paper_results.sh 
```
