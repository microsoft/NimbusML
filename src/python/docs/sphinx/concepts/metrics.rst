.. rxtitle:: Evaluation Metrics
.. rxdescription:: Model evaluation metrics the nimbusml provides

.. _metrics:

=========================
Evaluation Metrics
=========================

.. index:: evaluation metrics



Metrics from Pipeline.test()
----------------------------

The evaluation metrics for models are generated using the ``test()`` method of
:py:class:`nimbusml.Pipeline`.

The type of metrics to generate is inferred automatically by looking at the trainer type in the
pipeline. If a model has been loaded using the ``load_model()`` method, then the `evaltype` must
be specified explicitly.


Binary Classification Metrics
"""""""""""""""""""""""""""""

This corresponds to evaltype='binary'.

**AUC** - see `Receiver Operating Characteristic <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

**Accuracy** - see `Precision and Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_

**Positive Precision** - see  `Precision and Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_

**Positive Recall** - see  `Precision and Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_

**Negative Precision** -  see  `Precision and Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_

**Negative Recall** -  see  `Precision and Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_

**Log-loss** - see `Log Loss <http://wiki.fast.ai/index.php/Log_Loss>`_

**Log-loss reduction** - RIG(Y|X) * 100 = (H(Y) - H(Y|X)) / H(Y) * 100. Ranges from [-inf, 100], where
100 is perfect predictions and 0 indicates mean predictions.

**Test-set Entropy** - H(Y)

**F1 Score** - see `Precision and Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_

**AUPRC** - see `Area under Precision-Recall Curve <http://pages.cs.wisc.edu/~boyd/aucpr_final.pdf>`_

.. note:: Note about ROC

    The computed AUC is defined as the probability that the score
    for a positive example is higher than the score for a negative one
    (see `AucAggregator.cs <https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.Data/Evaluators/AucAggregator.cs#L135>`_
    in `ML.net <https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet>`_).
    This expression is asymptotically equivalent to the area under the curve
    which is what
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html>`_ computation.
    computes
    (see `auc <https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L101>`_).
    That explains discrepencies on small test sets.


Multiclass Classification Metrics
"""""""""""""""""""""""""""""""""

This corresponds to evaltype='multiclass'.

**Accuracy(micro-avg)** - Every sample-class pair contribute equally to the accuracy metric.

**Accuracy(macro-avg)** - Every class contributes equally to the accuracy metric. Minority classes are
given equal weight as the larger classes.

**Log-loss** - see `Log Loss <http://wiki.fast.ai/index.php/Log_Loss>`_

**Log-loss reduction** - RIG(Y|X) * 100 = (H(Y) - H(Y|X)) / H(Y) * 100. Ranges from [-inf, 100], where
100 is perfect predictions and 0 indicates mean predictions.

**(class N)** - Accuracy of class N

Regression Metrics
""""""""""""""""""

This corresponds to evaltype='regression'.

**L1(avg)** - E( \| y - y' \| )

**L2(avg)** - E( ( y - y' )^2 )

**RMS(avg)** - E( ( y - y' )^2 )^0.5

**Loss-fn(avg)** - Expected value of loss function. If using square loss, is equal to L2(avg)

Clustering Metrics
""""""""""""""""""

This corresponds to evaltype='cluster'.

**NMI** - measure of the mutual dependence of the variables. See `Normalized Variants
<https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants>`_. Range is in [0,1], where higher is better.

**AvgMinScore** - Mean distance of samples to centroids. Smaller is better.


Ranking Metrics
""""""""""""""""""

This corresponds to evaltype='ranking'.

**NDCG@N** - Normalized Discounted Cumulative Gain @ Top N positions. See `Discounted Cumulative
Gain <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_

**DCG@N** - Discounted Cumulative Gain @ Top N positions. See `Discounted Cumulative Gain
<https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_


Anomaly Detection Metrics
"""""""""""""""""""""""""

This corresponds to evaltype='anomaly'.

**AUC** - see `Receiver Operating Characteristic
<https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

**DR @K FP** - Detection rate at k false positives. When the test examples are sorted by the output
of the anomaly detector in descending order, denote by K the index of the k'th example whose label
is 0. Detection rate at k false positives is the detection rate at K.

**DR @K FPR** - Detection rate at fraction p false positives. When the test examples are sorted by
the output of the anomaly detector in descending order, denote by K the index such that a fraction
p of the label 0 examples are above K. Detection rate at fraction p false positives is the
detection rate at K.

**DR @NumPos** - Detection rate at number of anomalies. Denote by D the number of label 1 examples
in the test set. Detection rate at number of anomalies is equal to the detection rate at D.

**NumAnomalies** - Total number of anomalies detected.
