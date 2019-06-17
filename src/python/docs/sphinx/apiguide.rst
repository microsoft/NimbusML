.. rxtitle:: API Guide
.. rxdescription:: Summary of list of transforms and learners

=========
API Guide
=========

.. contents::
    :local:

Trainers
--------

Click on link to see details for each of the trainers. Some classifiers implement ``predict_proba()``
to produce calibrated probabilities. Others implement ``decision_function()`` only for raw scores.

Note
""""""""""""""""""
**Several listed components will be available in future releases and currently their doc reference
links are disabled, for ex. TimeSeries transforms below (ExponentialAverage, IIDChangePointDetector).**


Binary Classifiers
""""""""""""""""""

,, Trainer ,, predict_proba() ,, decision_function() ,,
,, ,,, ,, ,,, ,, ,,, ,,
,, :py:class:`AveragedPerceptronBinaryClassifier<nimbusml.linear_model.AveragedPerceptronBinaryClassifier>` ,,   ,, Yes ,,
,, :py:class:`FactorizationMachineBinaryClassifier<nimbusml.decomposition.FactorizationMachineBinaryClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`FastForestBinaryClassifier<nimbusml.ensemble.FastForestBinaryClassifier>` ,,   ,, Yes ,,
,, :py:class:`FastLinearBinaryClassifier<nimbusml.linear_model.FastLinearBinaryClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`FastTreesBinaryClassifier<nimbusml.ensemble.FastTreesBinaryClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`GamBinaryClassifier<nimbusml.ensemble.GamBinaryClassifier>` ,,   ,, Yes ,,
,, :py:class:`LightGbmBinaryClassifier<nimbusml.ensemble.LightGbmBinaryClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`LocalDeepSvmBinaryClassifier<nimbusml.svm.LocalDeepSvmBinaryClassifier>` ,,   ,, Yes ,,
,, :py:class:`LogisticRegressionBinaryClassifier<nimbusml.linear_model.LogisticRegressionBinaryClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`SgdBinaryClassifier<nimbusml.linear_model.SgdBinaryClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`SymSgdBinaryClassifier<nimbusml.linear_model.SymSgdBinaryClassifier>` ,, Yes ,, Yes ,,


Multiclass Classifiers
""""""""""""""""""""""

,, Trainer ,, predict_proba() ,, decision_function() ,,
,, ,,, ,, ,,, ,, ,,, ,,
,, :py:class:`FastLinearClassifier<nimbusml.linear_model.FastLinearClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`LightGbmClassifier<nimbusml.ensemble.LightGbmClassifier>` ,, Yes ,, Yes ,,
,, :py:class:`LogisticRegressionClassifier<nimbusml.linear_model.LogisticRegressionClassifier>` ,, Yes ,,   ,,
,, :py:class:`NaiveBayesClassifier<nimbusml.naive_bayes.NaiveBayesClassifier>` ,,   ,, Yes ,,
,, :py:class:`OneVsRestClassifier<nimbusml.multiclass.OneVsRestClassifier>` ,, Yes ,, Yes ,,


Regressors
""""""""""""""""""""""

,, Trainer ,,
,, ,,, ,,
,, :py:class:`FastForestRegressor<nimbusml.ensemble.FastForestRegressor>` ,,
,, :py:class:`FastLinearRegressor<nimbusml.linear_model.FastLinearRegressor>` ,,
,, :py:class:`FastTreesRegressor<nimbusml.ensemble.FastTreesRegressor>` ,,
,, :py:class:`FastTreesTweedieRegressor<nimbusml.ensemble.FastTreesTweedieRegressor>` ,,
,, :py:class:`GamRegressor<nimbusml.ensemble.GamRegressor>` ,,
,, :py:class:`LightGbmRegressor<nimbusml.ensemble.LightGbmRegressor>` ,,
,, :py:class:`OnlineGradientDescentRegressor<nimbusml.linear_model.OnlineGradientDescentRegressor>` ,,
,, :py:class:`OrdinaryLeastSquaresRegressor<nimbusml.linear_model.OrdinaryLeastSquaresRegressor>` ,,
,, :py:class:`PoissonRegressionRegressor<nimbusml.linear_model.PoissonRegressionRegressor>` ,,


Others
""""""

,, Trainer ,, Type ,,
,, ,,, ,, ,,, ,,
,, :py:class:`LightGbmRanker<nimbusml.ensemble.LightGbmRanker>` ,, ranker ,,
,, :py:class:`KMeansPlusPlus<nimbusml.cluster.KMeansPlusPlus>` ,, clusterer ,,
,, :py:class:`OneClassSvmAnomalyDetector<nimbusml.svm.OneClassSvmAnomalyDetector>` ,, anomaly ,,
,, :py:class:`PcaAnomalyDetector<nimbusml.decomposition.PcaAnomalyDetector>` ,, anomaly ,,

Transforms
----------

Click on link to see details for each of the transforms, and dependent subclasses.


Feature Extraction
""""""""""""""""""

,, Transform ,, Additional subclasses ,,
,, ,,, ,, ,,, ,,
,, :py:class:`LightLda<nimbusml.feature_extraction.text.LightLda>` ,,    ,,
,, :py:class:`Loader<nimbusml.feature_extraction.image.Loader>` ,,    ,,
,, :py:class:`NGramFeaturizer<nimbusml.feature_extraction.text.NGramFeaturizer>` ,, :py:class:`Ngram<nimbusml.feature_extraction.text.extractor.Ngram>`, :py:class:`NgramHash<nimbusml.feature_extraction.text.extractor.NgramHash>`, :py:class:`CustomStopWordsRemover<nimbusml.feature_extraction.text.stopwords.CustomStopWordsRemover>`, :py:class:`PredefinedStopWordsRemover<nimbusml.feature_extraction.text.stopwords.PredefinedStopWordsRemover>`   ,,
,, :py:class:`OneHotHashVectorizer<nimbusml.feature_extraction.categorical.OneHotHashVectorizer>` ,,    ,,
,, :py:class:`OneHotVectorizer<nimbusml.feature_extraction.categorical.OneHotVectorizer>` ,,    ,,
,, :py:class:`PcaTransformer<nimbusml.decomposition.PcaTransformer>` ,,    ,,
,, :py:class:`PixelExtractor<nimbusml.feature_extraction.image.PixelExtractor>` ,,    ,,
,, :py:class:`Resizer<nimbusml.feature_extraction.image.Resizer>` ,,    ,,
,, :py:class:`Sentiment<nimbusml.feature_extraction.text.Sentiment>` ,,    ,,
,, :py:class:`TreeFeaturizer<nimbusml.feature_extraction.TreeFeaturizer>` ,,    ,,
,, :py:class:`WordEmbedding<nimbusml.feature_extraction.text.WordEmbedding>` ,,    ,,


Feature Selection
"""""""""""""""""

,, Transform ,,
,, ,,, ,,
,, :py:class:`CountSelector<nimbusml.feature_selection.CountSelector>` ,,
,, :py:class:`MutualInformationSelector<nimbusml.feature_selection.MutualInformationSelector>` ,,


Preprocessing
"""""""""""""

,, Transform ,, 
,, ,,, ,, 
,, :py:class:`Binner<nimbusml.preprocessing.normalization.Binner>` ,,
,, :py:class:`BootstrapSampler<nimbusml.preprocessing.filter.BootstrapSampler>` ,,
,, :py:class:`CharTokenizer<nimbusml.preprocessing.text.CharTokenizer>` ,,
,, :py:class:`ColumnConcatenator<nimbusml.preprocessing.schema.ColumnConcatenator>` ,,
,, :py:class:`ColumnDropper<nimbusml.preprocessing.schema.ColumnDropper>` ,,
,, :py:class:`ColumnDuplicator<nimbusml.preprocessing.schema.ColumnDuplicator>` ,,
,, :py:class:`ColumnSelector<nimbusml.preprocessing.schema.ColumnSelector>` ,,
,, :py:class:`Expression<nimbusml.preprocessing.Expression>` ,,
,, :py:class:`Filter<nimbusml.preprocessing.missing_values.Filter>` ,,
,, :py:class:`GlobalContrastRowScaler<nimbusml.preprocessing.normalization.GlobalContrastRowScaler>` ,,
,, :py:class:`Handler<nimbusml.preprocessing.missing_values.Handler>` ,,
,, :py:class:`Indicator<nimbusml.preprocessing.missing_values.Indicator>` ,,
,, :py:class:`FromKey<nimbusml.preprocessing.FromKey>` ,,
,, :py:class:`LogMeanVarianceScaler<nimbusml.preprocessing.normalization.LogMeanVarianceScaler>` ,,
,, :py:class:`MeanVarianceScaler<nimbusml.preprocessing.normalization.MeanVarianceScaler>` ,,
,, :py:class:`MinMaxScaler<nimbusml.preprocessing.normalization.MinMaxScaler>` ,,
,, :py:class:`RangeFilter<nimbusml.preprocessing.filter.RangeFilter>` ,,
,, :py:class:`SkipFilter<nimbusml.preprocessing.filter.SkipFilter>` ,,
,, :py:class:`SupervisedBinner<nimbusml.preprocessing.normalization.SupervisedBinner>` ,,
,, :py:class:`TakeFilter<nimbusml.preprocessing.filter.TakeFilter>` ,,
,, :py:class:`TensorFlowScorer<nimbusml.preprocessing.TensorFlowScorer>` ,,
,, :py:class:`ToKey<nimbusml.preprocessing.ToKey>` ,,
,, :py:class:`TypeConverter<nimbusml.preprocessing.schema.TypeConverter>` ,,  


TimeSeries
""""""""""

,, Transform ,, 
,, ,,, ,, 
,, :py:class:`ExponentialAverage<nimbusml.preprocessing.timeseries.ExponentialAverage>` ,,
,, :py:class:`IIDChangePointDetector<nimbusml.preprocessing.timeseries.IIDChangePointDetector>` ,,
,, :py:class:`IIDSpikeDetector<nimbusml.preprocessing.timeseries.IIDSpikeDetector>` ,,
,, :py:class:`PercentileThreshold<nimbusml.preprocessing.timeseries.PercentileThreshold>` ,,
,, :py:class:`Pvalue<nimbusml.preprocessing.timeseries.Pvalue>` ,,
,, :py:class:`SlidingWindow<nimbusml.preprocessing.timeseries.SlidingWindow>` ,,
,, :py:class:`SsaChangePointDetector<nimbusml.preprocessing.timeseries.SsaChangePointDetector>` ,,
,, :py:class:`SsaSpikeDetector<nimbusml.preprocessing.timeseries.SsaSpikeDetector>` ,,


Subclasses
----------

These are auxillary classes used by transforms or trainers.

,, Subclasses ,, Used By  ,,
,, ,,, ,, ,,, ,,
,, :py:class:`CustomStopWordsRemover<nimbusml.feature_extraction.text.stopwords.CustomStopWordsRemover>`  ,, :py:class:`NGramFeaturizer<nimbusml.feature_extraction.text.NGramFeaturizer>`   ,,
,, :py:class:`Dart<nimbusml.ensemble.booster.Dart>`  ,, :py:class:`LightGbmBinaryClassifier<nimbusml.ensemble.LightGbmBinaryClassifier>`, :py:class:`LightGbmClassifier<nimbusml.ensemble.LightGbmClassifier>`, :py:class:`LightGbmRanker<nimbusml.ensemble.LightGbmRanker>`, :py:class:`LightGbmRegressor<nimbusml.ensemble.LightGbmRegressor>`    ,,
,, :py:class:`Gbdt<nimbusml.ensemble.booster.Gbdt>`  ,, :py:class:`LightGbmBinaryClassifier<nimbusml.ensemble.LightGbmBinaryClassifier>`, :py:class:`LightGbmClassifier<nimbusml.ensemble.LightGbmClassifier>`, :py:class:`LightGbmRanker<nimbusml.ensemble.LightGbmRanker>`, :py:class:`LightGbmRegressor<nimbusml.ensemble.LightGbmRegressor>`    ,,
,, :py:class:`Goss<nimbusml.ensemble.booster.Goss>`  ,,  :py:class:`LightGbmBinaryClassifier<nimbusml.ensemble.LightGbmBinaryClassifier>`, :py:class:`LightGbmClassifier<nimbusml.ensemble.LightGbmClassifier>`, :py:class:`LightGbmRanker<nimbusml.ensemble.LightGbmRanker>`, :py:class:`LightGbmRegressor<nimbusml.ensemble.LightGbmRegressor>`   ,,
,, :py:class:`LinearKernel<nimbusml.svm.kernel.LinearKernel>`  ,,  :py:class:`OneClassSvmAnomalyDetector<nimbusml.svm.OneClassSvmAnomalyDetector>`  ,,
,, :py:class:`Ngram<nimbusml.feature_extraction.text.extractor.Ngram>`  ,, :py:class:`NGramFeaturizer<nimbusml.feature_extraction.text.NGramFeaturizer>`   ,,
,, :py:class:`NgramHash<nimbusml.feature_extraction.text.extractor.NgramHash>`  ,, :py:class:`NGramFeaturizer<nimbusml.feature_extraction.text.NGramFeaturizer>`   ,,
,, :py:class:`PolynomialKernel<nimbusml.svm.kernel.PolynomialKernel>`  ,,  :py:class:`OneClassSvmAnomalyDetector<nimbusml.svm.OneClassSvmAnomalyDetector>`  ,,
,, :py:class:`PredefinedStopWordsRemover<nimbusml.feature_extraction.text.stopwords.PredefinedStopWordsRemover>`  ,, :py:class:`NGramFeaturizer<nimbusml.feature_extraction.text.NGramFeaturizer>`   ,,
,, :py:class:`RbfKernel<nimbusml.svm.kernel.RbfKernel>`  ,, :py:class:`OneClassSvmAnomalyDetector<nimbusml.svm.OneClassSvmAnomalyDetector>`   ,,
,, :py:class:`SigmoidKernel<nimbusml.svm.kernel.SigmoidKernel>`  ,, :py:class:`OneClassSvmAnomalyDetector<nimbusml.svm.OneClassSvmAnomalyDetector>`   ,,


Loss Functions
--------------

Trainers use a variety of loss functions. Click on the links for further details about each of these.

,, Loss Functions ,, Used By  ,,
,, ,,, ,, ,,, ,,
,, :py:class:`Exp<nimbusml.loss.Exp>`  ,, :py:class:`AveragedPerceptronBinaryClassifier<nimbusml.linear_model.AveragedPerceptronBinaryClassifier>`, :py:class:`SgdBinaryClassifier<nimbusml.linear_model.SgdBinaryClassifier>` ,,
,, :py:class:`Hinge<nimbusml.loss.Hinge>`  ,, :py:class:`AveragedPerceptronBinaryClassifier<nimbusml.linear_model.AveragedPerceptronBinaryClassifier>`, :py:class:`SgdBinaryClassifier<nimbusml.linear_model.SgdBinaryClassifier>`, :py:class:`FastLinearBinaryClassifier<nimbusml.linear_model.FastLinearBinaryClassifier>`, :py:class:`FastLinearClassifier<nimbusml.linear_model.FastLinearClassifier>` ,,
,, :py:class:`Log<nimbusml.loss.Log>`  ,, :py:class:`AveragedPerceptronBinaryClassifier<nimbusml.linear_model.AveragedPerceptronBinaryClassifier>`, :py:class:`SgdBinaryClassifier<nimbusml.linear_model.SgdBinaryClassifier>`, :py:class:`FastLinearBinaryClassifier<nimbusml.linear_model.FastLinearBinaryClassifier>`, :py:class:`FastLinearClassifier<nimbusml.linear_model.FastLinearClassifier>` ,,
,, :py:class:`Poisson<nimbusml.loss.Poisson>`  ,, :py:class:`OnlineGradientDescentRegressor<nimbusml.linear_model.OnlineGradientDescentRegressor>` ,,
,, :py:class:`SmoothedHinge<nimbusml.loss.SmoothedHinge>`  ,, :py:class:`AveragedPerceptronBinaryClassifier<nimbusml.linear_model.AveragedPerceptronBinaryClassifier>`, :py:class:`SgdBinaryClassifier<nimbusml.linear_model.SgdBinaryClassifier>`, :py:class:`FastLinearBinaryClassifier<nimbusml.linear_model.FastLinearBinaryClassifier>`, :py:class:`FastLinearClassifier<nimbusml.linear_model.FastLinearClassifier>` ,,
,, :py:class:`Squared<nimbusml.loss.Squared>`  ,, :py:class:`FastLinearRegressor<nimbusml.linear_model.FastLinearRegressor>`, :py:class:`OnlineGradientDescentRegressor<nimbusml.linear_model.OnlineGradientDescentRegressor>`   ,,
,, :py:class:`Tweedie<nimbusml.loss.Tweedie>`  ,, :py:class:`OnlineGradientDescentRegressor<nimbusml.linear_model.OnlineGradientDescentRegressor>`   ,,
