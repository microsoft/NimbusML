# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
NGramFeaturizer
"""

__all__ = ["NGramFeaturizer"]


from sklearn.base import TransformerMixin

from ...base_transform import BaseTransform
from ...internal.core.feature_extraction.text.ngramfeaturizer import \
    NGramFeaturizer as core
from ...internal.utils.utils import trace
from .extractor import Ngram


class NGramFeaturizer(core, BaseTransform, TransformerMixin):
    """

    Text transforms that can be performed on data before training
    a model.

    .. remarks::
        The ``NGramFeaturizer`` transform produces a matrix of token
        ngrams/skip-grams counts
        for a given corpus of text.
        There are two ways it can do this:

        * build a dictionary of n-grams and use the id in the dictionary as
          the index in the bag;
        * hash each n-gram and use the hash value as the index in the bag.

        The purpose of hashing is to convert variable-length text documents
        into
        equal-length numeric feature vectors, to support dimensionality
        reduction
        and to make the lookup of feature weights faster.

        The text transform is applied to text input columns. It offers
        language
        detection, tokenization, stopwords removing, text normalization and
        feature
        generation. It supports the following languages by default: English,
        French,
        German, Dutch, Italian, Spanish and Japanese.

        The n-grams are represented as count vectors, with vector slots
        corresponding either to n-grams (created using :py:class:`Ngram
        <nimbusml.feature_extraction.text.extractor.Ngram>` ) or to
        their hashes (created using :py:class:`NgramHash
        <nimbusml.feature_extraction.text.extractor.NgramHash>` ). Embedding
        ngrams in a vector
        space allows their contents to be compared in an efficient manner.
        The slot values in the vector can be weighted by the following
        factors:

        * *term frequency* - The number of occurrences of the slot in the
          text
        * *inverse document frequency* - A ratio (the logarithm of
          inverse relative slot frequency) that measures the information a
          slot
          provides by determining how common or rare it is across the entire
          text.
        * *term frequency-inverse document frequency* - the product
          term frequency and the inverse document frequency.

    :param columns: a dictionary of key-value pairs, where key is the output
        column name and value is a list of input column names.

        * Only one key-value pair is allowed.
        * Input column type: string.
        * Output column type:
          `Vector Type </nimbusml/concepts/types#vectortype-column>`_.

        The << operator can be used to set this value (see
        `Column Operator </nimbusml/concepts/columns>`_)

        For example
         * NGramFeaturizer(columns={'features': ['age', 'parity',
           'induced']})
         * NGramFeaturizer() << {'features': ['age', 'parity', 'induced']})

        For more details see `Columns </nimbusml/concepts/columns>`_.

    :param language: Specifies the language used in the data set. The
        following
        values are supported:

        * ``"AutoDetect"``: for automatic language detection.
        * ``"English"``
        * ``"French"``
        * ``"German"``
        * ``"Dutch"``
        * ``"Italian"``
        * ``"Spanish"``
        * ``"Japanese"``.

    :param stop_words_remover: Specifies the stopwords remover to use. There
        are
        three options supported:

        * `None`: No stopwords remover is used.
        * :py:class:`PredefinedStopWordsRemover
          <nimbusml.feature_extraction.text.stopwords.PredefinedStopWordsRemover>` :
          A precompiled language-specific lists
          of stop words is used that includes the most common words from
          Microsoft Office.
        * :py:class:`CustomStopWordsRemover
          <nimbusml.feature_extraction.text.stopwords.CustomStopWordsRemover>` : A
          user-defined list of stopwords. It accepts
          the following option: ``stopword``.

        The default value is `None`.

    :param text_case: Text casing using the rules of the invariant culture.
        Takes the
        following values:

        * ``"Lower"``
        * ``"Upper"``
        * ``"None"``

        The default value is ``"Lower"``.

    :param keep_diacritics: ``False`` to remove diacritical marks; ``True``
        to
        retain diacritical marks. The default value is ``False``.

    :param keep_punctuations: ``False`` to remove punctuation; ``True`` to
        retain punctuation. The default value is ``True``.

    :param keep_numbers: ``False`` to remove numbers; ``True`` to retain
        numbers. The default value is ``True``.

    :param output_tokens_column_name: Column containing the transformed text
        tokens.

    :param dictionary: A dictionary of whitelisted terms which accepts
        the following options:

        * ``Term``: An optional character vector of terms or categories.
        * ``DropUnknowns``: Drop items.
        * ``Sort``: Specifies how to order items when vectorized. Two
          orderings are supported:

            * ``"Occurrence"``: items appear in the order encountered.
            * ``"Value"``: items are sorted according to their default
              comparison.
              For example, text sorting will be case sensitive (e.g., 'A'
              then 'Z'
              then 'a').

        The default value is `None`.
        Note that the stopwords list takes precedence over the dictionary
        whitelist
        as the stopwords are removed before the dictionary terms are
        whitelisted.

    :param word_feature_extractor: Specifies the word feature extraction
        arguments. There
        are two different feature extraction mechanisms:

        * :py:class:`n_gram <nimbusml.feature_extraction.text.extractor.Ngram>`:
          Count-based feature extraction.
        * :py:class:`n_gram_hash
          <nimbusml.feature_extraction.text.extractor.NgramHash>`: Hashing-based
          feature extraction..

        The default value is ``None``.

    :param char_feature_extractor: Specifies the char feature extraction
        arguments. There
        are two different feature extraction mechanisms:

        * :py:class:`n_gram <nimbusml.feature_extraction.text.extractor.Ngram>`:
          Count-based feature extraction.
        * :py:class:`n_gram_hash
          <nimbusml.feature_extraction.text.extractor.NgramHash>`: Hashing-based
          feature extraction.
          The default value is `None`.

    :param vector_normalizer: Normalize vectors (rows) individually by
        rescaling
        them to unit norm. Takes one of the following values:

        * ``"None"``
        * ``"L2"``
        * ``"L1"``
        * ``"LInf"``

        The default value is ``"L2"``.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`n_gram <nimbusml.feature_extraction.text.extractor.Ngram>`,
        :py:class:`n_gram_hash
        <nimbusml.feature_extraction.text.extractor.NgramHash>`,
        :py:class:`CustomStopWordsRemover
        <nimbusml.feature_extraction.text.stopwords.CustomStopWordsRemover>`,
        :py:class:`PredefinedStopWordsRemover
        <nimbusml.feature_extraction.text.stopwords.PredefinedStopWordsRemover>`,
        :py:class:`get_sentiment <nimbusml.feature_extraction.text.Sentiment>`.

    .. index:: transform, featurizer, text

    Example:
       .. literalinclude:: /../nimbusml/examples/NGramFeaturizer.py
              :language: python
    """

    @trace
    def __init__(
            self,
            language='English',
            stop_words_remover=None,
            text_case='Lower',
            keep_diacritics=False,
            keep_punctuations=True,
            keep_numbers=True,
            output_tokens_column_name=None,
            dictionary=None,
            word_feature_extractor=Ngram(
                max_num_terms=[10000000]),
            char_feature_extractor=Ngram(
                ngram_length=3,
                all_lengths=False,
                max_num_terms=[10000000]),
        vector_normalizer='L2',
        columns=None,
            **params):

        if columns:
            params['columns'] = columns
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            language=language,
            stop_words_remover=stop_words_remover,
            text_case=text_case,
            keep_diacritics=keep_diacritics,
            keep_punctuations=keep_punctuations,
            keep_numbers=keep_numbers,
            output_tokens_column_name=output_tokens_column_name,
            dictionary=dictionary,
            word_feature_extractor=word_feature_extractor,
            char_feature_extractor=char_feature_extractor,
            vector_normalizer=vector_normalizer,
            **params)
        self._columns = columns

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
