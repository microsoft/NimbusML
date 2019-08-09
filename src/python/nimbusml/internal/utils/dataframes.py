# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
from collections import OrderedDict

import numpy as np
import six
from pandas import DataFrame, Series, concat, Categorical
from pandas.api.types import infer_dtype
from scipy.sparse import csr_matrix


def resolve_dataframe(dataframe):
    if isinstance(dataframe, DataFrame):
        ret = OrderedDict()
        ret['..mlVarInfo'] = {}
        rx_var_info = getattr(dataframe, "var_info", None)
        types = []

        for i in dataframe.columns:
            if six.PY2 and isinstance(i, unicode):
                i = i.encode('utf-8')
            if rx_var_info is not None and i in rx_var_info.keys():
                ret['..mlVarInfo'][i] = rx_var_info[i]
            # Does not work with multiindex (confusion with tuple and list)
            name_i = i if isinstance(i, str) else '.'.join(str(_) for _ in i)
            serie = dataframe.loc[:, (i,)].iloc[:, 0]
            if str(serie.dtype) == 'category':
                # Workaround, empty dataframe needs to be sent as an array to
                # convey type information
                if len(serie) == 0:
                    # Default numpy array will be float, categorical variable
                    # needs to be int
                    ret[name_i] = np.array(
                        [x + 1 for x in serie.cat.codes.values.tolist()]) \
                        .reshape((len(serie), 1)).astype(np.int)
                else:
                    ret[name_i] = [
                        x + 1 for x in serie.cat.codes.values.tolist()]
                ret['..mlVarInfo'][i] = [
                    str(cast) for cast in serie.cat.categories.tolist()]
                types.extend(['c'])

            else:
                if len(serie) == 0:
                    # Workaround, empty dataframe needs to be sent as an array
                    # to convey type information
                    ret[name_i] = serie.values.reshape((len(serie), 1))
                elif serie.dtype == np.object or str(serie.dtype) == '<U1':
                    # This column might still be numeric, so we do another
                    # check.
                    infered_dtype = infer_dtype(serie, skipna=True)
                    if not infered_dtype == 'string' and \
                            not infered_dtype == 'unicode':
                        ret[name_i] = serie.values
                        if infered_dtype == 'floating' or \
                                infered_dtype == 'mixed-integer-float':
                            s = serie.itemsize
                            if s == 8:
                                ret[str(i)] = serie.values.astype(
                                    np.float64, copy=False)
                                types.extend(
                                    [_global_dtype_to_char_dict[
                                         np.dtype(np.float64)]])
                            else:
                                ret[str(i)] = serie.values.astype(
                                    np.float32, copy=False)
                                types.extend(
                                    [_global_dtype_to_char_dict[
                                         np.dtype(np.float32)]])
                        elif infered_dtype == 'integer':
                            s = serie.itemsize
                            if s == 8:
                                ret[str(i)] = serie.values.astype(
                                    np.int64, copy=False)
                                types.extend(
                                    [_global_dtype_to_char_dict[
                                         np.dtype(np.int64)]])
                            elif s == 4:
                                ret[str(i)] = serie.values.astype(
                                    np.int32, copy=False)
                                types.extend(
                                    [_global_dtype_to_char_dict[
                                         np.dtype(np.int32)]])
                            elif s == 2:
                                ret[str(i)] = serie.values.astype(
                                    np.int16, copy=False)
                                types.extend(
                                    [_global_dtype_to_char_dict[
                                         np.dtype(np.int16)]])
                            else:
                                ret[str(i)] = serie.values.astype(
                                    np.int8, copy=False)
                                types.extend(
                                    [_global_dtype_to_char_dict[
                                         np.dtype(np.int8)]])
                        elif infered_dtype == 'boolean':
                            ret[str(i)] = serie.values.astype(
                                np.float64, copy=False)
                            types.extend(
                                [_global_dtype_to_char_dict['bool64']])
                        elif infered_dtype.startswith('mixed'):
                            raise TypeError(
                                "argument must be a string or a number")
                        else:
                            raise TypeError(
                                "Type %s not supported" % infered_dtype)
                    else:
                        ret[name_i] = serie.values.tolist()
                        if infered_dtype == 'string':
                            types.extend(
                                [_global_dtype_to_char_dict[
                                     np.dtype(np.string_)]])
                        else:
                            types.extend(
                                [_global_dtype_to_char_dict[
                                     np.dtype(np.unicode)]])
                else:
                    ret[name_i] = serie.values
                    if serie.dtype in _global_dtype_to_char_dict:
                        ch = _global_dtype_to_char_dict[serie.dtype]
                    else:
                        ch = _global_dtype_to_char_dict['unsupported']
                    types.extend([ch])

        ret['..mlColTypes'] = types
        return ret
    return None


def resolve_csr_matrix(matrix, y=None):
    ret = OrderedDict()
    ret['..mlVarInfo'] = {}
    if y is not None:
        ret = resolve_dataframe(y)

    if isinstance(matrix, csr_matrix):
        if y is not None:
            types = ret['..mlColTypes']
        else:
            types = []
        if not matrix.has_sorted_indices:
            matrix.sort_indices()
        if not matrix.has_canonical_format:
            matrix.sum_duplicates()

        ret['sparse'] = OrderedDict()
        ret['sparse']['values'] = matrix.data
        ret['sparse']['indices'] = matrix.indices
        ret['sparse']['indptr'] = matrix.indptr
        ret['sparse']['colCount'] = matrix.shape[1]
        if matrix.data.dtype in _global_dtype_to_char_dict:
            ch = _global_dtype_to_char_dict[matrix.data.dtype]
        else:
            ch = _global_dtype_to_char_dict['unsupported']
        types.extend([ch])
        ret['..mlColTypes'] = types
    return ret


def pd_concat(els, axis=0, join='inner'):
    """
    Concatenates several arrays.
    """
    if axis == 1:
        def get_dim(el):
            if isinstance(el, (DataFrame, np.ndarray, csr_matrix)):
                return el.shape[0]
            else:
                return len(el)

        def get_obj(el):
            if isinstance(el, (DataFrame, Series)):
                return el.reset_index(drop=True)
            elif isinstance(el, np.ndarray):
                return DataFrame(el)
            else:
                return el

        dims = [get_dim(el) for el in els]
        if min(dims) == max(dims):
            res = concat([get_obj(el) for el in els], axis=axis, join=join)
            hres = res.head()
            for i in hres.columns:
                if not isinstance(hres[i], Series):
                    raise RuntimeError(
                        "One column is not a series, this happens when one "
                        "of the input columns has name 'F?'.\n" +
                        "This happens for example when X and y contain the "
                        "same column name.\n" +
                        "nimbusml cannot distinguish between the label in X and "
                        "the label in Y.\n" +
                        "nimbusml generates intermediate columns with this kind "
                        "of name. Issue with column '{0}' among "
                        "columns\n{1}".format(
                            i,
                            res.columns))
            return res

    return concat(els, axis=axis, join=join)


def resolve_output(ret):
    data = dict()
    for key in ret.keys():
        if not isinstance(ret[key], dict):
            data[key] = ret[key]
        else:
            data[key] = Categorical.from_codes(
                ret[key]["..Data"], ret[key]["..KeyValues"])
    return DataFrame(data)


# Any changes to this dictionary must also be done in the enum
# ML_PY_TYPE_MAP_ENUM defined in DataViewInterop.h.
_global_dtype_to_char_dict = {
    np.dtype(np.bool): '?',
    'bool64': '!',
    np.dtype(np.ubyte): 'B',
    np.dtype(np.uint16): 'H',
    np.dtype(np.uint32): 'I',
    np.dtype(np.uint64): 'Q',
    np.dtype(np.int8): 'b',
    np.dtype(np.int16): 'h',
    np.dtype(np.int32): 'i',
    np.dtype(np.int64): 'q',
    np.dtype(np.float16): 'e',
    np.dtype(np.float32): 'f',
    np.dtype(np.float64): 'd',
    np.dtype(np.string_): 't',
    np.dtype(np.unicode): 'u',
    'unsupported': 'x'
}
