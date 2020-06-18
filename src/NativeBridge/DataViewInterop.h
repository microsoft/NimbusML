// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vector>
#include <iostream>
#include "stdafx.h"

#define ONEMASK ((size_t)(-1) / 0xFF)
#define PYTHON_DATA_KEY_INFO "..mlVarInfo"
#define PYTHON_DATA_COL_TYPES "..mlColTypes"

class DataSourceBlock;

// Callback function for getting labels for key-type columns. Returns success.
typedef MANAGED_CALLBACK_PTR(bool, GETLABELS)(DataSourceBlock *source, int col, int count, const char **buffer);


// The data source wrapper used for managed interop. Some of the fields of this are visible to managed code.
// As such, it is critical that this class NOT have a vtable, so virtual functions are illegal!
// This is filled in by native code and referenced by managed code.
// REVIEW: Need to figure out proper story for multi-threaded execution.
class DataSourceBlock
{
    // Fields that are visible to managed code come first and do not start with an underscore.
    // Fields that are only visible to this code start with an underscore.

private:
    // *** These fields are known by managed code. It is critical that this struct not have a vtable.
    //     It is also critical that the layout of this prefix NOT vary from release to release or build to build.

    // Number of columns.
    CxInt64 ccol;
    // Total number of rows. Zero for unknown.
    CxInt64 crow;

    // Column names.
    const char **names;
    // Column data kinds.
    const BYTE *kinds;
    // Column key type cardinalities. Zero for unbounded, -1 for non-key-types.
    const CxInt64 *keyCards;
    // Column vector type cardinalities. Zero for variable size, -1 for non-vector-types.
    const CxInt64 *vecCards;
    // The call back item getter function pointers. Currently only used for string
    // values (nullptr for others). For strings these are GETSTR function pointers.
    const void **getters;

    // Call back function for getting labels.
    GETLABELS getLabels;

private:
    // *** Stuff below here is not known by the managed code.

    std::vector<CxInt64> _mpnum;
    std::vector<CxInt64> _mptxt;
    std::vector<CxInt64> _mpkey;

    // The vectors below here are parallel.

    // Column names.
    std::vector<std::string> _vname;
    std::vector<const char*> _cname;
    // Column DataKind values.
    std::vector<BYTE> _vkind;
    // Column key type cardinalities. Zero for unbounded, -1 for non-key-types.
    std::vector<CxInt64> _vkeyCard;
    // Column vector type cardinalities. Zero for variable size, -1 for non-vector-types.
    std::vector<CxInt64> _vvecCard;
    // Data getters for the columns (null for non-text columns).
    std::vector<const void *> _vgetter;

    std::vector<const void*> _vdata;
    std::vector<pb::list> _vtextdata;
    std::vector<char*> _vtextdata_cache;
    std::vector<pb::list> _vkeydata;
    std::vector<pb::list> _vkeynames;

    // Stores the sparse data.
    // REVIEW: need better documentatoin here - is this a pointer, or buffer ? If buffer, why this is not a vector ? Where do we store type of values ? What is indptr ?
    void* _sparseValues;
    int* _sparseIndices;
    int* _indPtr;

public:
    DataSourceBlock(pb::dict& data);
    ~DataSourceBlock();

private:

    pb::object SelectItemForType(pb::list& container)
    {
        auto length = len(container);

        for (auto index = 0; index < length; index++)
        {
            pb::object item = container[index];

            if (!item.is_none())
            {
                return item;
            }
        }

        return pb::object();
    }

    // Callback methods. These are only needed from managed code via the embedded function pointers above,
    // so can be private.
    static MANAGED_CALLBACK(void) GetBL(DataSourceBlock *pdata, int col, long index, /*out*/ signed char &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const signed char *charData = reinterpret_cast<const signed char*>(pdata->_vdata[numCol]);
        dst = charData[index];
    }
    static MANAGED_CALLBACK(void) GetBL64(DataSourceBlock *pdata, int col, long index, /*out*/ signed char &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const double *charData = reinterpret_cast<const double*>(pdata->_vdata[numCol]);
        dst = (signed char)charData[index];
    }
    static MANAGED_CALLBACK(void) GetU1(DataSourceBlock *pdata, int col, long index, /*out*/ unsigned char &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const unsigned char *charData = reinterpret_cast<const unsigned char*>(pdata->_vdata[numCol]);
        dst = charData[index];
    }
    static MANAGED_CALLBACK(void) GetU2(DataSourceBlock *pdata, int col, long index, /*out*/ unsigned short &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const unsigned short *shortData = reinterpret_cast<const unsigned short*>(pdata->_vdata[numCol]);
        dst = shortData[index];
    }
    static MANAGED_CALLBACK(void) GetU4(DataSourceBlock *pdata, int col, long index, /*out*/ unsigned int &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const unsigned int *intData = reinterpret_cast<const unsigned int*>(pdata->_vdata[numCol]);
        dst = intData[index];
    }
    static MANAGED_CALLBACK(void) GetU8(DataSourceBlock *pdata, int col, long index, /*out*/ CxUInt64 &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const CxUInt64 *longData = reinterpret_cast<const CxUInt64*>(pdata->_vdata[numCol]);
        dst = longData[index];
    }
    static MANAGED_CALLBACK(void) GetI1(DataSourceBlock *pdata, int col, long index, /*out*/ signed char &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const signed char *charData = reinterpret_cast<const signed char*>(pdata->_vdata[numCol]);
        dst = charData[index];
    }
    static MANAGED_CALLBACK(void) GetI2(DataSourceBlock *pdata, int col, long index, /*out*/ short &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const short *shortData = reinterpret_cast<const short*>(pdata->_vdata[numCol]);
        dst = shortData[index];
    }
    static MANAGED_CALLBACK(void) GetI4(DataSourceBlock *pdata, int col, long index, /*out*/ int &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const int *intData = reinterpret_cast<const int*>(pdata->_vdata[numCol]);
        dst = intData[index];
    }
    static MANAGED_CALLBACK(void) GetI8(DataSourceBlock *pdata, int col, long index, /*out*/ CxInt64 &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const CxInt64 *longData = reinterpret_cast<const CxInt64*>(pdata->_vdata[numCol]);
        dst = longData[index];
    }
    static MANAGED_CALLBACK(void) GetR4(DataSourceBlock *pdata, int col, long index, /*out*/ float &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const float *floatData = reinterpret_cast<const float*>(pdata->_vdata[numCol]);
        dst = floatData[index];
    }
    static MANAGED_CALLBACK(void) GetR8(DataSourceBlock *pdata, int col, long index, /*out*/ double &dst)
    {
        CxInt64 numCol = pdata->_mpnum[col];
        assert(0 <= numCol && numCol < (CxInt64)pdata->_vdata.size());
        const double *doubleData = reinterpret_cast<const double*>(pdata->_vdata[numCol]);
        dst = doubleData[index];
    }

    // Call back from C# to map from data buffer and index to char* and convert to UTF16.
    static MANAGED_CALLBACK(void) GetTX(DataSourceBlock *pdata, int col, long index, const/*out*/ char*& pch, /*out*/int32_t &size, /*out*/int32_t &missing)
    {
        CxInt64 txCol = pdata->_mptxt[col];
        assert(0 <= txCol && txCol < (CxInt64)pdata->_vtextdata.size());
        pb::object s = pdata->_vtextdata[txCol][index];

        if (pb::isinstance<pb::str>(s))
        {
            size = -1;
            missing = -1;
            pch = (char*)PyUnicode_DATA(s.ptr()); 
            if (s.is_none())
            {
                size = 0;
                pch = 0;
            }
            else
            {
#if _MSC_VER
                Utf8ToUtf16le(pch, pch, size);
#endif
                pdata->_vtextdata_cache.push_back((char*)pch);
            }
        }
        else
        {
            // Missing values in Python are float.NaN.
            assert(pb::cast<float>(s) != NULL);
            missing = 1;
        }
    }

    // The method below executes in python 2.7 only! 
    // Call back from C# to get text data in UTF16 from unicode bytestring
    static MANAGED_CALLBACK(void) GetUnicodeTX(DataSourceBlock *pdata, int col, long index, const/*out*/ char*& pch, /*out*/int32_t &size, /*out*/int32_t &missing)
    {
        CxInt64 txCol = pdata->_mptxt[col];
        assert(0 <= txCol && txCol < (CxInt64)pdata->_vtextdata.size());
        auto s = pdata->_vtextdata[txCol][index];

        if (pb::isinstance<pb::str>(s))
        {
            size = -1;
            missing = -1;
            pch = (char*)PyUnicode_DATA(s.ptr());
#if _MSC_VER
            Utf8ToUtf16le(pch, pch, size);
#endif
            pdata->_vtextdata_cache.push_back((char*)pch);
        }
        else
        {
            // Missing values in Python are float.NaN.
            assert(pb::cast<float>(s) != NULL);
            missing = 1;
        }
    }

#if _MSC_VER
    static void Utf8ToUtf16le(const char* utf8Str, const/*out*/ char*& pch, /*out*/int &size)
    {
        // Allocate the utf16 string buffer.
        size = MultiByteToWideChar(CP_UTF8, 0, utf8Str, -1, NULL, 0);
        if (size == 0)
        {
            pch = 0;
            return;
        }

        wchar_t* utf16Str = new wchar_t[size];

        try
        {
            // Convert the utf8 string.
            MultiByteToWideChar(CP_UTF8, 0, utf8Str, -1, utf16Str, size);
        }
        catch (...)
        {
            // On exception clean up and re-throw.
            if (utf16Str) delete[] utf16Str;
            throw;
        }

        // size includes a NULL character at the end, discount it
        assert(utf16Str[size - 1] == L'\0');
        size -= 1;
        pch = (char*)utf16Str;
    }
#endif

    static MANAGED_CALLBACK(void) GetKeyInt(DataSourceBlock *pdata, int col, long index, /*out*/ int& dst)
    {
        CxInt64 keyCol = pdata->_mpkey[col];
        assert(0 <= keyCol && keyCol < (CxInt64)pdata->_vkeydata.size());

        auto & list = pdata->_vkeydata[keyCol];
        pb::object obj = pdata->SelectItemForType(list);
        assert(strcmp(obj.ptr()->ob_type->tp_name, "int") == 0);
        dst = pb::cast<int>(list[index]);
    }

    // Callback function for getting labels for key-type columns. Returns success.
    static MANAGED_CALLBACK(bool) GetKeyNames(DataSourceBlock *pdata, int col, int count, const char **buffer)
    {
        if (count <= 0 || buffer == nullptr)
        {
            // Invalid count or buffer, don't zero out buffer returning.
            assert(false);
            return false;
        }
        if (pdata == nullptr)
        {
            // Invalid pdata.
            return OnGetLabelsFailure(count, buffer);
        }
        if (0 > col || (size_t)col >= pdata->_mpkey.size())
        {
            // Invalid column id.
            return OnGetLabelsFailure(count, buffer);
        }
        if (pdata->_vkeyCard[col] != count)
        {
            // Column is not a key type.
            return OnGetLabelsFailure(count, buffer);
        }

        CxInt64 keyCol = pdata->_mpkey[col];
        pb::list & names = pdata->_vkeynames[keyCol];
        if (len(names) != count)
        {
            // No labels for this column. This is not a logic error.
            return OnGetLabelsFailure(count, buffer);
        }

        for (int i = 0; i < count; ++i, ++buffer)
            *buffer = (char*)PyUnicode_DATA(names[i].ptr());
        return true;
    }

    static bool OnGetLabelsFailure(int count, const char **buffer)
    {
        assert(false);
        for (int i = 0; i < count; i++)
            buffer[i] = nullptr;
        return false;
    }

    // Same method has two modes: if "inquire" is true, it returns the number of indices/values needed for the current row.
    // If "inquire" is false, it assumes that indices/values are big enough, and fills them in for the current row.
    static MANAGED_CALLBACK(void) GetBLVector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, unsigned char* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const unsigned char *boolData = reinterpret_cast<const unsigned char*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = boolData[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetU1Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, unsigned char* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const unsigned char *int8Data = reinterpret_cast<const unsigned char*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int8Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetU2Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, unsigned short* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const unsigned short *int16Data = reinterpret_cast<const unsigned short*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int16Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetU4Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, unsigned int* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const unsigned int *int32Data = reinterpret_cast<const unsigned int*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int32Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetU8Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, CxUInt64* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const unsigned long *int64Data = reinterpret_cast<const unsigned long*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int64Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetI1Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, signed char* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const signed char *int8Data = reinterpret_cast<const signed char*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int8Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetI2Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, short* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const short *int16Data = reinterpret_cast<const short*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int16Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetI4Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, int* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const int *int32Data = reinterpret_cast<const int*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int32Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetI8Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, CxInt64* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const CxInt64 *int64Data = reinterpret_cast<const CxInt64*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = int64Data[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetR4Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, float* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const float *floatData = reinterpret_cast<const float*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = floatData[pdata->_indPtr[index] + i];
        }
    }

    static MANAGED_CALLBACK(void) GetR8Vector(DataSourceBlock *pdata, int col, CxInt64 index, int* indices, double* values, bool inquire, /*out*/ int &size)
    {
        size = pdata->_indPtr[index + 1] - pdata->_indPtr[index];
        if (inquire)
            return;

        const double *doubleData = reinterpret_cast<const double*>(pdata->_sparseValues);
        for (int i = 0; i < size; i++)
        {
            indices[i] = pdata->_sparseIndices[pdata->_indPtr[index] + i];
            values[i] = doubleData[pdata->_indPtr[index] + i];
        }
    }
};

// A native wrapper around a managed IDataView for receiving data back from managed code.
// Managed code owns instances of this class. The lifetime is scoped to a call from managed code.
// This is filled in by managed code and referenced by native code.
struct DataViewBlock
{
    // *** These fields are shared from managed code. It is critical that this struct not have a vtable.
    //     It is also critical that the layout of this NOT vary from release to release or build to build.
    //     The managed code assumes that CxInt64 occupies 8 bytes, and each pointer occupies 8 bytes.

    // Number of columns.
    CxInt64 ccol;
    // Total number of rows. Zero for unknown.
    CxInt64 crow;

    // Column names.
    const char **names;
    // Column data kinds.
    const BYTE *kinds;
    // Column key type cardinalities. Only contains the values for the columns that have
    // key names.
    const int *keyCards;
    // The number of values in each row of a column.
    // A value count of 0 means that each row of the
    // column is variable length.
    const BYTE *valueCounts;
};

enum ML_PY_TYPE_MAP_ENUM {
    ML_PY_BOOL = '?',
    ML_PY_BOOL64 = '!',
    ML_PY_UINT8 = 'B',
    ML_PY_UINT16 = 'H',
    ML_PY_UINT32 = 'I',
    ML_PY_UINT64 = 'Q',
    ML_PY_INT8 = 'b',
    ML_PY_INT16 = 'h',
    ML_PY_INT32 = 'i',
    ML_PY_INT64 = 'q',
    ML_PY_FLOAT16 = 'e',
    ML_PY_FLOAT32 = 'f',
    ML_PY_FLOAT64 = 'd',
    ML_PY_CAT = 'c',
    ML_PY_TEXT = 't',
    ML_PY_UNICODE = 'u',
    ML_PY_DATETIME = 'z',
    ML_PY_UNSUPPORTED = 'x'
};
