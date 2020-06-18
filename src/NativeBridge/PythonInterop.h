// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <map>
#include <vector>


// Taken from ML.NET source code. These values should be stable.
enum DataKind
{
    I1 = 1,
    U1 = 2,
    I2 = 3,
    U2 = 4,
    I4 = 5,
    U4 = 6,
    I8 = 7,
    U8 = 8,
    R4 = 9,
    R8 = 10,
    TX = 11,
    BL = 12,
    TS = 13,
    DT = 14,
    DZ = 15,
};

class PyColumnBase
{
private:
    typedef std::map<int, PyColumnBase* (*)(const int&, size_t)> creation_map;
    typedef std::pair<int, PyColumnBase* (*)(const int&, size_t)> creation_map_entry;

    static creation_map* m_pSingleCreationMap;
    static creation_map* CreateSingleMap();

    static creation_map* m_pVariableCreationMap;
    static creation_map* CreateVariableMap();

    template <class T> static PyColumnBase* CreateSingle(const int& kind, size_t nRows);
    template <class T, class T2> static PyColumnBase* CreateVariable(const int& kind, size_t nRows);

protected:
    int _kind;

public:
    static PyColumnBase* Create(const int& kind, size_t numRows, size_t numCols);

    PyColumnBase(const int& kind);
    virtual ~PyColumnBase();

    const int& GetKind() const { return _kind; }
    void SetKind(int kind) { _kind = kind; }

    virtual size_t GetNumRows() = 0;
    virtual size_t GetNumCols() = 0;
};


/*
 * Template typed abstract base class which provides
 * the required interface for all derived classes.
 */
template <class T>
class PyColumn : public PyColumnBase
{
public:
    PyColumn(const int& kind) : PyColumnBase(kind) {}
    virtual ~PyColumn() {}
    virtual void SetAt(size_t nRow, size_t nCol, const T& value) = 0;
    virtual void AddToDict(pb::dict& dict,
                           const std::string& name,
                           const std::vector<std::string>* keyNames,
                           const size_t expectedRows) = 0;
};


/*
 * Handles the single value case.
 */
template <class T>
class PyColumnSingle : public PyColumn<T>
{
protected:
    std::vector<T>* _pData;

public:
    PyColumnSingle(const int& kind, size_t numRows = 0);
    virtual ~PyColumnSingle();
    virtual void SetAt(size_t nRow, size_t nCol, const T& value);
    virtual void AddToDict(pb::dict& dict,
                           const std::string& name,
                           const std::vector<std::string>* keyNames,
                           const size_t expectedRows);
    virtual size_t GetNumRows();
    virtual size_t GetNumCols();
    const std::vector<T>* GetData() const { return _pData; }
};

template <class T>
inline PyColumnSingle<T>::PyColumnSingle(const int& kind, size_t numRows)
    : PyColumn<T>(kind)
{
    _pData = new std::vector<T>();
    if (numRows > 0) {
        _pData->reserve(numRows);
    }
}

template <class T>
inline PyColumnSingle<T>::~PyColumnSingle()
{
    delete _pData;
}

template <class T>
inline void PyColumnSingle<T>::SetAt(size_t nRow, size_t nCol, const T& value)
{
    if (_pData->size() <= nRow)
        _pData->resize(nRow + 1);
    _pData->at(nRow) = value;
}

template <class T>
inline size_t PyColumnSingle<T>::GetNumRows()
{
    return _pData->size();
}

template <class T>
inline size_t PyColumnSingle<T>::GetNumCols()
{
    return 1;
}

typedef pb::object NullableString;

/*
 * Handles the variable value case.
 */
template <class T, class T2>
class PyColumnVariable : public PyColumn<T>
{
private:
    std::vector<std::vector<T2>*> _data;

    size_t _numRows;
    size_t _numDeletedColumns;

public:
    PyColumnVariable(const int& kind, size_t numRows = 0);
    virtual ~PyColumnVariable();
    virtual void SetAt(size_t nRow, size_t nCol, const T& value);
    virtual void AddToDict(pb::dict& dict,
                           const std::string& name,
                           const std::vector<std::string>* keyNames,
                           const size_t expectedRows);
    virtual size_t GetNumRows();
    virtual size_t GetNumCols();

    T2 GetMissingValue();
    T2 GetConvertedValue(const T& value);

    void AddColumnToDict(pb::dict& dict, const std::string& name, size_t index);

public:
    typedef struct
    {
        PyColumnVariable* instance;
        size_t column;
    } DeleteData;

    static void Deleter(PyObject* obj);
};

template <class T, class T2>
inline PyColumnVariable<T, T2>::PyColumnVariable(const int& kind, size_t numRows)
    : PyColumn<T>(kind),
      _numRows(numRows),
      _numDeletedColumns(0)
{
}

template <class T, class T2>
inline PyColumnVariable<T, T2>::~PyColumnVariable()
{
    for (unsigned int i = 0; i < _data.size(); i++)
    {
        if (_data[i] != nullptr) delete _data[i];
    }
}

template <class T, class T2>
inline size_t PyColumnVariable<T, T2>::GetNumRows()
{
    return _numRows;
}

template <class T, class T2>
inline size_t PyColumnVariable<T, T2>::GetNumCols()
{
    return _data.size();
}

template <class T, class T2>
inline T2 PyColumnVariable<T, T2>::GetMissingValue()
{
    return NAN;
}

template <class T, class T2>
inline T2 PyColumnVariable<T, T2>::GetConvertedValue(const T& value)
{
    return (T2)value;
}

template <>
inline NullableString PyColumnVariable<std::string, NullableString>::GetMissingValue()
{
    return pb::cast<pb::none>(Py_None);;
}

template <>
inline NullableString PyColumnVariable<std::string, NullableString>::GetConvertedValue(const std::string& value)
{
    return pb::str(value);
}
