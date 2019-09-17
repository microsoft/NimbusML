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

class PythonObjectBase
{
private:
    typedef std::map<int, PythonObjectBase* (*)(const int&, size_t, size_t)> creation_map;
    typedef std::pair<int, PythonObjectBase* (*)(const int&, size_t, size_t)> creation_map_entry;

    static creation_map* m_pSingleCreationMap;
    static creation_map* CreateSingleMap();

    static creation_map* m_pVariableCreationMap;
    static creation_map* CreateVariableMap();

    template <class T> static PythonObjectBase* CreateSingle(const int& name, size_t nRows, size_t nColumns);
    template <class T, class T2> static PythonObjectBase* CreateVariable(const int& name, size_t nRows, size_t nColumns);

protected:
    int _kind;
    size_t _numRows;
    size_t _numCols;

public:
    static PythonObjectBase* CreateObject(const int& kind, size_t numRows, size_t numCols);

    PythonObjectBase(const int& kind, size_t numRows = 0, size_t numCols = 0);
    virtual ~PythonObjectBase();

    const int& GetKind() const;
    void SetKind(int kind);

    virtual size_t GetNumRows();
    virtual size_t GetNumCols();
};

inline const int& PythonObjectBase::GetKind() const
{
    return _kind;
}

inline void PythonObjectBase::SetKind(int kind)
{
    _kind = kind;
}

inline size_t PythonObjectBase::GetNumRows()
{
    return _numRows;
}

inline size_t PythonObjectBase::GetNumCols()
{
    return _numCols;
}


/*
 * Template typed base class which provides the
 * required interface for all derived classes.
 */
template <class T>
class PythonObject : public PythonObjectBase
{
public:
    PythonObject(const int& kind, size_t numRows = 0, size_t numCols = 0);
    virtual ~PythonObject() {}
    virtual void SetAt(size_t nRow, size_t nCol, const T& value) = 0;
    virtual void AddToDict(bp::dict& dict,
                           const std::string& name,
                           const std::vector<std::string>* keyNames,
                           const size_t expectedRows) = 0;
};

template <class T>
inline PythonObject<T>::PythonObject(const int& kind, size_t numRows, size_t numCols)
    : PythonObjectBase(kind, numRows, numCols)
{
}


/*
 * PythonObject based class which
 * handles the single column case.
 */
template <class T>
class PythonObjectSingle : public PythonObject<T>
{
protected:
    std::vector<T>* _pData;

public:
    PythonObjectSingle(const int& kind, size_t numRows = 0, size_t numCols = 1);
    virtual ~PythonObjectSingle();
    virtual void SetAt(size_t nRow, size_t nCol, const T& value);
    virtual void AddToDict(bp::dict& dict,
                           const std::string& name,
                           const std::vector<std::string>* keyNames,
                           const size_t expectedRows);
    virtual size_t GetNumRows();
    virtual size_t GetNumCols();
    const std::vector<T>* GetData() const { return _pData; }
};

template <class T>
inline PythonObjectSingle<T>::PythonObjectSingle(const int& kind, size_t numRows, size_t numCols)
    : PythonObject<T>(kind, numRows, numCols)
{
    _pData = new std::vector<T>();
    if (numRows > 0)
        _pData->reserve(numRows*numCols);
}

template <class T>
inline PythonObjectSingle<T>::~PythonObjectSingle()
{
    delete _pData;
}

template <class T>
inline void PythonObjectSingle<T>::SetAt(size_t nRow, size_t nCol, const T& value)
{
    size_t index = nRow * this->_numCols + nCol;
    if (_pData->size() <= index)
        _pData->resize(index + 1);
    _pData->at(index) = value;
}

template <class T>
inline size_t PythonObjectSingle<T>::GetNumRows()
{
    return _pData->size();
}

template <class T>
inline size_t PythonObjectSingle<T>::GetNumCols()
{
    return 1;
}


/*
 * PythonObject based class which
 * handles the variable column case.
 */
template <class T, class T2>
class PythonObjectVariable : public PythonObject<T>
{
private:
    std::vector<std::vector<T2>*> _data;
    size_t _numDeletedColumns;

public:
    PythonObjectVariable(const int& kind, size_t numRows = 0, size_t numCols = 0);
    virtual ~PythonObjectVariable();
    virtual void SetAt(size_t nRow, size_t nCol, const T& value);
    virtual void AddToDict(bp::dict& dict,
                           const std::string& name,
                           const std::vector<std::string>* keyNames,
                           const size_t expectedRows);
    virtual size_t GetNumCols();

public:
    typedef struct
    {
        PythonObjectVariable* instance;
        size_t column;
    } DeleteData;

    static void Deleter(PyObject* obj);
};

template <class T, class T2>
inline PythonObjectVariable<T, T2>::PythonObjectVariable(const int& kind, size_t numRows, size_t numCols)
    : PythonObject<T>(kind, numRows, numCols),
      _numDeletedColumns(0)
{
}

template <class T, class T2>
inline PythonObjectVariable<T, T2>::~PythonObjectVariable()
{
    for (unsigned int i = 0; i < _data.size(); i++)
    {
        if (_data[i] != nullptr) delete _data[i];
    }
}

template <class T, class T2>
inline void PythonObjectVariable<T, T2>::SetAt(size_t nRow, size_t nCol, const T& value)
{
    if ((nRow + 1) > this->_numRows) this->_numRows = nRow + 1;

    /*
     * Make sure there are enough columns for the request.
     */
    for (size_t i = _data.size(); i < (nCol + 1); i++)
    {
        _data.push_back(new std::vector<T2>());
    }

    /*
     * Fill in any missing row values in
     * the specified column with NAN.
     */
    for (size_t i = _data[nCol]->size(); i < nRow; i++)
    {
        _data[nCol]->push_back(NAN);
    }

    _data[nCol]->push_back((T2)value);
}

template <class T, class T2>
inline size_t PythonObjectVariable<T, T2>::GetNumCols()
{
    return _data.size();
}
