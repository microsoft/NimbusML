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

    static creation_map* m_pCreationMap;
    static creation_map* CreateMap();

    template <class T> static PythonObjectBase* CreateObject(const int& name, size_t nRows, size_t nColumns);

protected:
    int _kind;

public:
    PythonObjectBase(const int& kind);
    static PythonObjectBase* CreateObject(const int& kind, size_t numRows, size_t numCols);
    const int& GetKind() const;
    void SetKind(int kind);
    virtual ~PythonObjectBase();
};

inline const int& PythonObjectBase::GetKind() const
{
    return _kind;
}

inline void PythonObjectBase::SetKind(int kind)
{
    _kind = kind;
}


/*
 * Template typed base class which provides the
 * required interface for all derived classes.
 */
template <class T>
class PythonObject : public PythonObjectBase
{
protected:
    size_t _numRows;
    size_t _numCols;

public:
    PythonObject(const int& kind, size_t numRows = 1, size_t numCols = 1);
    virtual ~PythonObject() {}
    virtual void SetAt(size_t nRow, size_t nCol, const T& value) = 0;
    virtual void AddToDict(bp::dict& dict, const std::string& name, const std::vector<std::string>* keyNames) = 0;
};

template <class T>
inline PythonObject<T>::PythonObject(const int& kind, size_t numRows, size_t numCols)
    : PythonObjectBase(kind)
{
    _numRows = numRows;
    _numCols = numCols;
}


/*
 * PythonObject based class which
 * handles the one column case.
 */
template <class T>
class PythonObjectSingle : public PythonObject<T>
{
protected:
    std::vector<T>* _pData;

public:
    PythonObjectSingle(const int& kind, size_t numRows = 1, size_t numCols = 1);
    virtual ~PythonObjectSingle();
    virtual void SetAt(size_t nRow, size_t nCol, const T& value);
    virtual void AddToDict(bp::dict& dict, const std::string& name, const std::vector<std::string>* keyNames);
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
