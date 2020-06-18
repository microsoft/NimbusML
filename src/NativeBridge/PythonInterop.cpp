// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <math.h> 
#include "stdafx.h"
#include "PythonInterop.h"


inline void destroyManagerCObject(PyObject* obj) {
    auto* b = static_cast<PyColumnBase*>(PyCapsule_GetPointer(obj, NULL));
    if (b) { delete b; }
}


PyColumnBase::PyColumnBase(const int& kind)
{
    _kind = kind;
}

PyColumnBase::~PyColumnBase()
{
}

PyColumnBase::creation_map* PyColumnBase::m_pSingleCreationMap = PyColumnBase::CreateSingleMap();
PyColumnBase::creation_map* PyColumnBase::m_pVariableCreationMap = PyColumnBase::CreateVariableMap();

PyColumnBase::creation_map* PyColumnBase::CreateSingleMap()
{
    PyColumnBase::creation_map* map = new PyColumnBase::creation_map();

    map->insert(creation_map_entry(BL, CreateSingle<signed char>));
    map->insert(creation_map_entry(I1, CreateSingle<signed char>));
    map->insert(creation_map_entry(I2, CreateSingle<signed short>));
    map->insert(creation_map_entry(I4, CreateSingle<signed int>));
    map->insert(creation_map_entry(I8, CreateSingle<CxInt64>));
    map->insert(creation_map_entry(U1, CreateSingle<unsigned char>));
    map->insert(creation_map_entry(U2, CreateSingle<unsigned short>));
    map->insert(creation_map_entry(U4, CreateSingle<unsigned int>));
    map->insert(creation_map_entry(U8, CreateSingle<CxUInt64>));
    map->insert(creation_map_entry(R4, CreateSingle<float>));
    map->insert(creation_map_entry(R8, CreateSingle<double>));
    map->insert(creation_map_entry(TX, CreateSingle<std::string>));
    map->insert(creation_map_entry(DT, CreateSingle<CxInt64>));
    return map;
}

PyColumnBase::creation_map* PyColumnBase::CreateVariableMap()
{
    PyColumnBase::creation_map* map = new PyColumnBase::creation_map();

    map->insert(creation_map_entry(BL, CreateVariable<signed char, float>));
    map->insert(creation_map_entry(I1, CreateVariable<signed char, float>));
    map->insert(creation_map_entry(I2, CreateVariable<signed short, float>));
    map->insert(creation_map_entry(I4, CreateVariable<signed int, double>));
    map->insert(creation_map_entry(I8, CreateVariable<CxInt64, double>));
    map->insert(creation_map_entry(U1, CreateVariable<unsigned char, float>));
    map->insert(creation_map_entry(U2, CreateVariable<unsigned short, float>));
    map->insert(creation_map_entry(U4, CreateVariable<unsigned int, double>));
    map->insert(creation_map_entry(U8, CreateVariable<CxUInt64, double>));
    map->insert(creation_map_entry(R4, CreateVariable<float, float>));
    map->insert(creation_map_entry(R8, CreateVariable<double, double>));
    map->insert(creation_map_entry(TX, CreateVariable<std::string, NullableString>));
    return map;
}

PyColumnBase* PyColumnBase::Create(const int& kind, size_t numRows, size_t numCols)
{
    if (numCols == 0)
    {
        creation_map::iterator found = m_pVariableCreationMap->find(kind);
        if (found != m_pVariableCreationMap->end())
            return found->second(kind, numRows);
    }
    else
    {
        creation_map::iterator found = m_pSingleCreationMap->find(kind);
        if (found != m_pSingleCreationMap->end())
            return found->second(kind, numRows);
    }

    std::stringstream message;
    message << "Columns of kind " << kind << " are not supported.";
    throw std::invalid_argument(message.str().c_str());
}

template <class T> PyColumnBase* PyColumnBase::CreateSingle(const int& kind, size_t nRows)
{
    return new PyColumnSingle<T>(kind, nRows);
}

template <class T, class T2> PyColumnBase* PyColumnBase::CreateVariable(const int& kind, size_t nRows)
{
    return new PyColumnVariable<T, T2>(kind, nRows);
}

template <class T>
void PyColumnSingle<T>::AddToDict(pb::dict& dict,
                                  const std::string& name,
                                  const std::vector<std::string>* keyNames,
                                  const size_t expectedRows)
{
    auto* data = _pData->data();

    switch (this->_kind)
    {
    case DataKind::BL:
    {
        pb::handle h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        dict[pb::str(name)] = pb::array(pb::dtype("bool"), _pData->size(), data, h);
    }
    break;
    case DataKind::I1:
    case DataKind::I2:
    case DataKind::I4:
    {
        pb::handle h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        pb::array npdata = pb::array(_pData->size(), data, h);
        if (keyNames == nullptr)
        {
            dict[pb::str(name)] = npdata;
        }
        else
        {
            dict[pb::str(name)] = pb::dict();
            dict[pb::str(name)]["..Data"] = npdata;
            pb::list list;
            for (int j = 0; j < keyNames->size(); j++)
            {
                pb::object obj;
                const std::string& value = keyNames->at(j);
                if (!value.empty())
                {
                    obj = pb::str(value);
                }
                list.append(obj);
            }
            dict[pb::str(name)]["..KeyValues"] = list;
        }
    }
    break;
    case DataKind::I8:
    case DataKind::U1:
    case DataKind::U2:
    case DataKind::U4:
    case DataKind::U8:
    case DataKind::R4:
    case DataKind::R8:
    {
        pb::handle h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        dict[pb::str(name)] = pb::array(_pData->size(), data, h);
    }
    break;
    case DataKind::DT:
    {
        pb::capsule h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        pb::array npdata = pb::array(_pData->size(), data, h);

        dict[pb::str(name)] = pb::dict();
        dict[pb::str(name)]["..DateTime"] = npdata;
    }
    break;
    }
}

template <>
void PyColumnSingle<std::string>::AddToDict(pb::dict& dict,
                                            const std::string& name,
                                            const std::vector<std::string>* keyNames,
                                            const size_t expectedRows)
{
    pb::list list;
    for (size_t i = 0; i < _pData->size(); i++)
    {
        pb::object obj;
        const std::string& value = _pData->at(i);
        obj = pb::str(value);
        list.append(obj);
    }
    dict[pb::str(name)] = list;
}

template <class T, class T2>
void PyColumnVariable<T, T2>::SetAt(size_t nRow, size_t nCol, const T& value)
{
    if ((nRow + 1) > _numRows) _numRows = nRow + 1;

    /*
     * Make sure there are enough columns for the request.
     */
    for (size_t i = _data.size(); i <= nCol; i++)
    {
        _data.push_back(new std::vector<T2>());
    }

    std::vector<T2>* pColData = _data[nCol];

    /*
     * Fill in any missing row values.
     */
    for (size_t i = pColData->size(); i < nRow; i++)
    {
        pColData->push_back(GetMissingValue());
    }

    pColData->push_back(GetConvertedValue(value));
}

/*
 * Note: an instance of this object should not be used
 * and should be considered invalid after the first time
 * this method has been called.
 */
template <class T, class T2>
void PyColumnVariable<T, T2>::Deleter(PyObject* obj)
{
    auto* deleteData = static_cast<PyColumnVariable<T, T2>::DeleteData*>(PyCapsule_GetPointer(obj, NULL));

    PyColumnVariable<T, T2>* instance = deleteData->instance;
    size_t column = deleteData->column;

    std::vector<T2>* data = instance->_data[column];
    if (data != nullptr)
    {
        instance->_data[column] = nullptr;
        instance->_numDeletedColumns++;
        delete data;

        if (instance->_numDeletedColumns == instance->_data.size())
        {
            delete instance;
        }
    }
}

template<class T, class T2>
void PyColumnVariable<T, T2>::AddToDict(pb::dict& dict,
                                        const std::string& name,
                                        const std::vector<std::string>* keyNames,
                                        const size_t expectedRows)
{
    size_t numRows = (expectedRows > _numRows) ? expectedRows : _numRows;
    size_t numCols = _data.size();

    if (numCols == 0)
    {
        /*
         * If there were no values set then create a
         * column so it can be filled with missing values.
         */
        _data.push_back(new std::vector<T2>());
        numCols = 1;
    }

    const std::string colNameBase = name + ".";
    int maxDigits = (int)ceil(log10(numCols));
    if (maxDigits == 0) maxDigits = 1;

    for (size_t i = 0; i < numCols; i++)
    {
        std::vector<T2>* pColData = _data[i];

        /*
         * Make sure all the columns are the same length.
         */
        for (size_t j = pColData->size(); j < numRows; j++)
        {
            pColData->push_back(GetMissingValue());
        }

        std::string colName = std::to_string(i);
        colName = std::string(maxDigits - colName.length(), '0') + colName;
        colName = colNameBase + colName;

        AddColumnToDict(dict, colName, i);
    }
}

template<class T, class T2>
void PyColumnVariable<T, T2>::AddColumnToDict(pb::dict& dict,
                                              const std::string& name,
                                              size_t index)
{
    auto* data = _data[index]->data();

    DeleteData* deleteData = new DeleteData();
    deleteData->instance = this;
    deleteData->column = index;

    pb::handle h(::PyCapsule_New((void*)deleteData, NULL, (PyCapsule_Destructor)&Deleter));
    dict[pb::str(name)] = pb::array(_data[index]->size(), data, h);
}

template<>
void PyColumnVariable<std::string, NullableString>::AddColumnToDict(pb::dict& dict,
                                                                    const std::string& name,
                                                                    size_t index)
{
    pb::list list;
    std::vector<NullableString>* pColData = _data[index];
    size_t numRows = pColData->size();

    for (size_t i = 0; i < numRows; i++)
    {
        pb::object obj;
        NullableString value = pColData->at(i);

        if (value)
        {
            obj = value;
        }

        list.append(obj);
    }

    dict[pb::str(name)] = list;
}
