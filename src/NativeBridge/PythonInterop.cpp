// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <math.h> 
#include "stdafx.h"
#include "PythonInterop.h"


inline void destroyManagerCObject(PyObject* obj) {
    auto* b = static_cast<PyColumnBase*>(PyCapsule_GetPointer(obj, NULL));
    if (b) { delete b; }
}


PyColumnBase::PyColumnBase(const int& kind, size_t numRows, size_t numCols)
{
    _kind = kind;
    _numRows = numRows;
    _numCols = numCols;
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
    return map;
}

PyColumnBase::creation_map* PyColumnBase::CreateVariableMap()
{
    PyColumnBase::creation_map* map = new PyColumnBase::creation_map();

    map->insert(creation_map_entry(BL, CreateVariable<signed char, float>));
    map->insert(creation_map_entry(I1, CreateVariable<signed char, float>));
    map->insert(creation_map_entry(I2, CreateVariable<signed short, float>));
    map->insert(creation_map_entry(I4, CreateVariable<signed int, float>));
    map->insert(creation_map_entry(I8, CreateVariable<CxInt64, double>));
    map->insert(creation_map_entry(U1, CreateVariable<unsigned char, float>));
    map->insert(creation_map_entry(U2, CreateVariable<unsigned short, float>));
    map->insert(creation_map_entry(U4, CreateVariable<unsigned int, float>));
    map->insert(creation_map_entry(U8, CreateVariable<CxUInt64, double>));
    map->insert(creation_map_entry(R4, CreateVariable<float, float>));
    map->insert(creation_map_entry(R8, CreateVariable<double, double>));
    //map->insert(creation_map_entry(TX, CreateVariable<std::string>));
    return map;
}

PyColumnBase* PyColumnBase::Create(const int& kind, size_t numRows, size_t numCols)
{
    if (numCols == 0)
    {
        creation_map::iterator found = m_pVariableCreationMap->find(kind);
        if (found != m_pVariableCreationMap->end())
            return found->second(kind, numRows, numCols);
    }
    else
    {
        creation_map::iterator found = m_pSingleCreationMap->find(kind);
        if (found != m_pSingleCreationMap->end())
            return found->second(kind, numRows, numCols);
    }

    std::stringstream message;
    message << "Columns of kind " << kind << " are not supported.";
    throw std::invalid_argument(message.str().c_str());
}

template <class T> PyColumnBase* PyColumnBase::CreateSingle(const int& kind, size_t nRows, size_t nColumns)
{
    return new PyColumnSingle<T>(kind, nRows, nColumns);
}

template <class T, class T2> PyColumnBase* PyColumnBase::CreateVariable(const int& kind, size_t nRows, size_t nColumns)
{
    return new PyColumnVariable<T, T2>(kind, nRows, nColumns);
}

template <class T>
void PyColumnSingle<T>::AddToDict(bp::dict& dict,
                                  const std::string& name,
                                  const std::vector<std::string>* keyNames,
                                  const size_t expectedRows)
{
    auto* data = _pData->data();

    switch (this->_kind)
    {
    case DataKind::BL:
    {
        bp::handle<> h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        dict[name] = np::from_data(
            data,
            np::dtype::get_builtin<bool>(),
            bp::make_tuple(_pData->size()),
            bp::make_tuple(sizeof(bool)), bp::object(h));
    }
    break;
    case DataKind::I1:
    case DataKind::I2:
    case DataKind::I4:
    {
        bp::handle<> h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        np::ndarray npdata = np::from_data(
            data,
            np::dtype::get_builtin<T>(),
            bp::make_tuple(_pData->size()),
            bp::make_tuple(sizeof(float)), bp::object(h));
        if (keyNames == nullptr)
        {
            dict[name] = npdata;
        }
        else
        {
            dict[name] = bp::dict();
            dict[name]["..Data"] = npdata;
            bp::list list;
            for (int j = 0; j < keyNames->size(); j++)
            {
                bp::object obj;
                const std::string& value = keyNames->at(j);
                if (!value.empty())
                {
                    obj = bp::object(value);
                }
                list.append(obj);
            }
            dict[name]["..KeyValues"] = list;
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
        bp::handle<> h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        dict[name] = np::from_data(
            data,
            np::dtype::get_builtin<T>(),
            bp::make_tuple(_pData->size()),
            bp::make_tuple(sizeof(T)), bp::object(h));
    }
    break;
    }
}

template <>
void PyColumnSingle<std::string>::AddToDict(bp::dict& dict,
                                            const std::string& name,
                                            const std::vector<std::string>* keyNames,
                                            const size_t expectedRows)
{
    bp::list list;
    for (size_t i = 0; i < _pData->size(); i++)
    {
        bp::object obj;
        const std::string& value = _pData->at(i);
        if (!value.empty())
        {
            obj = bp::object(value);
        }
        list.append(obj);
    }
    dict[name] = list;
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
void PyColumnVariable<T, T2>::AddToDict(bp::dict& dict,
                                        const std::string& name,
                                        const std::vector<std::string>* keyNames,
                                        const size_t expectedRows)
{
    size_t numRows = (expectedRows > this->_numRows) ? expectedRows : this->_numRows;
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
        /*
         * Make sure all the columns are the same length.
         */
        for (size_t j = _data[i]->size(); j < numRows; j++)
        {
            _data[i]->push_back(GetMissingValue());
        }

        std::string colName = std::to_string(i);
        colName = std::string(maxDigits - colName.length(), '0') + colName;
        colName = colNameBase + colName;

        AddColumnToDict(dict, colName, i);
    }
}

template<class T, class T2>
void PyColumnVariable<T, T2>::AddColumnToDict(bp::dict& dict,
                                              const std::string& name,
                                              size_t index)
{
    auto* data = _data[index]->data();

    DeleteData* deleteData = new DeleteData();
    deleteData->instance = this;
    deleteData->column = index;

    bp::handle<> h(::PyCapsule_New((void*)deleteData, NULL, (PyCapsule_Destructor)&Deleter));
    dict[name] = np::from_data(
        data,
        np::dtype::get_builtin<T2>(),
        bp::make_tuple(_data[index]->size()),
        bp::make_tuple(sizeof(T2)), bp::object(h));
}
