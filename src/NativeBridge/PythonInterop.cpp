// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "stdafx.h"
#include "PythonInterop.h"


inline void destroyManagerCObject(PyObject* obj) {
    auto* b = static_cast<PythonObjectBase*>(PyCapsule_GetPointer(obj, NULL));
    if (b) { delete b; }
}


PythonObjectBase::PythonObjectBase(const int& kind, size_t numRows = 0, size_t numCols = 0)
{
    _kind = kind;
    _numRows = numRows;
    _numCols = numCols;
}

PythonObjectBase::~PythonObjectBase()
{
}

PythonObjectBase::creation_map* PythonObjectBase::m_pSingleCreationMap = PythonObjectBase::CreateSingleMap();
PythonObjectBase::creation_map* PythonObjectBase::m_pVariableCreationMap = PythonObjectBase::CreateVariableMap();

PythonObjectBase::creation_map* PythonObjectBase::CreateSingleMap()
{
    PythonObjectBase::creation_map* map = new PythonObjectBase::creation_map();

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

PythonObjectBase::creation_map* PythonObjectBase::CreateVariableMap()
{
    PythonObjectBase::creation_map* map = new PythonObjectBase::creation_map();

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

PythonObjectBase* PythonObjectBase::CreateObject(const int& kind, size_t numRows, size_t numCols)
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

template <class T> PythonObjectBase* PythonObjectBase::CreateSingle(const int& kind, size_t nRows, size_t nColumns)
{
    return new PythonObjectSingle<T>(kind, nRows, nColumns);
}

template <class T, class T2> PythonObjectBase* PythonObjectBase::CreateVariable(const int& kind, size_t nRows, size_t nColumns)
{
    return new PythonObjectVariable<T, T2>(kind, nRows, nColumns);
}

template <class T>
void PythonObjectSingle<T>::AddToDict(bp::dict& dict,
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
void PythonObjectSingle<std::string>::AddToDict(bp::dict& dict,
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

template<class T, class T2>
inline void PythonObjectVariable<T, T2>::AddToDict(bp::dict& dict,
                                                   const std::string& name,
                                                   const std::vector<std::string>* keyNames,
                                                   const size_t expectedRows)
{
    const size_t numRows = (expectedRows > this->_numRows) ? expectedRows : this->_numRows;
    const size_t numCols = _data.size();

    if (numCols == 0)
    {
        /*
         * If there were no values set then create
         * a column so it can be filled with NANs.
         */
        _data.push_back(new std::vector<T2>());
        numCols = 1;
    }

    for (size_t i = 0; i < numCols; i++)
    {
        /*
         * Make sure all the columns are the same length.
         */
        for (size_t j = _data[i]->size(); j < numRows; j++)
        {
            _data[i]->push_back(NAN);
        }

        std::string colName = name + "." + std::to_string(i);

        auto* data = _data[i]->data();

        bp::handle<> h(::PyCapsule_New((void*)this, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
        dict[colName] = np::from_data(
            data,
            np::dtype::get_builtin<T2>(),
            bp::make_tuple(_data[i]->size()),
            bp::make_tuple(sizeof(T2)), bp::object(h));
    }
}
