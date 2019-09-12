// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "stdafx.h"
#include "PythonInterop.h"


inline void destroyManagerCObject(PyObject* obj) {
    auto* b = static_cast<PythonObjectBase*>(PyCapsule_GetPointer(obj, NULL));
    if (b) { delete b; }
}


PythonObjectBase::PythonObjectBase(const int& kind)
{
    _kind = kind;
}

PythonObjectBase::~PythonObjectBase()
{
}

PythonObjectBase::creation_map* PythonObjectBase::m_pCreationMap = PythonObjectBase::CreateMap();

PythonObjectBase::creation_map* PythonObjectBase::CreateMap()
{
    PythonObjectBase::creation_map* map = new PythonObjectBase::creation_map();

    map->insert(creation_map_entry(BL, CreateObject<signed char>));
    map->insert(creation_map_entry(I1, CreateObject<signed char>));
    map->insert(creation_map_entry(I2, CreateObject<signed short>));
    map->insert(creation_map_entry(I4, CreateObject<signed int>));
    map->insert(creation_map_entry(I8, CreateObject<CxInt64>));
    map->insert(creation_map_entry(U1, CreateObject<unsigned char>));
    map->insert(creation_map_entry(U2, CreateObject<unsigned short>));
    map->insert(creation_map_entry(U4, CreateObject<unsigned int>));
    map->insert(creation_map_entry(U8, CreateObject<CxUInt64>));
    map->insert(creation_map_entry(R4, CreateObject<float>));
    map->insert(creation_map_entry(R8, CreateObject<double>));
    map->insert(creation_map_entry(TX, CreateObject<std::string>));
    return map;
}

PythonObjectBase* PythonObjectBase::CreateObject(const int& kind, size_t numRows, size_t numCols)
{
    creation_map::iterator found = m_pCreationMap->find(kind);

    if (found == m_pCreationMap->end())
    {
        std::stringstream message;
        message << "Columns of kind " << kind << " are not supported.";
        throw std::invalid_argument(message.str().c_str());
    }

    return found->second(kind, numRows, numCols);
}

template <class T> PythonObjectBase* PythonObjectBase::CreateObject(const int& kind, size_t nRows, size_t nColumns)
{
    return new PythonObjectSingle<T>(kind, nRows, nColumns);
}

template <class T>
void PythonObjectSingle<T>::AddToDict(bp::dict& dict, const std::string& name, const std::vector<std::string>* keyNames)
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
void PythonObjectSingle<std::string>::AddToDict(bp::dict& dict, const std::string& name, const std::vector<std::string>* keyNames)
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
