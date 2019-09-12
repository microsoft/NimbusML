// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "stdafx.h"
#include "PythonInterop.h"

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
    map->insert(creation_map_entry(I2, CreateObject<short>));
    map->insert(creation_map_entry(I4, CreateObject<int>));
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

