// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <map>

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


template <class T>
class PythonObject : public PythonObjectBase
{
protected:
	std::shared_ptr<std::vector<T>> _pData;

	size_t _numRows;
	size_t _numCols;

public:
	PythonObject(const int& kind, size_t numRows = 1, size_t numCols = 1);
	virtual ~PythonObject();
	void SetAt(size_t nRow, size_t nCol, const T& value);
	const std::shared_ptr<std::vector<T> >& GetData() const;
};

template <class T>
inline PythonObject<T>::PythonObject(const int& kind, size_t numRows, size_t numCols)
	: PythonObjectBase(kind)
{
	_numRows = numRows;
	_numCols = numCols;

	_pData = std::make_shared<std::vector<T>>();
	if (_numRows > 0)
		_pData->reserve(_numRows*_numCols);
}

template <class T>
inline PythonObject<T>::~PythonObject()
{
}

template <class T>
inline void PythonObject<T>::SetAt(size_t nRow, size_t nCol, const T& value)
{
	size_t index = nRow*_numCols + nCol;
	if (_pData->size() <= index)
		_pData->resize(index + 1);
	_pData->at(index) = value;
}

template <class T>
inline const std::shared_ptr<std::vector<T>>& PythonObject<T>::GetData() const
{
	return _pData;
}