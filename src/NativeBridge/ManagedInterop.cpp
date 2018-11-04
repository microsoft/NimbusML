// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "stdafx.h"
#include <iostream>
#include "DataViewInterop.h"
#include "ManagedInterop.h"

#ifdef BOOST_PYTHON
#define SetDict2(cpptype, nptype); \
		{\
			PythonObject<cpptype>* col = dynamic_cast<PythonObject<cpptype>*>(column);\
			auto shrd = col->GetData();\
			auto* data = shrd->data();\
			dict[_names[i]] = np::from_data(\
				data,\
				np::dtype::get_builtin<nptype>(),\
				bp::make_tuple(shrd->size()),\
				bp::make_tuple(sizeof(nptype)), bp::object());\
		}
#else
#define SetDict2(cpptype, nptype); \
		{\
            PythonObject<cpptype>* col = dynamic_cast<PythonObject<cpptype>*>(column); \
            auto shrd = col->GetData(); \
            auto* data = shrd->data(); \
            dict[_names[i].c_str()] = bp::array(shrd->size(), data); \
        }
#endif

#define SetDict1(type) SetDict2(type, type)

#ifdef BOOST_PYTHON
#define SetDictAndKeys(type, i); \
		{\
			PythonObject<type>* col = dynamic_cast<PythonObject<type>*>(column);\
			auto shrd = col->GetData();\
			auto* data = shrd->data();\
			np::ndarray npdata = np::from_data(\
				data,\
				np::dtype::get_builtin<type>(),\
				bp::make_tuple(shrd->size()),\
				bp::make_tuple(sizeof(float)), bp::object());\
			if (keyNames == nullptr)\
			{\
				dict[_names[i]] = npdata;\
			}\
			else\
			{\
				dict[_names[i]] = bp::dict();\
				dict[_names[i]]["..Data"] = npdata;\
				auto shrd = keyNames->GetData();\
				bp::list list;\
				for (int j = 0; j < shrd->size(); j++)\
				{\
					bp::object obj;\
					const std::string& value = shrd->at(j);\
					if (!value.empty())\
					{\
						obj = bp::object(value);\
					}\
					list.append(obj);\
				}\
				dict[_names[i]]["..KeyValues"] = list;\
			}\
		}
#else
#define SetDictAndKeys(type, i); \
        {\
            PythonObject<type>* col = dynamic_cast<PythonObject<type>*>(column);\
            auto shrd = col->GetData();\
            auto* data = shrd->data();\
            bp::array npdata = bp::array(shrd->size(), data);\
            if (keyNames == nullptr)\
                dict[_names[i].c_str()] = npdata;\
            else\
            {\
                dict[_names[i].c_str()] = bp::dict(); \
                dict[_names[i].c_str()]["..Data"] = npdata;\
                auto shrd = keyNames->GetData();\
                bp::list list;\
                for (int j = 0; j < shrd->size(); j++)\
                {\
                    bp::object obj;\
                    const std::string& value = shrd->at(j);\
                    if (!value.empty())\
                        obj = bp::str(value);\
                    list.append(obj);\
                }\
                dict[_names[i].c_str()]["..KeyValues"] = list;\
            }\
        }
#endif

#define STATIC

EnvironmentBlock::~EnvironmentBlock()
{
	// Everything (except data buffers) that we might have exposed to managed code,
	// fill with dead values.
	FillDead(this->verbosity);
	FillDead(this->seed);
	FillDead(this->messageSink);
	FillDead(this->modelSink);
	FillDead(this->checkCancel);

	for (size_t i = 0; i < _vset.size(); i++)
		FillDead(_vset[i]);
}

EnvironmentBlock::EnvironmentBlock(int verbosity, int maxThreadsAllowed, int seed)
{
	// Assert that this class doesn't have a vtable.
	assert(offsetof(EnvironmentBlock, verbosity) == 0);

	this->_errCode = PyErrorCode_NoError;
	this->verbosity = verbosity;
	this->maxThreadsAllowed = maxThreadsAllowed;
	this->seed = seed;
	this->_kindMask = (1 << Warning) | (1 << Error);
	if (verbosity > 0)
		this->_kindMask |= (1 << Info);
	if (this->verbosity > 3)
		this->_kindMask |= (1 << Trace);
	this->dataSink = &DataSink;
	this->messageSink = &MessageSink;
	this->modelSink = &ModelSink;
	this->checkCancel = &CheckCancel;
}

STATIC MANAGED_CALLBACK(void) EnvironmentBlock::DataSink(EnvironmentBlock *penv, const DataViewBlock *pdata, void **&setters, void *&keyValueSetter)
{
	penv->DataSinkCore(pdata);
	setters = &penv->_vset[0];
	keyValueSetter = (void *)&SetKeyValue;
}

void EnvironmentBlock::DataSinkCore(const DataViewBlock * pdata)
{
	assert(pdata != nullptr);

	// Create a data set.
	CxInt64 numKeys = 0;
	for (int i = 0; i < pdata->ccol; i++)
	{
		BYTE kind = pdata->kinds[i];
		_columns.push_back(PythonObjectBase::CreateObject(kind, pdata->crow, 1));

		switch (kind)
		{
		case BL:
			_vset.push_back((void*)&SetBL);
			break;
		case I1:
			_vset.push_back((void*)&SetI1);
			break;
		case I2:
			_vset.push_back((void*)&SetI2);
			break;
		case I4:
			_vset.push_back((void*)&SetI4);
			break;
		case I8:
			_vset.push_back((void*)&SetI8);
			break;
		case U1:
			_vset.push_back((void*)&SetU1);
			break;
		case U2:
			_vset.push_back((void*)&SetU2);
			break;
		case U4:
			_vset.push_back((void*)&SetU4);
			break;
		case U8:
			_vset.push_back((void*)&SetU8);
			break;
		case R4:
			_vset.push_back((void*)&SetR4);
			break;
		case R8:
			_vset.push_back((void*)&SetR8);
			break;
		case TX:
			_vset.push_back((void*)&SetTX);
			break;
		case TS:  // tbd
		case DT:  // tbd
		case DZ:  // tbd
		default:
			throw std::invalid_argument("data type is not supported " + std::to_string(kind));
		}

		if (pdata->keyCards[i] >= 0)
		{
			_vKeyValues.push_back(new PythonObject<std::string>(TX, pdata->keyCards[i], 1));
			_columnToKeyMap.push_back(numKeys++);
		}
		else
			_columnToKeyMap.push_back(-1);

		_names.push_back(pdata->names[i]);
	}
}

STATIC MANAGED_CALLBACK(void) EnvironmentBlock::ModelSink(EnvironmentBlock * env,
	const unsigned char * pBinaryModel, size_t iModelLen)
{
}


STATIC MANAGED_CALLBACK(void) EnvironmentBlock::MessageSink(EnvironmentBlock * env, MessageKind kind,
	const char * sender, const char * message)
{
	bool bShowMessage = (env->_kindMask >> kind) & 1;
	string sMessage(message);
	string sSender(sender);

	if (bShowMessage)
	{
		CX_TraceIn("MessageSink");
		string sMessage = std::string(message);
		string sSender = std::string(sender);

		switch (kind)
		{
		default:
		case Info:
			sMessage = sMessage + "\n";
			break;
		case Warning:
			sMessage = "Warning: " + sMessage + "\n";
			break;
		case Trace:
			sMessage = sSender + ": " + sMessage + "\n";
			break;
		case Error:  // We will throw the error when ConnectToMlNet returns
			sMessage = "Error: " + sMessage;
			break;
		}

		// Redirect message to Python streams
		PyObject *sys = PyImport_ImportModule("sys");
		PyObject *pystream = PyObject_GetAttrString(sys, (kind == Error) ? "stderr" : "stdout");
		PyObject_CallMethod(pystream, "write", "s", sMessage.c_str());
		PyObject_CallMethod(pystream, "flush", NULL);
		Py_XDECREF(pystream);
		Py_XDECREF(sys);

		CX_TraceOut("MessageSink");
	}
}

STATIC MANAGED_CALLBACK(bool) EnvironmentBlock::CheckCancel()
{
	return false;
}

bp::dict EnvironmentBlock::GetData()
{
	if (_names.size() == 0)
	{
		return bp::dict();
	}

	bp::dict dict = bp::dict();
	for (size_t i = 0; i < _names.size(); i++)
	{
		PythonObjectBase* column = _columns[i];
		PythonObject<std::string>* keyNames = nullptr;
		if (_columnToKeyMap[i] >= 0)
			keyNames = _vKeyValues[_columnToKeyMap[i]];

		signed char kind = column->GetKind();
		switch (kind) {
		case -1:
		{
			PythonObject<signed char>* col = dynamic_cast<PythonObject<signed char>*>(column);
			auto shrd = col->GetData();
			bp::list list;
			for (size_t i = 0; i < shrd->size(); i++)
			{
				bp::object obj;
				signed char value = shrd->at(i);
#ifdef BOOST_PYTHON
                if (value < 0)
                    obj = bp::object(NAN);
                else if (value == 0)
                    obj = bp::object(false);
                else
					obj = bp::object(true);
#else
                if (value < 0)
                    obj = bp::cast<double>(NAN);
                else if (value == 0)
                    obj = bp::cast<bool>(false);
                else
                    obj = bp::cast<bool>(true);
#endif

				list.append(obj);
			}
#ifdef BOOST_PYTHON
            dict[_names[i]] = list;
#else
			dict[_names[i].c_str()] = list;
#endif
		}
		break;
		case BL:
			SetDict2(signed char, bool);
			break;
		case I1:
			SetDictAndKeys(signed char, i);
			break;
		case I2:
			SetDictAndKeys(signed short, i);
			break;
		case I4:
			SetDictAndKeys(signed int, i);
			break;
		case I8:
			SetDict1(CxInt64);
			break;
		case U1:
			SetDict1(unsigned char);
			break;
		case U2:
			SetDict1(unsigned short);
			break;
		case U4:
			SetDict1(unsigned int);
			break;
		case U8:
			SetDict1(CxUInt64);
			break;
		case R4:
			SetDict1(float);
			break;
		case R8:
			SetDict1(double);
			break;
		case TX:
		{
			PythonObject<string>* col = dynamic_cast<PythonObject<string>*>(column);
			auto shrd = col->GetData();
			bp::list list;
			for (size_t i = 0; i < shrd->size(); i++)
			{
				bp::object obj;
				const std::string& value = shrd->at(i);
				if (!value.empty())
#ifdef BOOST_PYTHON
                    obj = bp::object(value);
#else
					obj = bp::str(value);
#endif
				list.append(obj);
			}
#ifdef BOOST_PYTHON
			dict[_names[i]] = list;
#else
            dict[_names[i].c_str()] = list;
#endif
		}
		break;
		case TS:
		case DT:
		case DZ:
		default:
			throw std::invalid_argument("data type is not supported " + std::to_string(kind));
		}
	}
	return dict;
}
