// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "stdafx.h"
#include <iostream>
#include "DataViewInterop.h"
#include "ManagedInterop.h"


#define AddToDict(type); \
        {\
            PyColumn<type>* col = dynamic_cast<PyColumn<type>*>(column);\
            col->AddToDict(dict, _names[i], keyNames, maxRows);\
        }\

#define STATIC


EnvironmentBlock::~EnvironmentBlock()
{
    // Everything (except data buffers) that we might have exposed to managed code,
    // fill with dead values.
    FillDead(this->verbosity);
    FillDead(this->seed);
    FillDead(this->maxSlots);
    FillDead(this->messageSink);
    FillDead(this->modelSink);
    FillDead(this->checkCancel);

    for (size_t i = 0; i < _vset.size(); i++)
        FillDead(_vset[i]);
}

EnvironmentBlock::EnvironmentBlock(int verbosity, int maxSlots, int seed, const char* pythonPath)
{
    // Assert that this class doesn't have a vtable.
    assert(offsetof(EnvironmentBlock, verbosity) == 0);

    this->verbosity = verbosity;
    this->maxSlots = maxSlots;
    this->seed = seed;
    this->pythonPath = pythonPath;
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

    for (int i = 0; i < pdata->ccol; i++)
    {
        BYTE kind = pdata->kinds[i];
        _columns.push_back(PyColumnBase::Create(kind, pdata->crow, pdata->valueCounts[i]));

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
        case DT:
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
        case DZ:  // tbd
        default:
            throw std::invalid_argument("data type is not supported " + std::to_string(kind));
        }

        if (pdata->keyCards && (pdata->keyCards[i] >= 0))
        {
            _columnToKeyMap.insert(i);
            _vKeyValues.push_back(new PyColumnSingle<std::string>(TX, pdata->keyCards[i]));
        }

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
            env->_errMessage = sMessage;
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

pb::dict EnvironmentBlock::GetData()
{
    if (_columns.size() == 0)
    {
        return pb::dict();
    }

    size_t maxRows = 0;
    for (size_t i = 0; i < _columns.size(); i++)
    {
        size_t numRows = _columns[i]->GetNumRows();
        if (numRows > maxRows) maxRows = numRows;
    }

    CxInt64 numKeys = 0;
    pb::dict dict = pb::dict();
    for (size_t i = 0; i < _columns.size(); i++)
    {
        PyColumnBase* column = _columns[i];
        const std::vector<std::string>* keyNames = nullptr;
        if (_columnToKeyMap.find(i) != _columnToKeyMap.end())
            keyNames = _vKeyValues[numKeys++]->GetData();

        signed char kind = column->GetKind();
        switch (kind) {
        case -1:
        {
            PyColumnSingle<signed char>* col = dynamic_cast<PyColumnSingle<signed char>*>(column);
            auto shrd = col->GetData();
            pb::list list;
            for (size_t i = 0; i < shrd->size(); i++)
            {
                pb::object obj;
                signed char value = shrd->at(i);
                if (value < 0)
                    obj = pb::cast<double>(NAN);
                else if (value == 0)
                    obj = pb::cast<bool>(false);
                else
                    obj = pb::cast<bool>(true);

                list.append(obj);
            }
            dict[pb::str(_names[i])] = list;

        }
        break;
        case BL:
            AddToDict(signed char);
            break;
        case I1:
            AddToDict(signed char);
            break;
        case I2:
            AddToDict(signed short);
            break;
        case I4:
            AddToDict(signed int);
            break;
        case I8:
            AddToDict(CxInt64);
            break;
        case U1:
            AddToDict(unsigned char);
            break;
        case U2:
            AddToDict(unsigned short);
            break;
        case U4:
            AddToDict(unsigned int);
            break;
        case U8:
            AddToDict(CxUInt64);
            break;
        case R4:
            AddToDict(float);
            break;
        case R8:
            AddToDict(double);
            break;
        case TX:
            AddToDict(std::string);
            delete column;
            break;
        case DT:
            AddToDict(CxInt64);
            break;
        case TS:
        case DZ:
        default:
            throw std::invalid_argument("data type is not supported " + std::to_string(kind));
        }
    }
    return dict;
}
