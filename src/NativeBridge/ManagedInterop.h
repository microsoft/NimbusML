// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using namespace std;
#include "stdafx.h"
#include "PythonInterop.h"

#define CX_TraceOut(...)
#define CX_TraceIn(...)

class EnvironmentBlock;
class DataSourceBlock;
struct DataViewBlock;

// WARNING: These values are defined by the ML.NET code so should not be changed!
enum MessageKind
{
    Trace = 0,
    Info = 1,
    Warning = 2,
    Error = 3
};

// REVIEW: the exceptions thrown in the callbacks will not be caught by BxlServer on Linux.
// On Linux, CoreCLR will ignore previous stack frames, i.e., those before entering the managed code.
typedef MANAGED_CALLBACK_PTR(void, MODELSINK) (EnvironmentBlock * env,
    const unsigned char * binaryModel, size_t modelLen);

typedef MANAGED_CALLBACK_PTR(void, MESSAGESINK)(EnvironmentBlock *penv, MessageKind kind,
    const char * sender, const char * message);

typedef MANAGED_CALLBACK_PTR(void, DATASINK)(EnvironmentBlock *penv, const DataViewBlock *pdata,
    // Outputs:
    // * setters: item setter function pointers.
    // keyValueSetter: setter for key values.
    void **& setters, void *& keyValueSetter);

// Callback function for getting cancel flag.
typedef MANAGED_CALLBACK_PTR(bool, CHECKCANCEL)();

// Callback function for putting string values.
typedef MANAGED_CALLBACK_PTR(void, SETSTR)(void *pv, CxInt64 index, const char * pch, int cch, long width);

// The environment services for managed interop. Some of the fields of this are visible to managed code.
// As such, it is critical that this class NOT have a vtable, so virtual functions are illegal!
class CLASS_ALIGN EnvironmentBlock
{
    // Fields that are visible to managed code come first and do not start with an underscore.
    // Fields that are only visible to this code start with an underscore.

private:
    // *** These fields are known by managed code. It is critical that this struct not have a vtable.
    //     It is also critical that the layout of this prefix NOT vary from release to release or build to build.
    //     The managed code assumes that each pointer occupies 8 bytes.

    // Indicates a verbosity level. Zero means default (minimal). Larger generally means more information.
    int verbosity;

    // The random seed.
    int seed;

    // The message sink.
    MESSAGESINK messageSink;

    // The data sink.
    DATASINK dataSink;

    // The model sink.
    MODELSINK modelSink;

    // Max slots to return for vector valued columns(<=0 to return all).
    int maxSlots;

    // Check cancellation flag.
    CHECKCANCEL checkCancel;

    // Path to python executable
    const char* pythonPath;

public:
    EnvironmentBlock(int verbosity = 0, int maxSlots = -1, int seed = 42, const char* pythonPath = NULL);
    ~EnvironmentBlock();
    std::string GetErrorMessage() { return _errMessage; }
    bp::dict GetData();

private:
    static MANAGED_CALLBACK(void) DataSink(EnvironmentBlock *penv, const DataViewBlock *pdata, void **&setters, void *&keyValueSetter);
    static MANAGED_CALLBACK(void) MessageSink(EnvironmentBlock *penv, MessageKind kind, const char *sender, const char *message);
    static MANAGED_CALLBACK(void) ModelSink(EnvironmentBlock *penv, const unsigned char *pBinaryModel, size_t iModelLen);
    static MANAGED_CALLBACK(bool) CheckCancel();

private:
    void DataSinkCore(const DataViewBlock * pdata);

private:
    // This has a bit set for each kind of message that is desired.
    int _kindMask;
    // Fields used by the data callbacks. These keep the appropriate memory alive during the data operations.
    int _irowBase;
    int _crowWant;
    std::vector<void*> _vset;
    std::string _errMessage;

    std::vector<std::string> _names;
    std::vector<PythonObjectBase*> _columns;
    // Maps between the column index, and the index in _vKeyValues containing the key names, or -1 if 
    // there are no key names.
    std::vector<CxInt64> _columnToKeyMap;

    std::vector<PythonObjectSingle<std::string>*> _vKeyValues;

    static MANAGED_CALLBACK(void) SetR4(EnvironmentBlock *env, int col, long m, long n, float value)
    {
        PythonObject<float>* colObject = dynamic_cast<PythonObject<float>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetR8(EnvironmentBlock *env, int col, long m, long n, double value)
    {
        PythonObject<double>* colObject = dynamic_cast<PythonObject<double>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetBL(EnvironmentBlock *env, int col, long m, long n, signed char value)
    {
        PythonObject<signed char>* colObject = dynamic_cast<PythonObject<signed char>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
        if (value < 0)
            env->_columns[col]->SetKind(-1);
    }
    static MANAGED_CALLBACK(void) SetI1(EnvironmentBlock *env, int col, long m, long n, signed char value)
    {
        PythonObject<signed char>* colObject = dynamic_cast<PythonObject<signed char>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetI2(EnvironmentBlock *env, int col, long m, long n, short value)
    {
        PythonObject<short>* colObject = dynamic_cast<PythonObject<short>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetI4(EnvironmentBlock *env, int col, long m, long n, int value)
    {
        PythonObject<int>* colObject = dynamic_cast<PythonObject<int>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetI8(EnvironmentBlock *env, int col, long m, long n, CxInt64 value)
    {
        PythonObject<CxInt64>* colObject = dynamic_cast<PythonObject<CxInt64>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetU1(EnvironmentBlock *env, int col, long m, long n, unsigned char value)
    {
        PythonObject<unsigned char>* colObject = dynamic_cast<PythonObject<unsigned char>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetU2(EnvironmentBlock *env, int col, long m, long n, unsigned short value)
    {
        PythonObject<unsigned short>* colObject = dynamic_cast<PythonObject<unsigned short>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetU4(EnvironmentBlock *env, int col, long m, long n, unsigned int value)
    {
        PythonObject<unsigned int>* colObject = dynamic_cast<PythonObject<unsigned int>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetU8(EnvironmentBlock *env, int col, long m, long n, CxUInt64 value)
    {
        PythonObject<CxUInt64>* colObject = dynamic_cast<PythonObject<CxUInt64>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, value);
    }
    static MANAGED_CALLBACK(void) SetTX(EnvironmentBlock *env, int col, long m, long n, char* value, long length)
    {
        PythonObject<string>* colObject = dynamic_cast<PythonObject<string>*>(env->_columns[col]);
        assert(colObject != nullptr);
        colObject->SetAt(m, n, std::string(value, length));
    }
    static MANAGED_CALLBACK(void) SetKeyValue(EnvironmentBlock *env, int keyColumnIndex, int keyCode, char* value, long length)
    {
        assert(keyColumnIndex < env->_vKeyValues.size());
        PythonObject<string>* keyNamesObject = env->_vKeyValues[keyColumnIndex];
        keyNamesObject->SetAt(keyCode, 0, std::string(value, length));
    }
};


// Used to fill managed interop blocks with bad values when done with them. This helps find issues
// where managed code holds onto raw pointers too long.

#define BAD_QUAD 0xDEADF00D

inline void FillDead(int& x)
{
    assert(sizeof(int) == 4);
    x = BAD_QUAD;
}

inline void FillDead(CxInt64& x)
{
    assert(sizeof(CxInt64) == 8);
    assert(sizeof(int) == 4);
    ((int *)&x)[0] = BAD_QUAD;
    ((int *)&x)[1] = BAD_QUAD;
}

template<typename T>
inline void FillDead(T*& x)
{
    assert(sizeof(T*) == 8);
    assert(sizeof(int) == 4);
    ((int *)&x)[0] = BAD_QUAD;
    ((int *)&x)[1] = BAD_QUAD;
}

struct MlNetExecutionError : std::exception
{
    MlNetExecutionError(const char *message) : msg_(message) { }
    virtual char const *what() const noexcept { return msg_.c_str(); }

private:
    std::string msg_;
};
