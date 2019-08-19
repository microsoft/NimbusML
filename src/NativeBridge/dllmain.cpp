// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "stdafx.h"
#include <string>
#include "DataViewInterop.h"
#include "ManagedInterop.h"

#define PARAM_SEED "seed"
#define PARAM_MAX_SLOTS "maxSlots"
#define PARAM_GRAPH "graph"
#define PARAM_VERBOSE "verbose"
#define PARAM_MLNET_PATH "mlnetPath"
#define PARAM_DOTNETCLR_PATH "dotnetClrPath"
#define PARAM_DPREP_PATH "dprepPath"
#define PARAM_PYTHON_PATH "pythonPath"
#define PARAM_DATA "data"


enum FnId
{
    FnIdHelloMlNet = 1,
    FnIdGenericExec = 2,
};

// The general function getter.
typedef void*(STDCALL *FNGETTER)(FnId id);

// FnId::FnIdGenericExec
typedef int(STDCALL *GENERICEXEC)(
    void *penv, //const EnvironmentBlock *penv
    const char *pgraph,
    int cdata,
    const DataSourceBlock **ppdata
    );

#if _MSC_VER
#include "WinInterface.h"
typedef WinMlNetInterface MlNetInterface;
#else
#include "UnixInterface.h"
typedef UnixMlNetInterface MlNetInterface;
#endif

// Function pointers for the managed code entry points.
static MlNetInterface *g_mlnetInterface = nullptr;
static GENERICEXEC g_exec = nullptr;

// Ensure that we have the DotNetBridge managed code entry point.
GENERICEXEC EnsureExec(const char *mlnetpath, const char *coreclrpath, const char *dpreppath)
{
    if (g_mlnetInterface == nullptr)
        g_mlnetInterface = new MlNetInterface();

    if (g_exec == nullptr)
    {
        FNGETTER getter = g_mlnetInterface->EnsureGetter(mlnetpath, coreclrpath, dpreppath);
        if (getter != nullptr)
            g_exec = (GENERICEXEC)getter(FnIdGenericExec);
    }
    return g_exec;
}

void translate_mlnet_exception(MlNetExecutionError const& exc)
{
    // Use the Python 'C' API to set up an exception object
    ::PyErr_SetString(::PyExc_RuntimeError, exc.what());
}

bp::dict pxCall(bp::dict& params)
{
    bp::dict res = bp::dict();
    try
    {
        bp::extract<std::string> graph(params[PARAM_GRAPH]);
        bp::extract<std::string> mlnetPath(params[PARAM_MLNET_PATH]);
        bp::extract<std::string> dotnetClrPath(params[PARAM_DOTNETCLR_PATH]);
        bp::extract<std::string> dprepPath(params[PARAM_DPREP_PATH]);
        bp::extract<std::string> pythonPath(params[PARAM_PYTHON_PATH]);
        bp::extract<std::int32_t> verbose(params[PARAM_VERBOSE]);
        std::string s_mlnetPath = std::string(mlnetPath);
        std::string s_dotnetClrPath = std::string(dotnetClrPath);
        std::string s_dprepPath = std::string(dprepPath);
        std::string s_pythonPath = std::string(pythonPath);
        std::string s_graph = std::string(graph);
        const char *mlnetpath = s_mlnetPath.c_str();
        const char *coreclrpath = s_dotnetClrPath.c_str();
        const char *dpreppath = s_dprepPath.c_str();

        GENERICEXEC exec = EnsureExec(mlnetpath, coreclrpath, dpreppath);
        if (exec == nullptr)
            throw std::invalid_argument("Failed to communicate with the managed library. Paths searched: "
                + s_mlnetPath + " and " + s_dotnetClrPath);

        int seed = 42;
        if (params.has_key(PARAM_SEED))
            seed = bp::extract<int>(params[PARAM_SEED]);

        int max_slots = -1;
        if (params.has_key(PARAM_MAX_SLOTS))
            max_slots = bp::extract<int>(params[PARAM_MAX_SLOTS]);

        EnvironmentBlock env(std::int32_t(verbose), seed, max_slots, s_pythonPath.c_str());
        int retCode;
        if (params.has_key(PARAM_DATA) && bp::extract<bp::dict>(params[PARAM_DATA]).check())
        {
            bp::dict d = bp::extract<bp::dict>(params[PARAM_DATA]);
            DataSourceBlock data(d);
            const DataSourceBlock *datas[1];
            datas[0] = &data;
            retCode = exec(&env, s_graph.c_str(), 1, datas);
        }
        else
            retCode = exec(&env, s_graph.c_str(), 0, NULL);

        res = env.GetData();

        if (retCode == -1)
            // REVIEW: get the content of IChannel and add it the the error message.
            throw std::runtime_error("Returned code is -1. Check the log for error messages.");
    }
    catch (const std::exception& e)
    {
        throw MlNetExecutionError(e.what());
    }
    catch (bp::error_already_set const&)
    {
        PyErr_Print();
    }

    return res;
}

BOOST_PYTHON_MODULE(pybridge)
{
    //The managed code assumes that each pointer occupies 8 bytes.
    assert(sizeof(void*) == 8);

    //
    // initialize python
    //
    Py_Initialize();

    //
    // initialize numpy types
    //
    np::initialize();

    bp::register_exception_translator<MlNetExecutionError>(&translate_mlnet_exception);
    def("px_call", pxCall);
}