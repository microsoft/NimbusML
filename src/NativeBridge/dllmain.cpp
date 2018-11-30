// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "stdafx.h"
#include <string>
#include "DataViewInterop.h"
#include "ManagedInterop.h"

#define PARAM_SEED "seed"
#define PARAM_GRAPH "graph"
#define PARAM_VERBOSE "verbose"
#define PARAM_NIMBUSML_PATH "nimbusmlPath"
#define PARAM_DOTNETCLR_PATH "dotnetClrPath"
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
GENERICEXEC EnsureExec(const char *nimbuslibspath, const char *coreclrpath)
{
    if (g_mlnetInterface == nullptr)
        g_mlnetInterface = new MlNetInterface();

    if (g_exec == nullptr)
    {
        FNGETTER getter = g_mlnetInterface->EnsureGetter(nimbuslibspath, coreclrpath);
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
        auto graph = bp::extract_or_cast<std::string>(params[PARAM_GRAPH]);
        auto nimbusmlPath = bp::extract_or_cast<std::string>(params[PARAM_NIMBUSML_PATH]);
        auto dotnetClrPath = bp::extract_or_cast<std::string>(params[PARAM_DOTNETCLR_PATH]);
        auto verbose = bp::extract_or_cast<std::int32_t>(params[PARAM_VERBOSE]);
        std::int32_t i_verbose = std::int32_t(verbose);
        std::string s_nimbusmlPath = std::string(nimbusmlPath);
        std::string s_dotnetClrPath = std::string(dotnetClrPath);
        std::string s_graph = std::string(graph);
        const char *nimbuslibspath = s_nimbusmlPath.c_str();
        const char *coreclrpath = s_dotnetClrPath.c_str();

        GENERICEXEC exec = EnsureExec(nimbuslibspath, coreclrpath);
        if (exec == nullptr)
            throw std::invalid_argument("Failed to communicate with the managed library. Path searched: "
                + s_nimbusmlPath + " and " + s_dotnetClrPath);

        int seed = 42;
    #if BOOST_PYTHON
        if (params.has_key(PARAM_SEED))
            seed = bp::extract<int>(params[PARAM_SEED]);
    #else
        if (params.has_key_or_contains(PARAM_SEED))
    #endif
            seed = bp::extract_or_cast<int>(params[PARAM_SEED]);

        EnvironmentBlock env(i_verbose, 0, seed);
        int retCode;

#if BOOST_PYTHON
        if (params.has_key(PARAM_DATA) && bp::extract_or_cast<bp::dict>(params[PARAM_DATA]).check())
#else
        if (params.has_key_or_contains(PARAM_DATA) && bp::isinstance<bp::dict>(params[PARAM_DATA]))
#endif
        {
            bp::dict d = bp::extract_or_cast<bp::dict>(params[PARAM_DATA]);
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

#ifdef BOOST_PYTHON
BOOST_PYTHON_MODULE(pybridge)
#else
PYBIND11_MODULE(pybridge, m)
#endif
{
    //The managed code assumes that each pointer occupies 8 bytes.
    assert(sizeof(void*) == 8);

    //
    // initialize python
    //
    Py_Initialize();

#ifdef BOOST_PYTHON
    //
    // initialize numpy types
    //

    np::initialize();
    bp::register_exception_translator<MlNetExecutionError>(&translate_mlnet_exception);
    def("px_call", pxCall);
#else
    //static bp::register_exception<MlNetExecutionError>(m, "MlNetExecutionError");
    static bp::exception<BridgeExecutionError> exc(m, "BridgeExecutionError");
    bp::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const MlNetExecutionError &em) {
            ::PyErr_SetString(::PyExc_RuntimeError, em.what());
        }
        catch (const std::exception &e) {
            exc(e.what());
        }
    });

    m.def("px_call", pxCall);
#endif
}