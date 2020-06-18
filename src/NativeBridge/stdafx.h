// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#if defined( _MSC_VER )
#include "targetver.h"

//
// Exclude rarely-used stuff from Windows headers
//
#define WIN32_LEAN_AND_MEAN             
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret __stdcall
#define STDCALL __stdcall
#define W(str) L ## str
#define MANAGED_CALLBACK(ret) ret __stdcall
#define MANAGED_CALLBACK_PTR(ret, name) ret (__stdcall *name)
#define CLASS_ALIGN __declspec(align(8))

typedef __int64				CxInt64;
typedef unsigned __int64	CxUInt64;
typedef unsigned char		BYTE;

//
// Windows Header Files:
//
#include <windows.h>

#else // Linux/Mac
// For Unix, ignore __stdcall.
#define STDCALL
#define W(str) str
#define FAILED(Status) ((Status) < 0)
#define MANAGED_CALLBACK(ret) ret
#define MANAGED_CALLBACK_PTR(ret, name) ret ( *name)
#define CLASS_ALIGN __attribute__((aligned(8)))

typedef long long	CxInt64;
typedef unsigned long long	CxUInt64;
typedef unsigned char	BYTE;

#endif //_MSC_VER

#include <pybind11/pybind11.h>
#include <string>
#include <exception>
#include <pybind11/numpy.h>
#include <sysmodule.h>

//
// frequently used namespace aliases
//
namespace pb = pybind11;
