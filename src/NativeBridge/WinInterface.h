// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>

#include <cstdlib>
#include ".\inc\mscoree.h"

std::wstring Utf8ToUtf16le(const char* utf8Str)
{
    // Allocate the utf16 string buffer.
    unsigned int iSizeOfStr = MultiByteToWideChar(CP_UTF8, 0, utf8Str, -1, NULL, 0);
    wchar_t* utf16Str = new wchar_t[iSizeOfStr];
    std::wstring result;

    try
    {
        // Convert the utf8 string.
        MultiByteToWideChar(CP_UTF8, 0, utf8Str, -1, utf16Str, iSizeOfStr);

        // Copy it to a std::wstring to simplify heap management.
        result = utf16Str;

        // Free the utf16 string buffer.
        delete[] utf16Str;
        utf16Str = NULL;
    }
    catch (...)
    {
        // On exception clean up and re-throw.
        if (utf16Str) delete[] utf16Str;
        throw;
    }

    return result;
}

void ConvertToWinPath(std::wstring & dir)
{
    for (int ich = 0; ich < dir.size(); ich++)
    {
        if (dir[ich] == '/')
            dir[ich] = '\\';
    }
    if (dir[dir.size() - 1] != '\\')
        dir.append(W("\\"));
}

class WinMlNetInterface
{
private:
    FNGETTER _getter;

public:
    WinMlNetInterface() : _getter(nullptr), _host(nullptr), _hmodCore(nullptr)
    {
    }

private:
    // The coreclr.dll module.
    HMODULE _hmodCore;
    // The runtime host and app domain.
    ICLRRuntimeHost2 *_host;
    DWORD _domainId;

private:
    void Shutdown()
    {
        if (_host)
        {
            // Unload the app domain, waiting until done.
            HRESULT hr = _host->UnloadAppDomain(_domainId, true);
            if (FAILED(hr))
            {
                // REVIEW: Handle failure.
                //log << W("Failed to unload the AppDomain. ERRORCODE: ") << hr << Logger::endl;
                //return false;
            }

            hr = _host->Stop();
            if (FAILED(hr)) {
                // REVIEW: Handle failure.
                //log << W("Failed to stop the host. ERRORCODE: ") << hr << Logger::endl;
                //return false;
            }

            // Release the reference to the host.
            _host->Release();
            _host = nullptr;
        }

        if (_hmodCore)
        {
            // Free the module. This is done for completeness, but in fact CoreCLR.dll
            // was pinned earlier so this call won't actually free it. The pinning is
            // done because CoreCLR does not support unloading.
            ::FreeLibrary(_hmodCore);

            _hmodCore = nullptr;
        }
    }

    HMODULE EnsureCoreClrModule(const wchar_t *path)
    {
        if (_hmodCore == nullptr)
        {
            std::wstring pathCore(path);
            //std::wstring env(path);
            //env.append(L";").append(_wgetenv(L"PATH"));
            //_wputenv(env.c_str());

            pathCore.append(W("CoreCLR.dll"));

            // Load CoreCLR from the indicated directory.
            SetDllDirectoryW(path);
            HMODULE hmodCore = ::LoadLibraryExW(pathCore.c_str(), NULL,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
                // | LOAD_LIBRARY_SEARCH_SYSTEM32
                | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                );
            SetDllDirectoryW(nullptr);
            if (!hmodCore)
            {
                // REVIEW: Log failure somehow.
                return nullptr;
            }

            // Pin the module - CoreCLR.dll does not support being unloaded.
            HMODULE hmodTmp;
            if (!::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, pathCore.c_str(), &hmodTmp))
            {
                // REVIEW: Log failure somehow.
                // *m_log << W("Failed to pin: ") << pathCore << Logger::endl;
                return nullptr;
            }

            _hmodCore = hmodCore;
        }
        return _hmodCore;
    }

    // Appends all .dlls in the given directory to list, semi-colon terminated.
    void AddDllsToList(const wchar_t *path, std::wstring & list)
    {
        std::wstring wildPath(path);
        wildPath.append(W("*.dll"));

        WIN32_FIND_DATA data;
        HANDLE findHandle = FindFirstFile(wildPath.c_str(), &data);
        if (findHandle == INVALID_HANDLE_VALUE)
        {
            // REVIEW: Report failure somehow.
            return;
        }

        do
        {
            if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
                list.append(path).append(data.cFileName).append(W(";"));
        } while (0 != FindNextFile(findHandle, &data));
        FindClose(findHandle);
    }

    ICLRRuntimeHost2* EnsureClrHost(const wchar_t * libsRoot, const wchar_t * coreclrDirRoot)
    {
        if (_host != nullptr)
            return _host;

        // Set up paths.
        std::wstring dirmlnet(libsRoot);
        dirmlnet.append(L"Platform\\win-x64\\publish\\");

        std::wstring tpaList;
        AddDllsToList(libsRoot, tpaList);
        AddDllsToList(dirmlnet.c_str(), tpaList);

        //std::wstring dirClr1(L"E:\\sources\\NimbusML\\dependencies\\Python3.6\\Lib\\site-packages\\dotnetcore2\\bin\\shared\\Microsoft.NETCore.App\\2.1.0\\");
        AddDllsToList(coreclrDirRoot, tpaList);

        // Start the CoreCLR.
        HMODULE hmodCore = EnsureCoreClrModule(coreclrDirRoot);

        FnGetCLRRuntimeHost pfnGetCLRRuntimeHost =
            (FnGetCLRRuntimeHost)::GetProcAddress(hmodCore, "GetCLRRuntimeHost");
        if (!pfnGetCLRRuntimeHost)
            return nullptr;

        ICLRRuntimeHost2 *host;
        HRESULT hr = pfnGetCLRRuntimeHost(IID_ICLRRuntimeHost2, (IUnknown**)&host);
        if (FAILED(hr))
            return nullptr;

        // Default startup flags.
        hr = host->SetStartupFlags((STARTUP_FLAGS)
            (STARTUP_FLAGS::STARTUP_LOADER_OPTIMIZATION_SINGLE_DOMAIN |
                STARTUP_FLAGS::STARTUP_SINGLE_APPDOMAIN));
        if (FAILED(hr))
        {
            // log << W("Failed to set startup flags. ERRORCODE: ") << hr << Logger::endl;
            host->Release();
            return nullptr;
        }

        hr = host->Start();
        if (FAILED(hr))
        {
            // log << W("Failed to start CoreCLR. ERRORCODE: ") << hr << Logger::endl;
            host->Release();
            return nullptr;
        }

        DWORD appDomainFlags = APPDOMAIN_ENABLE_PLATFORM_SPECIFIC_APPS |
            APPDOMAIN_ENABLE_PINVOKE_AND_CLASSIC_COMINTEROP |
            APPDOMAIN_DISABLE_TRANSPARENCY_ENFORCEMENT;

        // Create an AppDomain.

        // Allowed property names:
        // APPBASE
        // - The base path of the application from which the exe and other assemblies will be loaded
        //
        // TRUSTED_PLATFORM_ASSEMBLIES
        // - The list of complete paths to each of the fully trusted assemblies
        //
        // APP_PATHS
        // - The list of paths which will be probed by the assembly loader
        //
        // APP_NI_PATHS
        // - The list of additional paths that the assembly loader will probe for ngen images
        //
        const wchar_t *property_keys[] = {
            W("TRUSTED_PLATFORM_ASSEMBLIES"),
            W("APP_PATHS"),
            W("APP_NI_PATHS"),
            W("AppDomainCompatSwitch"),
        };
        const wchar_t *property_values[] = {
            // TRUSTED_PLATFORM_ASSEMBLIES
            tpaList.c_str(),
            // APP_PATHS
            libsRoot,
            // APP_NI_PATHS
            dirmlnet.c_str(),
            // AppDomainCompatSwitch
            W("UseLatestBehaviorWhenTFMNotSpecified")
        };

        hr = host->CreateAppDomainWithManager(
            W("NativeBridge"),  // The friendly name of the AppDomain
            // Flags:
            // APPDOMAIN_ENABLE_PLATFORM_SPECIFIC_APPS
            // - By default CoreCLR only allows platform neutral assembly to be run. To allow
            //   assemblies marked as platform specific, include this flag
            //
            // APPDOMAIN_ENABLE_PINVOKE_AND_CLASSIC_COMINTEROP
            // - Allows sandboxed applications to make P/Invoke calls and use COM interop
            //
            // APPDOMAIN_SECURITY_SANDBOXED
            // - Enables sandboxing. If not set, the app is considered full trust
            //
            // APPDOMAIN_IGNORE_UNHANDLED_EXCEPTION
            // - Prevents the application from being torn down if a managed exception is unhandled
            appDomainFlags,
            NULL, // Name of the assembly that contains the AppDomainManager implementation.
            NULL, // The AppDomainManager implementation type name.
            sizeof(property_keys) / sizeof(wchar_t*), // The number of properties.
            property_keys,
            property_values,
            &_domainId);
        if (FAILED(hr))
        {
            // log << W("Failed call to CreateAppDomainWithManager. ERRORCODE: ") << hr << Logger::endl;
            host->Release();
            return nullptr;
        }

        _host = host;
        return _host;
    }

public:
    FNGETTER EnsureGetter(const char *nimbuslibspath, const char *coreclrpath)
    {
        if (_getter != nullptr)
            return _getter;

        std::wstring libsdir = Utf8ToUtf16le(nimbuslibspath);
        ConvertToWinPath(libsdir);

        std::wstring coreclrdir;
        if (strlen(coreclrpath) != 0)
        {
            coreclrdir = Utf8ToUtf16le(coreclrpath);
            ConvertToWinPath(coreclrdir);
        }
        else
        {
            coreclrdir = libsdir;
        }

        ICLRRuntimeHost2* host = EnsureClrHost(libsdir.c_str(), coreclrdir.c_str());
        if (host == nullptr)
            return nullptr;

        // CoreCLR currently requires using environment variables to set most CLR flags.
        // cf. https://github.com/dotnet/coreclr/blob/master/Documentation/project-docs/clr-configuration-knobs.md
        if(_wputenv(W("COMPlus_gcAllowVeryLargeObjects=1")) == -1)
            return nullptr;

        INT_PTR getter;
        HRESULT hr = host->CreateDelegate(
            _domainId,
            W("DotNetBridge"),
            W("Microsoft.MachineLearning.DotNetBridge.Bridge"),
            W("GetFn"),
            &getter);
        if (FAILED(hr))
            return nullptr;

        _getter = (FNGETTER)getter;
        return _getter;
    }
};