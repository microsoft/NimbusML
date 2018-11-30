// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstdlib>
#include <dirent.h>
#include <dlfcn.h>
#include <iomanip>
#include <limits.h>
#include <set>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define NATIVE_BRIDGE_LIB "/pybridge.so"
#ifdef __APPLE__
#define CORECLR_LIB "/libcoreclr.dylib"
#else
#define CORECLR_LIB "/libcoreclr.so"
#endif

#define CORECLR_INIT "coreclr_initialize"
#define CORECLR_DELEGATE "coreclr_create_delegate"
#define CORECLR_SHUTDOWN "coreclr_shutdown"

#define DOTNETBRIDGE "DotNetBridge"
#define DOTNETBRIDGE_FQDN "Microsoft.MachineLearning.DotNetBridge.Bridge"

#define GET_FN "GetFn"

// Define some common Windows types for Linux.
typedef void* HMODULE;
typedef unsigned int DWORD;
typedef int HRESULT;
typedef void* INT_PTR;

// Prototype of the coreclr_initialize function from the libcoreclr.so.
typedef int(*FnInitializeCoreCLR)(
    const char* exePath,
    const char* appDomainFriendlyName,
    int propertyCount,
    const char** propertyKeys,
    const char** propertyValues,
    HMODULE* hostHandle,
    DWORD* domainId);

// Prototype of the coreclr_shutdown function from the libcoreclr.so.
typedef int(*FnShutdownCoreCLR)(
    HMODULE hostHandle,
    DWORD domainId);

// Prototype of the coreclr_create_delegate function from the libcoreclr.so.
typedef int(*FnCreateDelegate)(
    HMODULE hostHandle,
    DWORD domainId,
    const char* entryPointAssemblyName,
    const char* entryPointTypeName,
    const char* entryPointMethodName,
    void** delegate);

// A handmade ICLRRuntimeHost2 for Linux to reduce the code disparity between Linux and Windows.
class ICLRRuntimeHost2
{
private:
    void* hostHandle;
    std::string pyBridgePath;
    FnInitializeCoreCLR initializeCoreCLR;
    FnCreateDelegate createDelegate;
    FnShutdownCoreCLR shutdownCoreCLR;

public:
    ICLRRuntimeHost2(HMODULE coreclrLib, std::string dirRoot) : hostHandle(nullptr), pyBridgePath(dirRoot)
    {
        pyBridgePath.append(NATIVE_BRIDGE_LIB);
        if (coreclrLib != nullptr)
        {
            initializeCoreCLR = (FnInitializeCoreCLR)dlsym(coreclrLib, CORECLR_INIT);
            createDelegate = (FnCreateDelegate)dlsym(coreclrLib, CORECLR_DELEGATE);
            shutdownCoreCLR = (FnShutdownCoreCLR)dlsym(coreclrLib, CORECLR_SHUTDOWN);
        }
    }

    ~ICLRRuntimeHost2()
    {
        pyBridgePath.clear();
        hostHandle = nullptr;
        initializeCoreCLR = nullptr;
        createDelegate = nullptr;
        shutdownCoreCLR = nullptr;
    }

public:
    HRESULT UnloadAppDomain(DWORD domainId, bool waitUntilDone)
    {
        if (shutdownCoreCLR == nullptr || hostHandle == nullptr)
            return -1;
        return shutdownCoreCLR(hostHandle, domainId);
    }

    HRESULT CreateDelegate(DWORD domainId, const char* assemblyName, const char* className, const char* methodName, INT_PTR* fnPtr)
    {
        if (createDelegate == nullptr || hostHandle == nullptr)
            return -1;
        return createDelegate(hostHandle, domainId, assemblyName, className, methodName, fnPtr);
    }

    HRESULT CreateAppDomainWithManager(
        const char* friendlyName,
        DWORD dwFlags,
        const char* wszAppDomainManagerAssemblyName,
        const char* wszAppDomainManagerTypeName,
        int numProperties,
        const char** propertyKeys,
        const char** propertyValues,
        DWORD* domainId)
    {
        if (initializeCoreCLR == nullptr)
            return -1;
        return initializeCoreCLR(
            pyBridgePath.c_str(),
            friendlyName,
            numProperties,
            propertyKeys,
            propertyValues,
            &hostHandle,
            domainId);
    }
};

class UnixMlNetInterface
{
private:
    FNGETTER _getter;

private:
    // The coreclr.dll module.
    HMODULE _hmodCore;
    // The runtime host and app domain.
    ICLRRuntimeHost2 *_host;
    DWORD _domainId;

public:
    UnixMlNetInterface() : _getter(nullptr), _hmodCore(nullptr), _host(nullptr)
    {
    }

    FNGETTER EnsureGetter(const char *nimbuslibspath, const char *coreclrpath)
    {
        if (_getter != nullptr)
            return _getter;

        std::string libsroot(nimbuslibspath);
        std::string coreclrdir(coreclrpath);

        ICLRRuntimeHost2* host = EnsureClrHost(libsroot.c_str(), coreclrdir.c_str());
        if (host == nullptr)
            return nullptr;

        // CoreCLR currently requires using environment variables to set most CLR flags.
        // cf. https://github.com/dotnet/coreclr/blob/master/Documentation/project-docs/clr-configuration-knobs.md
        // putenv("COMPlus_gcAllowVeryLargeObjects=1");

        INT_PTR getter;
        HRESULT hr = host->CreateDelegate(
            _domainId,
            W(DOTNETBRIDGE),
            W(DOTNETBRIDGE_FQDN),
            W(GET_FN),
            &getter);
        if (FAILED(hr))
            return nullptr;

        _getter = (FNGETTER)getter;
        return _getter;
    }

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
                //return false;
            }
            delete _host;
            _host = nullptr;
        }

        if (_hmodCore)
        {
            dlclose(_hmodCore);
            _hmodCore = nullptr;
        }
    }

    HMODULE EnsureCoreClrModule(const char *path)
    {
        if (_hmodCore == nullptr)
        {
            std::string pathCore(path);
            pathCore.append(CORECLR_LIB);

            if (pathCore.length() >= PATH_MAX)
            {
                std::stringstream message;
                message << "dll location at a path longer than maximum allowed: " << pathCore.c_str() << "; max allowed: " << PATH_MAX;
                throw std::runtime_error(message.str().c_str());
            }

            _hmodCore = dlopen(pathCore.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (_hmodCore == nullptr) {
                std::stringstream message;
                message << "Unable to open dll: " << dlerror();
                throw std::runtime_error(message.str().c_str());
            }
        }
        return _hmodCore;
    }

    // Appends all .dlls in the given directory to list, semi-colon terminated.
    void AddDllsToList(const char *path, std::string & list)
    {
        DIR *dir = opendir(path);
        if (dir == nullptr)
            return;
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            std::string filename(entry->d_name);

            // Check if the extension matches the one we are looking for.
            int extPos = filename.length() - 4;
            if ((extPos <= 0) || (filename.compare(extPos, 4, ".dll") != 0))
                continue;

            list.append(path);
            list.append("/");
            list.append(filename);
            list.append(":");
        }

        closedir(dir);
    }

    ICLRRuntimeHost2* EnsureClrHost(const char * libsRoot, const char * coreclrDirRoot)
    {
        if (_host != nullptr)
            return _host;

        std::string tpaList;
        AddDllsToList(libsRoot, tpaList);
        AddDllsToList(coreclrDirRoot, tpaList);

        // Start the CoreCLR.
        HMODULE hmodCore = EnsureCoreClrModule(coreclrDirRoot);

        ICLRRuntimeHost2 *host = new ICLRRuntimeHost2(hmodCore, libsRoot);
        HRESULT hr;
        // App domain flags are not used by UnixCoreConsole.
        DWORD appDomainFlags = 0;

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
        const char *property_keys[] = {
            W("TRUSTED_PLATFORM_ASSEMBLIES"),
            W("APP_PATHS"),
            W("AppDomainCompatSwitch"),
        };
        const char *property_values[] = {
            // TRUSTED_PLATFORM_ASSEMBLIES
            tpaList.c_str(),
            // APP_PATHS
            libsRoot,
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
            sizeof(property_keys) / sizeof(char*), // The number of properties.
            property_keys,
            property_values,
            &_domainId);

        if (FAILED(hr))
            return nullptr;

        _host = host;
        return _host;
    }
};