cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(PyBridge)

# enable C++14 features
set(CMAKE_CXX_STANDARD 14)
set(ARCHITECTURE x64)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
endif()

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -Wall")

add_definitions(
-D_POSIX
-D_FILE_OFFSET_BITS=64 -D_USE_ZLIB_
-D_USE_TERADATA_ -DUSE_IN_DB_TERADATA
-DENABLE_DS_VIA_SQLSAT
)

set(CXX_STANDARD_FLAGS -std=c++14 -fpermissive -m64 -funit-at-a-time)
set(CXX_PARAM_FLAGS --param inline-unit-growth=450
                    --param large-function-growth=1200
                    --param max-inline-insns-single=4500)

# turn semicolon-separated string into a space-separated one and
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    string(REPLACE ";" " " CMAKE_CXX_FLAGS_STR "${CXX_STANDARD_FLAGS} ${CXX_PARAM_FLAGS}")
else()
    string(REPLACE ";" " " CMAKE_CXX_FLAGS_STR "${CXX_STANDARD_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_STR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fPIC")

get_filename_component(DEPENDENCIES ../../dependencies ABSOLUTE)
MESSAGE( STATUS "DEPENDENCIES: " ${DEPENDENCIES} )

set(CMAKE_INSTALL_DIR "${CMAKE_BINARY_DIR}/../")
MESSAGE( STATUS "CMAKE_INSTALL_DIR: " ${CMAKE_INSTALL_DIR})
MESSAGE( STATUS "CMAKE_BINARY_DIR: " ${CMAKE_BINARY_DIR} )
MESSAGE( STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR} )
MESSAGE( STATUS "PYTHON_DIR: " ${PYTHON_DIR} )
MESSAGE( STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME} )

include_directories(${PYTHON_DIR}/include ${PYTHON_DIR}/include/python3.6m ${PYTHON_DIR}/include/python3.7m)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=unused-variable -Wno-error=attributes -Wno-error=unused-value")
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'/Library/Frameworks/Python.framework/Versions/${PYTHON_VER}/lib'" )
else ()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'\${ORIGIN}'" )
endif()

set (PYBRIDGE_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/UnixInterface.h
    ${CMAKE_SOURCE_DIR}/stdafx.h
    ${CMAKE_SOURCE_DIR}/PythonInterop.h
    ${CMAKE_SOURCE_DIR}/DataViewInterop.h
    ${CMAKE_SOURCE_DIR}/ManagedInterop.h
    ${CMAKE_SOURCE_DIR}/dllmain.cpp
    ${CMAKE_SOURCE_DIR}/PythonInterop.cpp
    ${CMAKE_SOURCE_DIR}/DataViewInterop.cpp
    ${CMAKE_SOURCE_DIR}/ManagedInterop.cpp)

add_library(PyBridge_objects OBJECT ${PYBRIDGE_SOURCE_FILES})
add_library(PyBridge SHARED $<TARGET_OBJECTS:PyBridge_objects>)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	if (PYTHON_VER STREQUAL "3.8")
		set(LIB_PYTHON ${PYTHON_DIR}/libpython${PYTHON_VER}.so)
	else ()
		set(LIB_PYTHON ${PYTHON_DIR}/lib/libpython${PYTHON_VER}m.so)
	endif()
endif()

set(PYBRIDGE_LIB_DEPS ${LIB_PYTHON})

set_target_properties(PyBridge PROPERTIES PREFIX "")
set_target_properties(PyBridge PROPERTIES OUTPUT_NAME "pybridge")
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # Set so that loader won't have trouble with Python libpython vs normal libpython
    set_target_properties(PyBridge PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    # Explicitly use .so extension for both Linux and Mac because python will not open .dylib libraries (the default on Mac)
    set_target_properties(PyBridge PROPERTIES SUFFIX ".so")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(PyBridge ${PYBRIDGE_LIB_DEPS})
else()
    target_link_libraries(PyBridge ${PYBRIDGE_LIB_DEPS} -static-libstdc++ -lgomp)
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_DIR} LIBRARY NAMELINK_SKIP)