- CamelCase or snake_case. Function names start with uppercase, variable names start with lowercase. Follow PyTorch conventions where applicable.

## 1. Project Build Rules

#### Subproject Directory Structure

```
├── ProjectName
│   ├── CMakeLists.txt
│   ├── include
│   │   └── ProjectName
│   │       ├── xxx.h
│   │       └── xxx2.h
│   └── src
│       ├── xxx.cpp
│       └── xxx2.cpp
```

#### Subproject CMakeLists.txt

Each subproject is packaged as a static library:

```cmake
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(ProjectName STATIC ${srcs})
target_include_directories(ProjectName PUBLIC include)
```

#### Header Files (.h)

- **Template functions must be implemented in .h files** — implementing them in .cpp will cause linker errors.
- Standalone (non-member) functions implemented in headers must be marked `inline` to avoid multiple-definition errors.

```cpp
#pragma once
namespace ProjectName {
    void FunctionName();
}
```

#### Source Files (.cpp)

- Should not contain additional internal includes. Long functions should be declared in .h and implemented in .cpp.

```cpp
#include <ProjectName/ModuleName.h>
namespace ProjectName {
    void FunctionName() { /* implementation */ }
}
```

#### Root CMakeLists.txt

Sets default build type, C++ standard, and adds all subprojects:

```cmake
cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

project(SEC-PPDL LANGUAGES CXX)

add_subdirectory(Datatype)
add_subdirectory(HE)
add_subdirectory(Operator)
add_subdirectory(Layer)
add_subdirectory(Test)
```

## 2. Intra- and Inter-project References

#### Within a Subproject

Example structure:

```
├── LinearLayer
│   ├── CMakeLists.txt
│   ├── include
│   │   └── LinearLayer
│   │       ├── Conv.h
│   │       └── DWConv.h
│   └── src
│       ├── Conv.cpp
│       └── DWConv.cpp
```

In `Conv.h`, reference `DWConv.h` directly with the project namespace:

```cpp
#include <LinearLayer/DWConv.h>
LinearLayer::DWConv dwconv;
```

#### Across Subprojects

To use `HE` in `LinearLayer`, include and use with namespace:

```cpp
#include <HE/HE.h>
HE::HE he;
```

Also add the dependency in CMakeLists.txt (`HE` must be compiled before `LinearLayer`):

```cmake
target_link_libraries(LinearLayer PUBLIC HE)
```
