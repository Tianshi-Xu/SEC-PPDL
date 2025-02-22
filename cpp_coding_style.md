- 驼峰或下划线，函数名首字母大写，变量首字母小写，torch有的按torch。

## 一、项目构建规则
#### 单个子项目目录结构
- 目录结构
```
├── 项目名字
│   ├── CMakeLists.txt
│   ├── include
│   │   └── 项目名字
│   │       ├── xxx.h
│   │       └── xxx2.h
│   └── src
│       ├── xxx.cpp
│       └── xxx2.cpp
```
- 子项目内部的CMakeLists.txt写法，这样每个子项目都打包在了`项目名字`的library中。
```
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(项目名字 STATIC ${srcs})
target_include_directories(项目名字 PUBLIC include)
```
- 头文件.h写法，**注意，凡是定义了模板的函数，都应该在.h文件中实现，在cpp文件中实现使用时编译会报错**。并且，头文件中实现的独立函数(即非类成员函数)必须包含`inline`关键字，否则编译也会报错。
```
#pragma once
namespace 项目名 {
    void 函数名();
}
```
- 源文件.cpp写法，注意，.cpp文件内部不应包含任何其他include。内容较长的函数应该在.h文件中声明，在.cpp文件中实现。
```
#include <项目名/模块名.h>
namespace 项目名 {
    void 函数名() { 函数实现 }
}
```

#### 根目录CMakeLists.txt写法
- 设置了默认的构建模式，设置了统一的 C++ 版本等各种选项。然后通过 project 命令初始化了根项目。最后把各个子项目`add_subdirectory`即可。
```
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
## 二、项目内及项目间引用
#### 项目内引用
- 例如子项目目录结构如下
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
- 在`Conv.h`中引用`DWConv.h`，可以直接include，使用时加上namespace。
```
#include <LinearLayer/DWConv.h>
LinearLayer::DWConv dwconv;
```
#### 项目间引用
- 例如要在LinearLayer中引用HE中的函数，在代码中依旧直接include，使用时加上namespace。
```
#include <HE/HE.h>
HE::HE he;
```
- 此外，在CMakeLists.txt中添加
```
target_link_libraries(LinearLayer PUBLIC HE)
```
- 注意`HE`需要先于`LinearLayer`编译出来。
