# 定义库目标 (这里用 STATIC 静态库，简单方便)
add_library(${PROJECT_NAME} STATIC
    tensor.cpp
    node.cpp
)

# 设置头文件包含路径
# PUBLIC: 库内部和库的使用者都能看到 include/ 目录
target_include_directories(${PROJECT_NAME} PUBLIC 
    "${PROJECT_SOURCE_DIR}/include"
)

# 开启所有警告 (面试加分项：展示你对代码质量的洁癖)
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -mavx2 -mfma)
endif()
