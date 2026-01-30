# 定义测试可执行文件
add_executable(unit_tests
    test_tensor.cpp
    test_thread_pool.cpp
    test_model.cpp
)

# 链接你的库 + GTest
target_link_libraries(unit_tests PRIVATE 
    TinyInfer           # 你的项目库
    GTest::gtest_main   # GoogleTest (含 main 函数)
)

# 注册到 CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)