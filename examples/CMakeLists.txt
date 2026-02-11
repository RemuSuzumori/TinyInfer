# This creates the FIRST executable: demo_inference.exe
add_executable(demo_inference demo_inference.cpp)
target_link_libraries(demo_inference PRIVATE TinyInfer)

# This creates the SECOND executable: demo_mnist.exe
add_executable(demo_mnist demo_mnist.cpp)
target_link_libraries(demo_mnist PRIVATE TinyInfer)

add_executable(mnist_check mnist_check.cpp)
target_link_libraries(mnist_check PRIVATE TinyInfer)