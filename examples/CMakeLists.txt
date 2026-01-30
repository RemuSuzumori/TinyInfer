add_executable(demo_inference
    demo_inference.cpp
)

target_link_libraries(demo_inference PRIVATE
    TinyInfer
)