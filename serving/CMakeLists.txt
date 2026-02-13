add_library(tinyinfer_serving_lib
    src/server.cpp
    src/session.cpp
    src/protocol.cpp
)

target_include_directories(tinyinfer_serving_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tinyinfer_serving_lib
    PUBLIC TinyInfer pthread
)

add_executable(tinyinfer-serving src/main.cpp)
target_link_libraries(tinyinfer-serving PRIVATE tinyinfer_serving_lib)

add_executable(client_demo tests/client_demo.cpp)
