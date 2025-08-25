cmake_minimum_required(VERSION 3.10)

project(mobileclip_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

find_package(OpenCV REQUIRED)

set(clip_tokenizer tokenizer/clip_tokenizer.cpp)

# mobileclip_model
add_executable(mobileclip
    main.cc
	mobileclip_demo.cc
    ${clip_tokenizer}
)

target_link_libraries(mobileclip
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
    dl
)

target_include_directories(mobileclip PRIVATE
    ${CMAKE_SOURCE_DIR}/tokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(mobileclip Threads::Threads)
endif()

install(TARGETS mobileclip DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/text.txt DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/dog_224x224.jpg DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/000000039769.jpg DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/vocab.txt DESTINATION ./model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)