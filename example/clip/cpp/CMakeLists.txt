cmake_minimum_required(VERSION 3.10)

project(rknn_clip_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

set(CMAKE_CXX_FLAGS "-std=c++17")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/timer)
include_directories(${CMAKE_SOURCE_DIR}/tokenizer)
include_directories(${CMAKE_SOURCE_DIR}/unilib)
include_directories(${CMAKE_SOURCE_DIR}/rknpu2/rknn_clip_utils)

set(clip_file rknpu2/clip.cc)
set(rknn_clip_utils rknpu2/rknn_clip_utils/rknn_clip_utils.cc)
set(clip_tokenizer tokenizer/clip_tokenizer.cpp)

set(cn_clip_file rknpu2/cn_clip.cc)
set(cn_clip_tokenizer tokenizer/cn_clip_tokenizer.cpp)
set(unilib_files 
    unilib/unicode.cpp
    unilib/uninorms.cpp
    unilib/unistrip.cpp
)

# clip_demo
add_executable(clip_demo
    clip_demo.cc
	postprocess.cc
    ${clip_file}
    ${rknn_clip_utils}
	${clip_tokenizer}
)

target_link_libraries(clip_demo
	imageutils
    fileutils
    ${LIBRKNNRT}
    dl
)

target_include_directories(clip_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

# cn_clip_demo
add_executable(cn_clip_demo
    cn_clip_demo.cc
	postprocess.cc
    ${cn_clip_file}
    ${rknn_clip_utils}
	${cn_clip_tokenizer}
    ${unilib_files}
)

target_link_libraries(cn_clip_demo
	imageutils
    fileutils
    ${LIBRKNNRT}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(clip_demo Threads::Threads)
    target_link_libraries(cn_clip_demo Threads::Threads)
endif()

target_include_directories(cn_clip_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS clip_demo DESTINATION .)
install(TARGETS cn_clip_demo DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/text.txt DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/cn_text.txt DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/dog_224x224.jpg DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/000000039769.jpg DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/pokemon.jpg  DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/vocab.txt DESTINATION ./model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)