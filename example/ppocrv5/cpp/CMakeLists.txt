cmake_minimum_required(VERSION 3.10)

project(rknn_ppocrv5_demo)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
# file(GLOB OpenCV_FILES "${OpenCV_DIR}/../../libopencv*")
# install(PROGRAMS ${OpenCV_FILES} DESTINATION lib)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    clipper.cc
    rknpu2/ppocrv5.cc
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    ${OpenCV_LIBS}
    # ${LIBRGA}
    ${LIBRKNNRT}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()


target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
    # ${RGA_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/general_ocr_002.png DESTINATION model)
set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/PP-OCRv5_mobile_det.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()

set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/PP-OCRv5_mobile_rec.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()