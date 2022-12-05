

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(test_kenlm VERSION 0.1)

set(KENLM_TEST_SOURCE
		test_kenlm.cpp
	)


# This directory has children that need to be processed
include_directories("../")
aux_source_directory(. DIR_SRCS)

add_executable(${PROJECT_NAME} test_kenlm.cpp)


target_link_libraries(${PROJECT_NAME}
  /home/sfy/study/package/kenlm/build/lib/libkenlm_builder.so
  /home/sfy/study/package/kenlm/build/lib/libkenlm_filter.so
  /home/sfy/study/package/kenlm/build/lib/libkenlm_util.so
  /home/sfy/study/package/kenlm/build/lib/libkenlm.so
  # Boost is required for building binaries and tests
  "$<BUILD_INTERFACE:${Boost_LIBRARIES}>"
  )

set(KNELM_MAX_ORDER 6 CACHE STRING "maximum supported ngram order")
target_compile_definitions(${PROJECT_NAME} PUBLIC -DKENLM_MAX_ORDER=${KNELM_MAX_ORDER})


