# This CMake file was created by Lane Schwartz <dowobeha@gmail.com>

# Explicitly list the source files for this subdirectory
#
# If you add any source files to this subdirectory
#    that should be included in the kenlm library,
#        (this excludes any unit test files)
#    you should add them to the following list:
#
# In order to set correct paths to these files
#    in case this variable is referenced by CMake files in the parent directory,
#    we prefix all files with ${CMAKE_CURRENT_SOURCE_DIR}.
#
set(KENLM_FILTER_SOURCE
		${CMAKE_CURRENT_SOURCE_DIR}/arpa_io.cc
		${CMAKE_CURRENT_SOURCE_DIR}/phrase.cc
		${CMAKE_CURRENT_SOURCE_DIR}/vocab.cc
	)

# Group these objects together for later use.
#
# Given add_library(foo OBJECT ${my_foo_sources}),
# refer to these objects as $<TARGET_OBJECTS:foo>
#
add_library(kenlm_filter SHARED ${KENLM_FILTER_SOURCE})
target_link_libraries(kenlm_filter PUBLIC kenlm_util)
# Since headers are relative to `include/kenlm` at install time, not just `include`
target_include_directories(kenlm_filter PUBLIC $<INSTALL_INTERFACE:include/kenlm>)

AddExes(EXES filter phrase_table_vocab
        LIBRARIES kenlm_filter kenlm)

install(
  TARGETS kenlm_filter
  EXPORT kenlmTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
)
