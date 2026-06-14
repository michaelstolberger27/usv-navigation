cmake_minimum_required(VERSION 3.8)
project(colav_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)

add_library(colav_core src/control.cpp)
target_include_directories(colav_core PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

install(DIRECTORY include/ DESTINATION include)
install(TARGETS colav_core EXPORT export_colav_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
ament_export_targets(export_colav_core HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_control test/test_control.cpp)
  target_link_libraries(test_control colav_core)
  target_compile_definitions(test_control PRIVATE
    REFERENCE_CSV="${CMAKE_CURRENT_SOURCE_DIR}/test/reference_control.csv")
endif()

ament_package()
