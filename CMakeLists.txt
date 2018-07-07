CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(CMAKE_CXX_COMPILER "clang++")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wc++14-extensions -Wno-deprecated -Ofast -march=native -fno-signed-zeros -fno-trapping-math")


FIND_PACKAGE(OpenCV)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(/usr/local/lib)

ADD_EXECUTABLE(
  TestNanoNet
  source/main.cpp
)
TARGET_LINK_LIBRARIES(
  TestNanoNet
  ${OpenCV_LIBS}
)
