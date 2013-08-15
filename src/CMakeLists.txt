# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

# ----------------------------------------------------------------------
# Base CMake setup
cmake_minimum_required (VERSION 2.6)
#PROJECT(qt_use_cmake)



#-----------------	basic setting 	-------------------



#-----------------   common setting -------------------
SET(ROOT_DIR ../../)
SET(EXECUTABLE_OUTPUT_PATH ${ROOT_DIR})
#SET(LIBRARY_OUTPUT_PATH ${ROOT_DIR}/lib)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
MESSAGE(STATUS "CMAKE_CURRENT_BINARY_DIR " ${CMAKE_CURRENT_BINARY_DIR})



#-------------- add common files 	--------------------
file(GLOB src_main "*.cpp")
file(GLOB h_main "*.h")

MESSAGE(STATUS "CPP: " ${src_main})
MESSAGE(STATUS "H: " ${h_main})


add_subdirectory(object_detector)
#add_subdirectory(ui)


#MESSAGE(STATUS "This is BINARY dir " ${qt_use_cmake_BINARY_DIR})
#MESSAGE(STATUS "This is SOURCE dir "${qt_use_cmake_SOURCE_DIR})
#MESSAGE(STATUS "This is BUILD dir "${PRO_BUILD_DIR})


#-----------------  set qt files ---------------
FIND_PACKAGE(Qt REQUIRED)
INCLUDE(${QT_USE_FILE})
#ADD_DEFINITIONS(${QT_DEFINITIONS})
SET(QT_ROOT /usr/share/qt4)
MESSAGE(STATUS "QT_ROOT" ${QT_ROOT})

#MESSAGE("QT_DEF" “${QT_DEFINITIONS}”)
#set(QT_QMAKE_EXECUTABLE ${QT_ROOT}/bin/qmake)
#set(QT_MOC_EXECUTABLE ${QT_ROOT}/bin/moc)
#set(QT_RCC_EXECUTABLE ${QT_ROOT}/bin/rcc)
#set(QT_UIC_EXECUTABLE ${QT_ROOT}/bin/uic)
#set(QT_INCLUDE_DIR ${QT_ROOT}/include)
#set(QT_LIBRARY_DIR ${QT_ROOT}/lib)
#set(QT_QTCORE_LIBRARY ${QT_ROOT}/lib)

SET(src_forms widget_demo.ui)
QT4_WRAP_UI(headers_ui ${src_forms})
QT4_WRAP_CPP(h_moc ${h_main})

#IF(COMMAND uic)
MESSAGE("have hehe " ${QT_UIC_EXECUTABLE})
#endif(COMMAND uic)



#---------------  set gpu files -----------------
add_subdirectory(cuda_code)


#find_package(CUDA 4.0 REQUIRED)
# collect source files
#file( GLOB  hdr *.hpp *.h )
#MESSAGE(STATUS "CUDA FILES ARE: " ${src_cuda})
#SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

#---------------    set OpenCV libraries    --------------------
set(opencv_LIBRARIES
    opencv_core opencv_imgproc opencv_highgui opencv_ml
    opencv_video opencv_features2d
    opencv_calib3d
    opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
   ) # quick hack for opencv2.4 support





#--------------- build application -----------
SET(appname main_demo)
ADD_EXECUTABLE(${appname} ${src_main} ${h_moc} ${headers_ui})

MESSAGE(STATUS "ui: " ${headers_ui})
TARGET_LINK_LIBRARIES( ${appname}
	${QT_QTMAIN_LIBRARY} 
        ${QT_LIBRARIES}
        ${opencv_LIBRARIES}
        yl_cuda_code
        yl_object_detector
        )
