# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yerrick/shared_Development/TrafficSign/program/main_demo_cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yerrick/shared_Development/TrafficSign/program/main_demo_cmake

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/yerrick/shared_Development/TrafficSign/program/main_demo_cmake/CMakeFiles /home/yerrick/shared_Development/TrafficSign/program/main_demo_cmake/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/yerrick/shared_Development/TrafficSign/program/main_demo_cmake/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named main_demo

# Build rule for target.
main_demo: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 main_demo
.PHONY : main_demo

# fast build rule for target.
main_demo/fast:
	$(MAKE) -f build/src/CMakeFiles/main_demo.dir/build.make build/src/CMakeFiles/main_demo.dir/build
.PHONY : main_demo/fast

#=============================================================================
# Target rules for targets named yl_cpp_stuff_library

# Build rule for target.
yl_cpp_stuff_library: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 yl_cpp_stuff_library
.PHONY : yl_cpp_stuff_library

# fast build rule for target.
yl_cpp_stuff_library/fast:
	$(MAKE) -f build/src/ex_code/CMakeFiles/yl_cpp_stuff_library.dir/build.make build/src/ex_code/CMakeFiles/yl_cpp_stuff_library.dir/build
.PHONY : yl_cpp_stuff_library/fast

#=============================================================================
# Target rules for targets named yl_cuda_stuff_library

# Build rule for target.
yl_cuda_stuff_library: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 yl_cuda_stuff_library
.PHONY : yl_cuda_stuff_library

# fast build rule for target.
yl_cuda_stuff_library/fast:
	$(MAKE) -f build/src/ex_code/CMakeFiles/yl_cuda_stuff_library.dir/build.make build/src/ex_code/CMakeFiles/yl_cuda_stuff_library.dir/build
.PHONY : yl_cuda_stuff_library/fast

#=============================================================================
# Target rules for targets named yl_objects_detection_app

# Build rule for target.
yl_objects_detection_app: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 yl_objects_detection_app
.PHONY : yl_objects_detection_app

# fast build rule for target.
yl_objects_detection_app/fast:
	$(MAKE) -f build/src/ex_code/CMakeFiles/yl_objects_detection_app.dir/build.make build/src/ex_code/CMakeFiles/yl_objects_detection_app.dir/build
.PHONY : yl_objects_detection_app/fast

#=============================================================================
# Target rules for targets named yl_object_detector

# Build rule for target.
yl_object_detector: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 yl_object_detector
.PHONY : yl_object_detector

# fast build rule for target.
yl_object_detector/fast:
	$(MAKE) -f build/src/object_detector/CMakeFiles/yl_object_detector.dir/build.make build/src/object_detector/CMakeFiles/yl_object_detector.dir/build
.PHONY : yl_object_detector/fast

#=============================================================================
# Target rules for targets named yl_cuda_code

# Build rule for target.
yl_cuda_code: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 yl_cuda_code
.PHONY : yl_cuda_code

# fast build rule for target.
yl_cuda_code/fast:
	$(MAKE) -f build/src/cuda_code/CMakeFiles/yl_cuda_code.dir/build.make build/src/cuda_code/CMakeFiles/yl_cuda_code.dir/build
.PHONY : yl_cuda_code/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... main_demo"
	@echo "... yl_cpp_stuff_library"
	@echo "... yl_cuda_stuff_library"
	@echo "... yl_objects_detection_app"
	@echo "... yl_object_detector"
	@echo "... yl_cuda_code"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

