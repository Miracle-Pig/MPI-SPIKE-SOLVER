# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shw/Zone/workspace/MpiSpikeSolver

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shw/Zone/workspace/MpiSpikeSolver/build

# Include any dependencies generated for this target.
include CMakeFiles/test_spike.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_spike.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_spike.dir/flags.make

CMakeFiles/test_spike.dir/test/test_spike.cc.o: CMakeFiles/test_spike.dir/flags.make
CMakeFiles/test_spike.dir/test/test_spike.cc.o: ../test/test_spike.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shw/Zone/workspace/MpiSpikeSolver/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_spike.dir/test/test_spike.cc.o"
	/home/shw/Zone/software/mpich-3.4.2/mpich-install/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_spike.dir/test/test_spike.cc.o -c /home/shw/Zone/workspace/MpiSpikeSolver/test/test_spike.cc

CMakeFiles/test_spike.dir/test/test_spike.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_spike.dir/test/test_spike.cc.i"
	/home/shw/Zone/software/mpich-3.4.2/mpich-install/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shw/Zone/workspace/MpiSpikeSolver/test/test_spike.cc > CMakeFiles/test_spike.dir/test/test_spike.cc.i

CMakeFiles/test_spike.dir/test/test_spike.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_spike.dir/test/test_spike.cc.s"
	/home/shw/Zone/software/mpich-3.4.2/mpich-install/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shw/Zone/workspace/MpiSpikeSolver/test/test_spike.cc -o CMakeFiles/test_spike.dir/test/test_spike.cc.s

CMakeFiles/test_spike.dir/test/test_spike.cc.o.requires:

.PHONY : CMakeFiles/test_spike.dir/test/test_spike.cc.o.requires

CMakeFiles/test_spike.dir/test/test_spike.cc.o.provides: CMakeFiles/test_spike.dir/test/test_spike.cc.o.requires
	$(MAKE) -f CMakeFiles/test_spike.dir/build.make CMakeFiles/test_spike.dir/test/test_spike.cc.o.provides.build
.PHONY : CMakeFiles/test_spike.dir/test/test_spike.cc.o.provides

CMakeFiles/test_spike.dir/test/test_spike.cc.o.provides.build: CMakeFiles/test_spike.dir/test/test_spike.cc.o


# Object files for target test_spike
test_spike_OBJECTS = \
"CMakeFiles/test_spike.dir/test/test_spike.cc.o"

# External object files for target test_spike
test_spike_EXTERNAL_OBJECTS =

../bin/test_spike: CMakeFiles/test_spike.dir/test/test_spike.cc.o
../bin/test_spike: CMakeFiles/test_spike.dir/build.make
../bin/test_spike: ../lib/libmpi_spike_solver.a
../bin/test_spike: CMakeFiles/test_spike.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shw/Zone/workspace/MpiSpikeSolver/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test_spike"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_spike.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_spike.dir/build: ../bin/test_spike

.PHONY : CMakeFiles/test_spike.dir/build

CMakeFiles/test_spike.dir/requires: CMakeFiles/test_spike.dir/test/test_spike.cc.o.requires

.PHONY : CMakeFiles/test_spike.dir/requires

CMakeFiles/test_spike.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_spike.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_spike.dir/clean

CMakeFiles/test_spike.dir/depend:
	cd /home/shw/Zone/workspace/MpiSpikeSolver/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shw/Zone/workspace/MpiSpikeSolver /home/shw/Zone/workspace/MpiSpikeSolver /home/shw/Zone/workspace/MpiSpikeSolver/build /home/shw/Zone/workspace/MpiSpikeSolver/build /home/shw/Zone/workspace/MpiSpikeSolver/build/CMakeFiles/test_spike.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_spike.dir/depend

