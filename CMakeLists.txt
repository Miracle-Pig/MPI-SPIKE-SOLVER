cmake_minimum_required(VERSION 2.8)

project(MpiSpikeSolver)

set(CMAKE_C_COMPILER /home/shw/Zone/software/mpich-3.4.2/mpich-install/bin/mpicc)
set(CMAKE_C_FLAGS "-O0 -fPIC -ggdb -std=c99 -Wall")


set(CMAKE_CXX_COMPILER /home/shw/Zone/software/mpich-3.4.2/mpich-install/bin/mpicxx)
set(CMAKE_CXX_FLAGS "-O0 -fPIC -ggdb -std=c++11 -Wall")


add_definitions(-DMPI_DEBUG)

# add_definitions(-DDETAIL_TIME)
# add_definitions(-DSTAGE_TIME)
# add_definitions(-DSECTION_TIME)
# add_definitions(-DTOTAL_TIME)
# add_definitions(-DCOMM_TIME)

# add_definitions(-DINFO_PRINT)


set(INC_DIR
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/factorization
    ${PROJECT_SOURCE_DIR}/src/solve
    ${PROJECT_SOURCE_DIR}/src/facAndSolve
    ${PROJECT_SOURCE_DIR}/include
    /home/shw/Zone/software/mpich-3.4.2/mpich-install/include/
    # /opt/intel/oneapi/mpi/2021.3.0/include
)

set(LINK_DIR
    ${PROJECT_SOURCE_DIR}/lib
    # /opt/intel/oneapi/mpi/2021.3.0/lib
    /home/shw/Zone/software/mpich-3.4.2/mpich-install/lib/
    /home/shw/Zone/project/spike-1.0/lib/x64
    /home/shw/Zone/software/scalapack-2.1.0
)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

# find_package(MPI)
# if(MPI_C_FOUND)
#     set(CMAKE_C_FLAGS "${MPI_C_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${MPI_CXX_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}" )
# endif()

set(LIB_SRC
    src/util.cc
    src/factorization/FactorizationStage.cc
    src/factorization/FacUtils.cc
    src/solve/SolveStage.cc
    src/solve/SolveUtils.cc
    src/facAndSolve/FacAndSolve.cc
    src/facAndSolve/RecursiveStage.cc
)

add_library(mpi_spike_solver STATIC ${LIB_SRC})

set(LIBS
    mpi_spike_solver
    spike
    # mkl_scalapack_ilp64
    # mkl_intel_ilp64
    # mkl_core
    # mkl_intel_thread
    # mkl_blacs_intelmpi_ilp64
    # iomp5
    scalapack
    lapack
    blas
    gfortran
    m
    mpi
    
)

add_executable(test_spike test/test_spike.cc)
target_link_libraries(test_spike ${LIBS})


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)