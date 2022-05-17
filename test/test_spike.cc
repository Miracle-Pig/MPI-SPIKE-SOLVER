#include "MpiSpikeSolver.h"

/* Usage:mpirun -n nproces ./test_spike matrixA matrixB */
int main(int argc, char **argv)
{

  int myId, nProcess, klu, nLevel, nPart,
      n,
      nrhs,
      allocInfo;
  int *ipiv;
  BandedMatrix A;
  double *localF,
      ***V, ***W,
      **R;

#ifdef SECTION_TIME
  double sectionTime, sectionMaxTime;
#endif

  // #ifdef TOTAL_TIME
  //   double totalTime, totalMaxTime;
  // #endif

  Options_t options;
  MpiInfo_t mpiInfo;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcess);

  mpiInfo.myId = myId;
  mpiInfo.nProcess = nProcess;
  mpiInfo.comm = MPI_COMM_WORLD;

  /* 
    ! Set default options 
  */
  if(myId == 0)
  {
    std::cout<< "The number of process: " << nProcess << std::endl;
  }
  SetDefaultOptions(&options, &mpiInfo);
  // options.IsPivot = YES;
  // options.IsSym = YES;
  options.SelfMakeA = YES;
  // options.IsRatioEqual = YES;

  /* 
    ! Reading matrix file 
  */
#ifdef SECTION_TIME
  sectionTime = MPI_Wtime();
#endif

  ReadAndBoradcastMatrix(&A, &nrhs, localF, argv, &options, &mpiInfo);

#ifdef SECTION_TIME
  sectionTime = MPI_Wtime() - sectionTime;
  MPI_Reduce(&sectionTime, &sectionMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo.comm);
  if (myId == 0)
  {
    std::cout << "Time of reading and boradcasting matrix: " << sectionMaxTime << std::endl;
  }
#endif

  n = A.n;
  nPart = options.PartitionCnt;
  nLevel = options.LevelsCnt;
  klu = std::max(A.kl, A.ku);

  V = DAllocate_3D(nLevel, nPart, (2 * klu) * klu, &allocInfo);
  W = DAllocate_3D(nLevel, nPart, (2 * klu) * klu, &allocInfo);
  R = DZAllocate_2D(nPart, 2 * klu * 2 * klu, &allocInfo);
  if (options.IsPivot)
  {
    ipiv = IAllocate_1D(n, &allocInfo);
  }
  if (klu < nrhs)
  {
    options.IsSovleAndFac = NO;
  }
  else
  {
    options.IsSovleAndFac = YES;
  }
  // options.IsSovleAndFac = NO;

#ifdef SECTION_TIME
  sectionTime = MPI_Wtime();
#endif

  if (!options.IsSovleAndFac)
  {
    if (myId == 0)
    {
      std::cout << "Separate factorization and solve stage:" << std::endl;
    }
    /* 
      ! Spike factorization stage,get matrix S in 
      ! AX=B==>DSX=B
    */
    FactorizationStage(&A, V, W, R, ipiv, &options, &mpiInfo);

    /* 
      ! Spike solve stage
    */
    SolveStage(&A, nrhs, localF, V, W, R, ipiv, &options, &mpiInfo);
  }
  else
  {
    if (myId == 0)
    {
      std::cout << "Merge factorization and solve stage:" << std::endl;
    }
    FacAndSolve(&A, localF, nrhs, V, W, R, ipiv, &options, &mpiInfo);
  }

#ifdef SECTION_TIME
  sectionTime = MPI_Wtime() - sectionTime;
  MPI_Reduce(&sectionTime, &sectionMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo.comm);
  if (myId == 0)
  {
    std::cout << "Time of spike: " << sectionMaxTime << std::endl;
  }
#endif

#ifdef COMM_TIME
  if (myId == 0)
  {
    for (auto it : mpiInfo.commTime)
    {
      std::cout << it.first << ": " << it.second << std::endl;
    }
  }
#endif

  if (myId == 0)
  {
    for (int i = 0; i < 10; i++)
    {
      std::cout << std::setprecision(17) << localF[i] << std::endl;
    }
  }

  // #ifdef MPI_DEBUG
  //   if (myId == 0)
  //   {
  //     std::cout << "Breakpoint 1" << std::endl;
  //   }
  //   int gdb_break = 1;
  //   while (gdb_break)
  //   {
  //   };
  // #endif

  if (options.IsPivot)
  {
    IDeallocate_1D(ipiv);
  }
  DDeallocate_3D(V, nLevel, nPart);
  DDeallocate_3D(W, nLevel, nPart);
  DDeallocate_2D(R, nPart);
  DDeallocate_1D(A.storageLoc);
  DDeallocate_1D(localF);

  MPI_Finalize();
  return 0;
}
