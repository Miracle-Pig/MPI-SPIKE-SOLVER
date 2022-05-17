#ifndef __SRC_UTIL_H__
#define __SRC_UTIL_H__

#include "enum.h"

#include <utility>
#include <cstddef>
#include <unordered_map>
#include <string>
#include <mpi.h>

/* Struct of  matrix */
typedef struct
{
    int n;  //! order of A
    int kl; //! subdiagno
    int ku; //! supdiagno
    double *storageGlob,
        *storageLoc;
} BandedMatrix;

typedef struct
{
    int ld;
    char format;
    double *storage;

} DenseMatrix;

typedef struct
{
    MPI_Comm comm;
    int myId;
    int nProcess;
    std::unordered_map<std::string, double> commTime;
} MpiInfo_t;

/* 
    !Options to deal with 
*/
typedef struct
{
    int IfPrint;       //! Print flag 1 for timing information, 2 for partition information, 0 for no printing (except errors, naturally)
    int OptimizeFlag;  //! spikeinit optimize flag. 0: Use user defined ratios. 1: Set tuning parameters to large NRHS. 2: DSPIKE_GBSV will attempt to find tuning ratios (See dspike_gbsv)
    YesOrNo_t IsPivot; //! is pivoting enabled. Defaults to 0, no
    YesOrNo_t HasB;
    YesOrNo_t IsSym;
    YesOrNo_t SelfMakeA;
    YesOrNo_t IsSovleAndFac;
    YesOrNo_t IsTestTime;
    double LargeRatio; //! ratio large. Defaults to 2.2
    double SmallRatio; //! ratio small. Defaults to 3.5
    double EqualRatio; //! ratio small. Defaults to 3.5
    YesOrNo_t IsRatioEqual;
    double K;         //! constant K -- tuning constant for system
    int lwork;        //! number of klu*klu blocks required for work array
    int IterativeMax; //! max number of iterations for iterative solve
    int DResidualTol; //! exponent for residual tolerance for double precision (and double complex)-- I.E, res tolerance is 10^-DResidualTol
    int SResidualTol; //! exponent for residual tolerance for single precision (and single complex)-- I.E, res tolerance is 10^-SResidualTol
    Norm_t Norm;      //! Norm type used.
    /*
        ! Internal paramaters (Unlikely to be useful to users) 
     */
    int PartitionCnt; //! partitions for level0
    int LevelsCnt;    //! levels
    int NProcesses;   //! number of processes
    int NPar1Proc;    //! one-process partitions
    int NPar2Proc;    //! two-process paritions
} Options_t;

void ABORT(const std::string &s, MPI_Comm &comm);

/* Memory allocate or deallocate */

int *IAllocate_1D(size_t len, int *allocInfo);
void IDeallocate_1D(int *array);

double *DAllocate_1D(size_t len, int *allocInfo);
double *DZAllocate_1D(size_t len, int *allocInfo);
void DDeallocate_1D(double *array);

double **DAllocate_2D(size_t r, size_t c, int *allocInfo);
double **DZAllocate_2D(size_t r, size_t c, int *allocInfo);
void DDeallocate_2D(double **array, int r);

double ***DAllocate_3D(size_t r, size_t c, size_t l, int *allocInfo);
void DDeallocate_3D(double ***array, size_t r, size_t c);

/* Allocate memory of banded global matrix A */
void DAllocateA(BandedMatrix *A, int n, int kl, int ku);

/* Set default Option */
void SetDefaultOptions(Options_t *options, MpiInfo_t *mpiInfo);

/* Read matrix from files */
void ReadAndBoradcastMatrix(BandedMatrix *A, int *nrhs, double *&globalF, char **argv, Options_t *options, MpiInfo_t *mpiInfo);
void ReadB(FILE *file, double *&B, int *nrhs);
void ReadBigBandA(FILE *file, double *A, int dim, int len, int supDiag, int subDiag);

/* Determine the number of the subDiagonals and supDiagonals */
void BandWidth(FILE *file, int *dim, int *len, int *supDiag, int *subDiag);

/* Caculate the size of partitions */
void CaculEqPartitionsSize(int *partitionsSize, int n, Options_t *options);
void CaculPartitionsSize(int *partitionsSize, int n, Options_t *options);
#endif