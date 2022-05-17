#ifndef __FACTORIZATION_H__
#define __FACTORIZATION_H__

#include "MpiSpikeSolver.h"

void FactorizationStage(BandedMatrix *A, double ***V, double ***W, double **R, int *ipiv, Options_t *options, MpiInfo_t *mpiInfo);

void GetReducedSys(int klu, double *Vi, double *Wi, double *Ri);

void RecursiveSpike(int kl, int ku, double ***V, double ***W, double **R, int nPart, int nLevel, MpiInfo_t *mpiInfo);

void ReducedSysSolve(int klu, int nrhs, double *Ri, double *xBot, int ldb, double *xTop, int ldt);

#endif