#ifndef __SOLVE_H__
#define __SOLVE_H__

#include "MpiSpikeSolver.h"

void SolveStage(BandedMatrix *A, int nrhs, double *F, double ***V, double ***W, double **R, int *ipiv, Options_t *options, MpiInfo_t *mpiInfo);

void SolveReducedSys(int kl, int ku, int nrhs, double ***V, double ***W, double **R, int nPart, int nLevel, double **Gr, MpiInfo_t *mpiInfo);

#endif