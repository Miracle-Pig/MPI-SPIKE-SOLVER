#ifndef __FACANDSOLVE_H__
#define __FACANDSOLVE_H__

#include "MpiSpikeSolver.h"

void FacAndSolve(BandedMatrix *A, double *localF, int nrhs, double ***V, double ***W,
                 double **R, int *ipiv, Options_t *options, MpiInfo_t *mpiInfo);

void RecursiveStage(int kl, int ku, int nrhs, double ***V, double ***W, double **R,
                    double **Gr, int nPart, int nLevel, MpiInfo_t *mpiInfo);

#endif