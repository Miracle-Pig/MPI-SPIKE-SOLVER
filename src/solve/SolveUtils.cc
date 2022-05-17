#include "solve.h"

void SolveReducedSys(int kl, int ku, int nrhs, double ***V, double ***W, double **R, int nPart, int nLevel, double **Gr, MpiInfo_t *mpiInfo)
{
    int klu,
        ldv, ldw, ldgr,
        allocInfo,
        nSubSys, nSubSpike,
        myId;
    double alpha, beta;
    double *aux1, *aux2;
    char transa, transb, uplo;

#ifdef COMM_TIME
    double commTime, commMaxTime;
#endif
#ifdef DETAIL_TIME
    double detailTime, roundTime;
#endif

    myId = mpiInfo->myId;
    klu = std::max(kl, ku);
    ldv = 2 * klu, ldw = 2 * klu, ldgr = 2 * klu;
    transa = 'N', transb = 'N';

    aux1 = DZAllocate_1D(klu * nrhs, &allocInfo);
    aux2 = DZAllocate_1D(klu * nrhs, &allocInfo);

    for (int j = 0; j < nLevel; j++)
    {
#ifdef DETAIL_TIME
        roundTime = MPI_Wtime();
#endif
        nSubSys = pow(2, nLevel - j - 1);
        nSubSpike = pow(2, j);
        for (int i = 0; i < klu * nrhs; i++)
        {
            aux1[i] = 0.0;
            aux2[i] = 0.0;
        }

        for (int i = nSubSpike - 1; i < nPart; i += nPart / nSubSys)
        {
            if (myId == i)
            {
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime();
#endif
                /* 
                    !Get Gi{b,j+1} and Gi+1{t,j+1} 
                */
                //! 1st level:i = 0,2,4,6;2nd level
                uplo = 'F';
                dlacpy_(&uplo, &klu, &nrhs, Gr[i + 1], &ldgr, aux2, &klu);

                ReducedSysSolve(klu, nrhs, R[i], Gr[i] + klu, ldgr, aux2, klu);

                dlacpy_(&uplo, &klu, &nrhs, Gr[i] + klu, &ldgr, aux1, &klu);
                /* 
                    !Get the upper part of Gl~i{t&b,j+1}
                */
                for (int l = i - nSubSpike + 1; l <= i - 1; l++)
                {
                    alpha = -1.0, beta = 1.0;
                    dgemm_(&transa, &transb, &ldv, &nrhs, &klu, &alpha, V[j][l], &ldv, aux2, &klu, &beta, Gr[l], &ldgr);
                }
                //! Get Gi{t,j+1}
                alpha = -1.0, beta = 1.0;
                dgemm_(&transa, &transb, &klu, &nrhs, &klu, &alpha, V[j][i], &ldv, aux2, &klu, &beta, Gr[i], &ldgr);

                /* 
                    !Get the lower part of Gi+1~l{t&b,j+1}
                 */
                for (int l = i + 1; l <= i + nSubSpike; l++)
                {
                    if (l != i + 1)
                    {
                        alpha = -1.0, beta = 1.0;
                        dgemm_(&transa, &transb, &ldw, &nrhs, &klu, &alpha, W[j][l], &ldw, aux1, &klu, &beta, Gr[l], &ldgr);
                    }
                    else
                    {
                        //! Get Gi+1{b,j+1}
                        uplo = 'F';
                        dlacpy_(&uplo, &klu, &nrhs, aux2, &klu, Gr[l], &ldgr);

                        alpha = -1.0, beta = 1.0;
                        dgemm_(&transa, &transb, &klu, &nrhs, &klu, &alpha, W[j][l] + klu, &ldw, aux1, &klu, &beta, Gr[l] + klu, &ldgr);
                    }
                }
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime() - detailTime;

                std::cout << "Process " << i << " recursive time of simplifing matrix Yi :" << detailTime << " in " << j + 1 << " level" << std::endl;
                fflush(stdout);
#endif
            }
        }

#ifdef COMM_TIME
        commTime = MPI_Wtime();
#endif
        //! 广播时，广播的根进程不占用进程时间
        for (int i = nSubSpike - 1; i < nPart; i += nPart / nSubSys)
        {
            for (int l = i - nSubSpike + 1; l <= i - 1; l++)
            {
                MPI_Bcast(Gr[l], 2 * klu * nrhs, MPI_DOUBLE, i, mpiInfo->comm);
            }
            MPI_Bcast(Gr[i], 2 * klu * nrhs, MPI_DOUBLE, i, mpiInfo->comm);
            for (int l = i + 1; l <= i + nSubSpike; l++)
            {
                if (l != i + 1)
                {
                    MPI_Bcast(Gr[l], 2 * klu * nrhs, MPI_DOUBLE, i, mpiInfo->comm);
                }
                else
                {
                    MPI_Bcast(Gr[l], 2 * klu * nrhs, MPI_DOUBLE, i, mpiInfo->comm);
                }
            }
        }
#ifdef COMM_TIME
        commTime = MPI_Wtime() - commTime;
        MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
        if (myId == 0)
        {
            mpiInfo->commTime["Time of boardcasting matirx Gr at level" + std::to_string((long long)j)] = commMaxTime;
        }
#endif

#ifdef DETAIL_TIME
        roundTime = MPI_Wtime() - roundTime;
        std::cout << "Time of " << j + 1 << "th round of process " << myId << " :" << roundTime << std::endl;
#endif
    }

    DDeallocate_1D(aux1);
    DDeallocate_1D(aux2);
}