#include "facandsolve.h"

void RecursiveStage(int kl, int ku, int nrhs, double ***V, double ***W, double **R, double **Gr, int nPart, int nLevel, MpiInfo_t *mpiInfo)
{
    int klu,
        ldv, ldw, ldgr,
        allocInfo,
        nSubSys, nSubSpike,
        myId, p;
    double alpha, beta;
    double *aux1, *aux2,
        *aux3, *aux4;
    char transa, transb, uplo;
    //! true: fac stage;false: solve stage

#ifdef COMM_TIME
    double commTime, commMaxTime;
#endif
#ifdef DETAIL_TIME
    double detailTime, roundTime;
#endif

    myId = mpiInfo->myId;
    klu = std::max(kl, ku);
    ldv = 2 * klu, ldw = 2 * klu, ldgr = 2 * klu;

    aux1 = DZAllocate_1D(klu * klu, &allocInfo); //! aux1(xb/xi{b}/Vp{b,j+1})
    aux2 = DZAllocate_1D(klu * klu, &allocInfo); //! aux2(xt/xi+1{t}/Vp+1{t,j+1})

    aux3 = DZAllocate_1D(klu * nrhs, &allocInfo);
    aux4 = DZAllocate_1D(klu * nrhs, &allocInfo);

    transa = 'N', transb = 'N';

    /* 
        ! Example:nPart = 8,nLevel = 3
        ! At 1st level,j = 0:process 0,2,4,6;
        !              partition 0,1,2,3,4,5 solve Vp{,1},partition 2,3,4,5,6,7 solve Wp{,1}
        ! At 2nd level,j = 1:process 1,5;
        !              partition 0,1,2,3 solve Vp{,2},partition 4,5,6,7 solve Wp{,2}
     */

    for (int jj = 0; jj < nLevel; jj++)
    {
#ifdef DETAIL_TIME
        roundTime = MPI_Wtime();
#endif

        if (jj > 0)
        {

            int j = jj - 1;
            nSubSys = pow(2, nLevel - j - 1); //! Number of reduced system to be solved
            nSubSpike = pow(2, j);            //! Number of spikes by subsystem
            /* 
                ! Get spike V of S[j+1] 
            */
            //! #1
            p = myId; //! 1st:p=0,2,4,myId=0,2,4; 2nd:p=1, myId=1;
            if (((p - nSubSpike + 1) % (nPart / nSubSys)) == 0 && (p + 1 < nPart - nPart / nSubSys))
            {
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime();
#endif
                /*   
                    ! Solve:
                    ! | I         Vp{b,j+1}|  |Vp{b,j+1}  |  = |0        |
                    ! | Wp+1{t,j} I        |  |Vp+1{t,j+1}|    |Vp+1{t,j}|
                */
                //! aux2==>Vp+1{t,j}
                for (int i = 0; i < klu * klu; i++)
                    aux1[i] = 0.0;
                uplo = 'F';
                dlacpy_(&uplo, &klu, &klu, V[j][p + 1], &ldv, aux2, &klu);

                //! Get Vp{b,j+1}(aux1) Vp+1{t,j+1}(aux2)
                ReducedSysSolve(klu, klu, R[p], aux1, klu, aux2, klu);
                /* 
                    !Get the upper part of spike V of S[j+1]
                */
                if (j > 0)
                {
                    //! Get Vl{t,j+1} and Vl{b,j+1}
                    for (int l = p - nSubSpike + 1; l <= p - 1; l++)
                    {
                        transa = 'N', transb = 'N', alpha = -1.0, beta = 0.0;
                        dgemm_(&transa, &transb, &ldv, &klu, &klu, &alpha, V[j][l], &ldv, aux2, &klu, &beta, V[j + 1][l], &ldv);
                    }
                }
                //! Get Vp{t,j+1}
                //! Solve Vp{t,j+1}=-Vp{t,j}*Vp+1{t,j+1}
                transa = 'N', transb = 'N', alpha = -1.0, beta = 0.0;
                dgemm_(&transa, &transb, &klu, &klu, &klu, &alpha, V[j][p], &ldv, aux2, &klu, &beta, V[j + 1][p], &ldv);
                //! Get Vp{b,j+1}
                uplo = 'F';
                dlacpy_(&uplo, &klu, &klu, aux1, &klu, V[j + 1][p] + klu, &ldv);
                /*
                    !  Get the lower part of spike V of S[j+1]
                */

                for (int l = p + 1; l <= p + nSubSpike; l++)
                {
                    //! Get Vl{t,j+1} and Vl{b,j+1}
                    if (l != p + 1)
                    {
                        uplo = 'F';
                        dlacpy_(&uplo, &ldv, &klu, V[j][l], &ldv, V[j + 1][l], &ldv);
                        transa = 'N', transb = 'N', alpha = -1.0, beta = 1.0;
                        dgemm_(&transa, &transb, &ldv, &klu, &klu, &alpha, W[j][l], &ldv, aux1, &klu, &beta, V[j + 1][l], &ldv);
                    }
                    else
                    {
                        //! Get Vp+1{t,j+1}
                        uplo = 'F';
                        dlacpy_(&uplo, &klu, &klu, aux2, &klu, V[j + 1][l], &ldv);
                        dlacpy_(&uplo, &klu, &klu, V[j][l] + klu, &ldv, V[j + 1][l] + klu, &ldv);
                        //! Get Vp+1{b,j+1}
                        //! Solve Vp+1{b,j+1}=Vp+1{b,j+1}-Wp+1{b,j}*Vp{b,j+1}
                        transa = 'N', transb = 'N', alpha = -1.0, beta = 1.0;
                        dgemm_(&transa, &transb, &klu, &klu, &klu, &alpha, W[j][l] + klu, &ldw, aux1, &klu, &beta, V[j + 1][l] + klu, &ldv);
                    }
                }
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime() - detailTime;

                std::cout << "Process " << p << " recursive time of simplifing matrix Si to get matrix Vi+1: " << detailTime << " in " << jj + 1 << " level" << std::endl;
                fflush(stdout);
#endif
            }

            /*  
                !Get spike W of S[j+1]
            */
            p = p - 1; //! 1st:p=2,4,6 myId=3,5,7;2nd:p=5,myId=6
            if ((p + 1 >= nSubSpike + nPart / nSubSys) && ((p + 1 - nSubSpike) % (nPart / nSubSys)) == 0)
            {
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime();
#endif
                for (int i = 0; i < klu * klu; i++)
                    aux2[i] = 0.0;
                uplo = 'F';
                dlacpy_(&uplo, &klu, &klu, W[j][p] + klu, &ldw, aux1, &klu);
                //! Get Wp{b,j+1}(aux1) Vp+1{t,j+1}(aux2)
                ReducedSysSolve(klu, klu, R[p], aux1, klu, aux2, klu);
                /* 
                !Get the upper part of spike W of S[j+1]
             */
                if (j > 0)
                {
                    for (int l = p - nSubSpike + 1; l <= p - 1; l++)
                    {
                        //! Get Wl{t,j+1} and Wl{b,j+1}
                        uplo = 'F';
                        dlacpy_(&uplo, &ldw, &klu, W[j][l], &ldw, W[j + 1][l], &ldw);
                        transa = 'N', transb = 'N', alpha = -1.0, beta = 1.0;
                        dgemm_(&transa, &transb, &ldv, &klu, &klu, &alpha, V[j][l], &ldv, aux2, &klu, &beta, W[j + 1][l], &ldw);
                    }
                }
                uplo = 'F';
                dlacpy_(&uplo, &klu, &klu, W[j][p], &ldw, W[j + 1][p], &ldw);
                //! Get Wp{t,j+1}
                transa = 'N', transb = 'N', alpha = -1.0, beta = 1.0;
                dgemm_(&transa, &transb, &klu, &klu, &klu, &alpha, V[j][p], &ldv, aux2, &klu, &beta, W[j + 1][p], &ldw);
                //! Get Wp{b,j+1}
                uplo = 'F';
                dlacpy_(&uplo, &klu, &klu, aux1, &klu, W[j + 1][p] + klu, &ldw);
                /* 
                !Get the lower part of spike W of S[j+1]
             */
                for (int l = p + 1; l <= p + nSubSpike; l++)
                {
                    if (l != p + 1)
                    {
                        //! Get Wl{t,j+1} and Wl{b,j+1}
                        transa = 'N', transb = 'N', alpha = -1.0, beta = 0.0;
                        dgemm_(&transa, &transb, &ldv, &klu, &klu, &alpha, W[j][l], &ldv, aux1, &klu, &beta, W[j + 1][l], &ldw);
                    }
                    else
                    {
                        //! Get Wp+1{t,j+1}
                        uplo = 'F';
                        dlacpy_(&uplo, &klu, &klu, aux2, &klu, W[j + 1][l], &ldw);
                        //! Get Wp+1{b,j+1}
                        transa = 'N', transb = 'N', alpha = -1.0, beta = 0.0;
                        dgemm_(&transa, &transb, &klu, &klu, &klu, &alpha, W[j][l] + klu, &ldw, aux1, &klu, &beta, W[j + 1][l] + klu, &ldw);
                    }
                }
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime() - detailTime;

                std::cout << "Process " << p << " recursive time of simplifing matrix Si to get matrix Wi+1: " << detailTime << " in " << jj + 1 << " level" << std::endl;
                fflush(stdout);
#endif
            }

#ifdef COMM_TIME
            commTime = MPI_Wtime();
#endif

            for (int id = 0; id < nPart; id++)
            {
                if (((id - nSubSpike + 1) % (nPart / nSubSys)) == 0 && (id + 1 < nPart - nPart / nSubSys))
                {
                    if (j > 0)
                    {
                        for (int l = id - nSubSpike + 1; l <= id - 1; l++)
                        {
                            MPI_Bcast(V[j + 1][l], ldv * klu, MPI_DOUBLE, id, mpiInfo->comm);
                        }
                    }
                    MPI_Bcast(V[j + 1][id], ldv * klu, MPI_DOUBLE, id, mpiInfo->comm);
                    for (int l = id + 1; l <= id + nSubSpike; l++)
                    {
                        MPI_Bcast(V[j + 1][l], ldv * klu, MPI_DOUBLE, id, mpiInfo->comm);
                    }
                }
                int _id = id - 1;
                if ((_id + 1 >= nSubSpike + nPart / nSubSys) && ((_id + 1 - nSubSpike) % (nPart / nSubSys)) == 0)
                {
                    if (j > 0)
                    {
                        for (int l = _id - nSubSpike + 1; l <= _id - 1; l++)
                        {
                            MPI_Bcast(W[j + 1][l], ldv * klu, MPI_DOUBLE, id, mpiInfo->comm);
                        }
                    }

                    MPI_Bcast(W[j + 1][_id], ldv * klu, MPI_DOUBLE, id, mpiInfo->comm);
                    /* 
                        !Get the lower part of spike W of S[j+1]
                    */
                    for (int l = _id + 1; l <= _id + nSubSpike; l++)
                    {
                        MPI_Bcast(W[j + 1][l], ldv * klu, MPI_DOUBLE, id, mpiInfo->comm);
                    }
                }
            }

#ifdef COMM_TIME
            commTime = MPI_Wtime() - commTime;
            MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
            if (myId == 0)
            {
                mpiInfo->commTime["Time of boardcasting matirx Vj+1 and Wj+1 at level " + std::to_string((long long)jj)] = commMaxTime;
            }
#endif

#ifdef DETAIL_TIME
            detailTime = MPI_Wtime();
#endif

            //! All processes synchronize before computing reduce system for level j+1
            // MPI_Barrier(mpiInfo->comm);
            /* 
                !New reduce systems for level j+1 
            */
            nSubSys = pow(2, nLevel - j - 2);
            nSubSpike = pow(2, j + 1);
            p = myId;
            if (((p + 1 - nSubSpike) % (nPart / nSubSys)) == 0)
            {
                //! overwritten the original R[p]???
                GetReducedSys(klu, V[j + 1][p], W[j + 1][p + 1], R[p]);
            }
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime() - detailTime;

            std::cout << "Process " << p << " recursive time of getting new reduce systems: " << detailTime << " in " << jj << " level" << std::endl;
            fflush(stdout);
#endif

#ifdef COMM_TIME
            commTime = MPI_Wtime();
#endif

            for (int id = 0; id < nPart; id++)
            {
                if (((id + 1 - nSubSpike) % (nPart / nSubSys)) == 0)
                {
                    MPI_Bcast(R[id], 2 * klu * 2 * klu, MPI_DOUBLE, id, mpiInfo->comm);
                }
            }
#ifdef COMM_TIME
            commTime = MPI_Wtime() - commTime;
            MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
            if (myId == 0)
            {
                mpiInfo->commTime["Time of boardcasting new reduce systems at level " + std::to_string((long long)jj)] = commMaxTime;
            }
#endif
        }

        nSubSys = pow(2, nLevel - jj - 1);
        nSubSpike = pow(2, jj);

        for (int i = nSubSpike - 1; i < nPart; i += nPart / nSubSys)
        {
            if (myId == i)
            {
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime();
#endif
                for (int k = 0; k < klu * nrhs; k++)
                {
                    aux3[k] = 0.0;
                    aux4[k] = 0.0;
                }
                /* 
                    !Get Gi{b,j+1} and Gi+1{t,j+1} 
                */
                //! 1st level:i = 0,2,4,6;2nd level
                uplo = 'F';
                dlacpy_(&uplo, &klu, &nrhs, Gr[i + 1], &ldgr, aux4, &klu);

                ReducedSysSolve(klu, nrhs, R[i], Gr[i] + klu, ldgr, aux4, klu);

                dlacpy_(&uplo, &klu, &nrhs, Gr[i] + klu, &ldgr, aux3, &klu);
                /* 
                    !Get the upper part of Gl~i{t&b,j+1}
                */
                for (int l = i - nSubSpike + 1; l <= i - 1; l++)
                {
                    alpha = -1.0, beta = 1.0;
                    dgemm_(&transa, &transb, &ldv, &nrhs, &klu, &alpha, V[jj][l], &ldv, aux4, &klu, &beta, Gr[l], &ldgr);
                }
                //! Get Gi{t,j+1}
                alpha = -1.0, beta = 1.0;
                dgemm_(&transa, &transb, &klu, &nrhs, &klu, &alpha, V[jj][i], &ldv, aux4, &klu, &beta, Gr[i], &ldgr);

                /* 
                    !Get the lower part of Gi+1~l{t&b,j+1}
                 */
                for (int l = i + 1; l <= i + nSubSpike; l++)
                {
                    if (l != i + 1)
                    {
                        alpha = -1.0, beta = 1.0;
                        dgemm_(&transa, &transb, &ldw, &nrhs, &klu, &alpha, W[jj][l], &ldw, aux3, &klu, &beta, Gr[l], &ldgr);
                    }
                    else
                    {
                        //! Get Gi+1{b,j+1}
                        uplo = 'F';
                        dlacpy_(&uplo, &klu, &nrhs, aux4, &klu, Gr[l], &ldgr);

                        alpha = -1.0, beta = 1.0;
                        dgemm_(&transa, &transb, &klu, &nrhs, &klu, &alpha, W[jj][l] + klu, &ldw, aux3, &klu, &beta, Gr[l] + klu, &ldgr);
                    }
                }
#ifdef DETAIL_TIME
                detailTime = MPI_Wtime() - detailTime;

                std::cout << "Process " << i << " recursive time of simplifing matrix Yi :" << detailTime << " in " << jj + 1 << " level" << std::endl;
                fflush(stdout);
#endif
            }
        }
#ifdef COMM_TIME
        commTime = MPI_Wtime();
#endif
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
            mpiInfo->commTime["Time of boardcasting matirx Gr at level " + std::to_string((long long)jj)] = commMaxTime;
        }
#endif

#ifdef DETAIL_TIME
        roundTime = MPI_Wtime() - roundTime;
        std::cout << "Time of " << jj + 1 << "th round of process " << myId << " :" << roundTime << std::endl;
#endif
    }

    DDeallocate_1D(aux1);
    DDeallocate_1D(aux2);
    DDeallocate_1D(aux3);
    DDeallocate_1D(aux4);
}
