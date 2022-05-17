#include "factorization.h"
#include <string>

void GetReducedSys(int klu, double *Vi, double *Wi, double *Ri)
{
    int ldv, ldw, ldr;
    double alpha, beta;
    char uplo, transa, transb;

    ldv = 2 * klu, ldw = 2 * klu, ldr = 2 * klu;

    for (int i = 0; i < ldr; i++)
        for (int j = 0; j < 2 * klu; j++)
            if ((i < klu && j < klu) || ((i >= klu && i < ldr) && (j >= klu && j < 2 * klu)))
                Ri[TC(i, j, ldr)] = 0.0;

    /* Extract reduced system */

    uplo = 'F';
    dlacpy_(&uplo, &klu, &klu, Wi, &ldw, Ri + klu, &ldr);
    dlacpy_(&uplo, &klu, &klu, Vi + klu, &ldv, Ri + ldr * klu, &ldr);

    for (int i = 0; i < klu; i++)
    {
        Ri[TC(i, i, ldr)] = 1.0;
        Ri[TC(i + klu, i + klu, ldr)] = 1.0;
    }

    /* Caculate I-Wi+1{t,0}*Vi{b,0} */
    transa = 'N', transb = 'N', alpha = -1.0, beta = 1.0;
    dgemm_(&transa, &transb, &klu, &klu, &klu, &alpha, Ri + klu, &ldr, Ri + ldr * klu, &ldr, &beta, Ri, &ldr);
}

void ReducedSysSolve(int klu, int nrhs, double *Ri, double *xBot, int ldb, double *xTop, int ldt)
{
    int ldr,
        allocInfo, info;
    int *ipiv;
    double alpha, beta;
    char transa, transb, uplo;
    double *_R;

    ldr = 2 * klu;

    transa = 'N', transb = 'N', alpha = -1.0, beta = 1.0;
    dgemm_(&transa, &transb, &klu, &nrhs, &klu, &alpha, Ri + TC(klu, 0, ldr), &ldr, xBot, &ldb, &beta, xTop, &ldt); //Get gi+1{t}-Wi+1{t}*gi{b}

    _R = DAllocate_1D(klu * klu, &allocInfo);
    uplo = 'F';
    dlacpy_(&uplo, &klu, &klu, Ri, &ldr, _R, &klu);

    ipiv = IAllocate_1D(klu, &allocInfo);
    dgesv_(&klu, &nrhs, _R, &klu, ipiv, xTop, &ldt, &info); //Solve _Ri*Vi+1{t,l+1}=Vi+1{t,l}-Wi+1{t,l}*0

    transa = 'N', transb = 'N', alpha = -1.0, beta = 1.0, ldr = 2 * klu;
    dgemm_(&transa, &transb, &klu, &nrhs, &klu, &alpha, Ri + TC(0, klu, ldr), &ldr, xTop, &ldt, &beta, xBot, &ldb); //Solve Vi{b,l+1}=0-Vi{b,l}*Vi+1{t,l+1},

    IDeallocate_1D(ipiv);
    DDeallocate_1D(_R);
}

/* S[1]=D[1]*D[2]*...*D[n-1]*S[n],n=level*/
void RecursiveSpike(int kl, int ku, double ***V, double ***W, double **R, int nPart, int nLevel, MpiInfo_t *mpiInfo)
{
    int myId, p,
        nSubSys,
        nSubSpike,
        klu,
        ldv, ldw,
        allocInfo;

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
    ldv = 2 * klu, ldw = 2 * klu;

    aux1 = DZAllocate_1D(klu * klu, &allocInfo); //! aux1(xb/xi{b}/Vp{b,j+1})
    aux2 = DZAllocate_1D(klu * klu, &allocInfo); //! aux2(xt/xi+1{t}/Vp+1{t,j+1})

    /* 
        ! Example:nPart = 8,nLevel = 3
        ! At 1st level,j = 0:process 0,2,4,6;
        !              partition 0,1,2,3,4,5 solve Vp{,1},partition 2,3,4,5,6,7 solve Wp{,1}
        ! At 2nd level,j = 1:process 1,5;
        !              partition 0,1,2,3 solve Vp{,2},partition 4,5,6,7 solve Wp{,2}
     */

    for (int j = 0; j < nLevel - 1; j++)
    {
#ifdef DETAIL_TIME
        roundTime = MPI_Wtime();
#endif
        nSubSys = pow(2, nLevel - j - 1); //! Number of reduced system to be solved
        nSubSpike = pow(2, j);            //! Number of spikes by subsystem
        /* 
            ! Get spike V of S[j+1] 
         */
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

            std::cout << "Process " << p << " recursive time of simplifing matrix Si to get matrix Vi+1: " << detailTime << " in " << j + 1 << " level" << std::endl;
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

            std::cout << "Process " << p << " recursive time of simplifing matrix Si to get matrix Wi+1: " << detailTime << " in " << j + 1 << " level" << std::endl;
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
        }

        for (int id = 0; id < nPart; id++)
        {
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
            mpiInfo->commTime["Time of boardcasting matirx Vj+1 and Wj+1 at level " + std::to_string((long long)j)] = commMaxTime;
        }
#endif
        //! All processes synchronize before computing reduce system for level j+1
        // MPI_Barrier(mpiInfo->comm);

        /* 
            !New reduce systems for level j+1 
         */
#ifdef DETAIL_TIME
        detailTime = MPI_Wtime();
#endif
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

        std::cout << "Process " << p << " recursive time of getting new reduce systems: " << detailTime << " in " << j + 1 << " level" << std::endl;
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
            mpiInfo->commTime["Time of boardcasting new reduce systems at level " + std::to_string((long long)j)] = commMaxTime;
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
