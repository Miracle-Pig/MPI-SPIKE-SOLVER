#include "solve.h"

void SolveStage(BandedMatrix *A, int nrhs, double *localF, double ***V, double ***W, double **R, int *ipiv, Options_t *options, MpiInfo_t *mpiInfo)
{

    int nPart, nLevel, myId,
        allocInfo, info,
        kl, ku, klu, kd, lda, ldf, ldgr, _lda, _n, n,
        mySize, _mySize, p,
        startAiL, startAiU, startFi, startGri,
        startBi, startBj, startCi, startCj,
        incx, incy, inc;
    int *partitionsSize,
        *ipivAux;
    double alpha;
    double
        **Gr,
        *G,
        *localA,
        *aux;
    char uplo, trans, transa, diag, side;
    YesOrNo_t pivot;

#ifdef COMM_TIME
    double commTime, commMaxTime;
#endif

#ifdef STAGE_TIME
    double stageTime, stageTimes[mpiInfo->nProcess];
    MPI_Status status;
#endif

// #ifdef DETAIL_TIME
//     double detailTime;
// #endif

    nPart = options->PartitionCnt;
    nLevel = options->LevelsCnt;
    pivot = options->IsPivot;
    myId = mpiInfo->myId;
    n = A->n;
    kl = A->kl, ku = A->ku;
    localA = A->storageLoc;
    klu = std::max(kl, ku);
    ldgr = 2 * klu;

    Gr = DZAllocate_2D(nPart, 2 * klu * nrhs, &allocInfo);
    partitionsSize = IAllocate_1D(nPart + 1, &allocInfo);
    if (options->IsRatioEqual)
    {
        CaculEqPartitionsSize(partitionsSize, n, options);
    }
    else
    {
        CaculPartitionsSize(partitionsSize, n, options);
    }

    G = DZAllocate_1D((partitionsSize[nPart - 1] - partitionsSize[nPart - 2]) * nrhs, &allocInfo);

    p = myId;
    mySize = partitionsSize[p + 1] - partitionsSize[p];
    ldf = mySize;

    if (pivot)
    {
        startAiU = klu - kl;
        kd = kl + ku;
        lda = kl + ku + klu + 1;
    }
    else
    {
        startAiL = ku;
        startAiU = 0;
        kd = ku;
        lda = 1 + kl + ku;
    }
    startFi = 0;

    /* *************************
       * DY Factorization Stage*
       * Solve D*G=F           * 
       * *********************** */

#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif

    if (p < nPart - 1)
    {
        if (p == 0)
        {
            if (pivot)
            {
                startAiL = klu - kl;
                //! Solve L1*U1*G1=F1(half sweep)
                trans = 'N';
                dgbtrsl_(&trans, &mySize, &kl, &ku, &nrhs, localA + TC(startAiL, 0, lda), &lda, ipiv + partitionsSize[p], localF, &ldf, &info);
                //! Get U1*G1{b}
                uplo = 'F';
                dlacpy_(&uplo, &klu, &nrhs, localF + TC(mySize - klu, 0, ldf), &ldf, Gr[p] + klu, &ldgr);
            }
            else
            {
                //! Solve L1*U1*G1=F1
                uplo = 'L', trans = 'N', diag = 'U';
                dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &kl, localA + TC(startAiL, 0, lda), &lda, localF, &ldf);
                //! Get U1*G1{b}
                uplo = 'F';
                dlacpy_(&uplo, &klu, &nrhs, localF + TC(mySize - klu, 0, ldf), &ldf, Gr[p] + TC(klu, 0, ldgr), &ldgr);
            }
            //! Get G1{b}(half sweep)
            uplo = 'U', trans = 'N', diag = 'N';
            dtbsm_(&uplo, &trans, &diag, &klu, &nrhs, &kd, localA + TC(startAiU, mySize - klu, lda), &lda, Gr[p] + klu, &ldgr);
        }
        else
        {
            mySize = partitionsSize[nPart - 1] - partitionsSize[nPart - 2];

            uplo = 'F';
            dlacpy_(&uplo, &mySize, &nrhs, localF, &ldf, G, &mySize);
            if (pivot)
            {
                startAiL = klu - kl;
                //! Solve Ai*Gi=Fi
                trans = 'N';
                dgbtrs_(&trans, &mySize, &kl, &ku, &nrhs, localA + TC(startAiL, 0, lda), &lda, ipiv + partitionsSize[p], G, &mySize, &info);
            }
            else
            {
                //! Solve Lp*Up*Gp=Fp
                uplo = 'L', trans = 'N', diag = 'U';
                dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &kl, localA + TC(startAiL, 0, lda), &lda, G, &mySize);
                //! Solve Up*Gp=inv(Lp)*Fp
                uplo = 'U', trans = 'N', diag = 'N';
                dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &ku, localA + TC(startAiU, 0, lda), &lda, G, &mySize);
            }
            //! Get Grp{t} and Grp{b}
            uplo = 'F';
            dlacpy_(&uplo, &klu, &nrhs, G, &mySize, Gr[p], &ldgr);
            dlacpy_(&uplo, &klu, &nrhs, G + TC(mySize - klu, 0, mySize), &mySize, Gr[p] + TC(klu, 0, ldgr), &ldgr);
            DDeallocate_1D(G);
        }
    }
    else
    {
        if (pivot)
        {
            startAiU = klu;
            startAiL = klu - ku;
            //! ???
            for (int j = 0; j < nrhs / 2; j++)
            {
                incx = 1, incy = -1;
                dswap_(&mySize, localF + TC(0, j, ldf), &incx, localF + TC(0, nrhs - j - 1, ldf), &incy);
            }

            if (nrhs % 2 == 1)
            {
                _mySize = mySize / 2, incx = 1, incy = -1;
                dswap_(&_mySize, localF + TC(0, nrhs / 2, ldf), &incx, localF + TC(0 + (mySize + 1) / 2, nrhs / 2, ldf), &incy);
            }
            //! Solve Un*Ln*Gn=Fn
            trans = 'N';
            dgbtrsl_(&trans, &mySize, &ku, &kl, &nrhs, localA + TC(startAiL, 0, lda), &lda, ipiv + partitionsSize[p], localF, &ldf, &info);
            //! Get Un*Grn
            uplo = 'F';
            dlacpy_(&uplo, &klu, &nrhs, localF + TC(mySize - klu, 0, ldf), &ldf, Gr[p], &ldgr);

            trans = 'N';
            dgbtrsu_(&trans, &klu, &ku, &kl, &nrhs, localA + TC(startAiL, mySize - klu, lda), &lda, ipiv, Gr[p], &ldgr, &info);

            for (int j = 0; j < nrhs / 2; j++)
            {
                incx = 1, incy = -1;
                dswap_(&klu, Gr[p] + TC(0, j, ldgr), &incx, Gr[p] + TC(0, nrhs - j - 1, ldgr), &incy);
            }

            if (nrhs % 2 == 1)
            {
                _n = klu / 2, incx = 1, incy = -1;
                dswap_(&_n, Gr[p] + TC(0, nrhs / 2, ldgr), &incx, Gr[p] + TC((klu + 1) / 2, nrhs / 2, ldgr), &incy);
            }
        }
        else
        {
            //! Solve Un*Ln*Gn=Fn
            uplo = 'U', trans = 'N', diag = 'U';
            dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &ku, localA + TC(startAiU, 0, lda), &lda, localF, &ldf);

            uplo = 'F';
            dlacpy_(&uplo, &klu, &nrhs, localF, &ldf, Gr[p], &ldgr);
            //! Solve Ln*Gn=inv(Un)*Fn(half sweep) and get Grn{t}
            uplo = 'L', trans = 'N', diag = 'N';
            dtbsm_(&uplo, &trans, &diag, &klu, &nrhs, &kl, localA + TC(startAiL, 0, lda), &lda, Gr[p], &ldgr);
        }
    }
    //! Until now,F1 and Fn have been changed,F1=inv(L1)*F1=U1*G1,Fn=inv(Un)*Fn=Ln*Gn

#ifdef STAGE_TIME
    stageTime = MPI_Wtime() - stageTime;
    stageTimes[0] = stageTime;
    if (myId != 0)
    {
        MPI_Send(&stageTime, 1, MPI_DOUBLE, 0, 100 + myId, mpiInfo->comm);
    }
    else
    {
        for (int i = 1; i < nPart; i++)
        {
            MPI_Recv(&stageTimes[i], 1, MPI_DOUBLE, i, 100 + i, mpiInfo->comm, &status);
        }
    }
    if (myId == 0)
    {
        for (int i = 0; i < nPart; i++)
        {
            std::cout << "Process " << i << ",time of DY factorization stage: " << stageTimes[i] << std::endl;
        }
    }
#endif

#ifdef COMM_TIME
    commTime = MPI_Wtime();
#endif

    for (int i = 0; i < nPart; i++)
    {
        MPI_Bcast(Gr[i], 2 * klu * nrhs, MPI_DOUBLE, i, mpiInfo->comm);
    }

#ifdef COMM_TIME
    commTime = MPI_Wtime() - commTime;
    MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
    if (myId == 0)
    {
        mpiInfo->commTime["Time of boardcasting matrix Gr1"] = commMaxTime;
    }
#endif

    /* 
        ! Solve D[1]*D[2]*D[nLevel-1]*Gr[nLevel]=Gr[1]
     */

#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif
    SolveReducedSys(kl, ku, nrhs, V, W, R, nPart, nLevel, Gr, mpiInfo);
#ifdef STAGE_TIME
    stageTime = MPI_Wtime() - stageTime;
    stageTimes[0] = stageTime;
    if (myId != 0)
    {
        MPI_Send(&stageTime, 1, MPI_DOUBLE, 0, 100 + myId, mpiInfo->comm);
    }
    else
    {
        for (int i = 1; i < nPart; i++)
        {
            MPI_Recv(&stageTimes[i], 1, MPI_DOUBLE, i, 100 + i, mpiInfo->comm, &status);
        }
    }
    if (myId == 0)
    {
        for (int i = 0; i < nPart; i++)
        {
            std::cout << "Process " << i << ",time of recursive solve stage: " << stageTimes[i] << std::endl;
        }
    }
#endif
    //! Until now,we got the top and bottom of solution x

    /* *************************************
       *           Retrieval Stage         *
       *get the midddle part of solution x *
       ************************************* */

#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif
    if (p == 0)
    {
        if (pivot)
        {
            startBi = klu;
            startBj = 0;
            startAiL = klu - kl;
        }
        else
        {
            startBi = 0;
        }
        //! Get -B1*G2{t}
        side = 'L', uplo = 'L', transa = 'N', diag = 'N', alpha = -1, _lda = lda - 1;
        dtrmm_(&side, &uplo, &transa, &diag, &ku, &nrhs, &alpha, localA + TC(startBi, 0, lda), &_lda, Gr[p + 1], &ldgr);

        if (pivot)
        {
            ipivAux = IAllocate_1D(2 * klu, &allocInfo);
            int idx = 0;
            for (int i = partitionsSize[p + 1] - 2 * klu; i < partitionsSize[p + 1]; i++)
                ipivAux[idx++] = ipiv[i] - (mySize - 2 * klu);
            aux = DZAllocate_1D(2 * klu * nrhs, &allocInfo);

            uplo = 'F';
            dlacpy_(&uplo, &ku, &nrhs, Gr[p + 1], &ldgr, aux + TC(2 * klu - ku, 0, ldgr), &ldgr);
            trans = 'N';
            dgbtrsl_(&trans, &ldgr, &kl, &ku, &nrhs, localA + TC(startAiL, mySize - 2 * klu, lda), &lda, ipivAux, aux, &ldgr, &info);

            for (int i = 0; i < nrhs; i++)
            {
                alpha = 1.0, incx = 1, incy = 1;
                daxpy_(&ldgr, &alpha, aux + TC(0, i, ldgr), &incx, localF + TC(mySize - 2 * klu, i, ldf), &incy);
            }

            IDeallocate_1D(ipivAux);
            DDeallocate_1D(aux);
        }
        else
        {
            //! Solve L1*U1*X'=-B1*G2{t}(half sweep)
            uplo = 'L', trans = 'N', diag = 'U';
            dtbsm_(&uplo, &trans, &diag, &ku, &nrhs, &kl, localA + TC(ku, mySize - ku, lda), &lda, Gr[p + 1], &ldgr);
            for (int i = 0; i < nrhs; i++)
            {
                //! Get inv(L1)*(F1-B1*G2{t})
                alpha = 1.0, incx = 1, incy = 1;
                daxpy_(&ku, &alpha, Gr[p + 1] + TC(0, i, ldgr), &incx, localF + TC(mySize - ku, i, ldf), &incy);
            }
        }
    }

    if (p > 0 && p < nPart - 1)
    {
        if (pivot)
        {
            startBi = klu;
            startCi = klu + ku + kl;
            startBj = 0;

            if (p > 0)
            {
                startCj = mySize - kl;
                inc = 1;
                startFi = mySize - ku;
                startGri = 0;
            }
            else
            {
                startCj = mySize / 2 - kl;
                startFi = mySize / 2;
                inc = -1;
                startGri = nrhs;
            }
        }
        else
        {
            inc = 1;
            startBi = 0;
            startBj = 0;
            startCi = kl + ku;
            startCj = mySize - kl;
            startFi = mySize - ku;
            startGri = 0;
        }

        //! Get -Bp*Gp+1{t}
        side = 'L', uplo = 'L', transa = 'N', diag = 'N', alpha = -1, _lda = lda - 1;
        dtrmm_(&side, &uplo, &transa, &diag, &ku, &nrhs, &alpha, localA + TC(startBi, startBj, lda), &_lda, Gr[p + 1], &ldgr);
        //! Get -Cp*Gp-1{b}
        side = 'L', uplo = 'U', transa = 'N', diag = 'N', alpha = -1, _lda = lda - 1;
        dtrmm_(&side, &uplo, &transa, &diag, &kl, &nrhs, &alpha, localA + TC(startCi, startCj, lda), &_lda, Gr[p - 1] + 2 * klu - kl, &ldgr);

        for (int i = 0; i < nrhs; i++)
        {
            //! Get Fp-Cp*Gp-1{b}
            alpha = 1.0, incx = 1, incy = 1;
            daxpy_(&kl, &alpha, Gr[p - 1] + TC(2 * klu - kl, i, ldgr), &incx, localF + TC(0, i, ldf), &incy);
            //! Get Fp-Bp*Gp+1{t}-Cp*Gp-1{b}
            alpha = 1.0, incx = 1, incy = 1;
            daxpy_(&ku, &alpha, Gr[p + 1] + TC(0, startGri + i * inc, ldgr), &incx, localF + TC(startFi, i, ldf), &incy);
        }
    }

    if (p == nPart - 1)
    {
        if (pivot)
        {
            startCi = klu;
            startCj = 0;
            startAiL = klu - ku;

            ipivAux = IAllocate_1D(2 * klu, &allocInfo);
            int idx = 0;
            for (int i = partitionsSize[p + 1] - 2 * klu; i < partitionsSize[p + 1]; i++)
                ipivAux[idx++] = ipiv[i] - (mySize - 2 * klu);

            aux = DZAllocate_1D(2 * klu * nrhs, &allocInfo);
            for (int j = 0; j < nrhs / 2; j++)
            {
                incx = 1, incy = -1;
                dswap_(&kl, Gr[p - 1] + TC(2 * klu - kl, j, ldgr), &incx, Gr[p - 1] + TC(2 * klu - kl, nrhs - j - 1, ldgr), &incy);
            }

            if (nrhs % 2 == 1)
            {
                int _kl = kl / 2, incx = 1, incy = -1;
                dswap_(&_kl, Gr[p - 1] + TC(2 * klu - kl, nrhs / 2, ldgr), &incx, Gr[p - 1] + TC(2 * klu - kl + (kl + 1) / 2, nrhs / 2, ldgr), &incy);
            }
            //! Get -Cn*Gn-1{b} ???Why is uplo 'L'? In band storage,Ci is shape of 'L'?
            side = 'L', uplo = 'L', transa = 'N', diag = 'N', alpha = -1, _lda = lda - 1;
            dtrmm_(&side, &uplo, &transa, &diag, &kl, &nrhs, &alpha, localA + TC(startCi, startCj, lda), &_lda, Gr[p - 1] + TC(2 * klu - kl, 0, ldgr), &ldgr);

            uplo = 'F';
            dlacpy_(&uplo, &kl, &nrhs, Gr[p - 1] + TC(2 * klu - kl, 0, ldgr), &ldgr, aux + TC(2 * klu - kl, 0, ldgr), &ldgr);

            trans = 'N';
            dgbtrsl_(&trans, &ldgr, &ku, &kl, &nrhs, localA + TC(startAiL, mySize - 2 * klu, lda), &lda, ipivAux, aux, &ldgr, &info);

            for (int i = 0; i < nrhs; i++)
            {
                alpha = 1.0, incx = 1, incy = 1;
                daxpy_(&ldgr, &alpha, aux + TC(0, i, ldgr), &incx, localF + TC(mySize - 2 * klu, i, ldf), &incy);
            }

            IDeallocate_1D(ipivAux);
            DDeallocate_1D(aux);
        }
        else
        {
            //! Get -Cn*Gn-1{b}
            side = 'L', uplo = 'U', transa = 'N', diag = 'N', alpha = -1, _lda = lda - 1;
            dtrmm_(&side, &uplo, &transa, &diag, &kl, &nrhs, &alpha, localA + TC(kl + ku, mySize - kl, lda), &_lda, Gr[p - 1] + TC(2 * klu - kl, 0, p - 1), &ldgr);
            //! Solve Un*Ln*X=-Cn*Gn-1{b}
            uplo = 'U', trans = 'N', diag = 'U';
            dtbsm_(&uplo, &trans, &diag, &kl, &nrhs, &ku, localA + TC(0, 0, lda), &lda, Gr[p - 1] + TC(2 * klu - kl, 0, ldgr), &ldgr);

            for (int i = 0; i < nrhs; i++)
            {
                //! Get inv(Un)*(Fn-Cn*Gn-1{b})
                alpha = 1.0, incx = 1, incy = 1;
                daxpy_(&kl, &alpha, Gr[p - 1] + TC(2 * klu - kl, 0, ldgr), &incx, localF + TC(0, i, ldf), &incy);
            }
        }
    }
#ifdef STAGE_TIME
    stageTime = MPI_Wtime() - stageTime;
    stageTimes[0] = stageTime;
    if (myId != 0)
    {
        MPI_Send(&stageTime, 1, MPI_DOUBLE, 0, 100 + myId, mpiInfo->comm);
    }
    else
    {
        for (int i = 1; i < nPart; i++)
        {
            MPI_Recv(&stageTimes[i], 1, MPI_DOUBLE, i, 100 + i, mpiInfo->comm, &status);
        }
    }
    if (myId == 0)
    {
        for (int i = 0; i < nPart; i++)
        {
            std::cout << "Process " << i << ",time of retrieval stage 1: " << stageTimes[i] << std::endl;
        }
    }
#endif

#ifdef COMM_TIME
    commTime = MPI_Wtime();
#endif
    for (int i = 0; i < nPart; i++)
    {
        for (int j = 0; j < nrhs; j++)
        {
            MPI_Bcast(Gr[1] + ldgr * j, klu, MPI_DOUBLE, 0, mpiInfo->comm);
        }
    }
    for (int i = 1; i < nPart - 1; i++)
    {
        for (int j = 0; j < nrhs; j++)
        {
            MPI_Bcast(Gr[i - 1] + ldgr * j + klu, klu, MPI_DOUBLE, i, mpiInfo->comm);
            MPI_Bcast(Gr[i + 1] + ldgr * j, klu, MPI_DOUBLE, i, mpiInfo->comm);
        }
    }
    for (int i = 0; i < nPart; i++)
    {
        for (int j = 0; j < nrhs; j++)
        {
            MPI_Bcast(Gr[nPart - 2] + ldgr * j + klu, klu, MPI_DOUBLE, nPart - 1, mpiInfo->comm);
        }
    }
#ifdef COMM_TIME
    commTime = MPI_Wtime() - commTime;
    MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
    if (myId == 0)
    {
        mpiInfo->commTime["Time of boardcasting matrix Gr"] = commMaxTime;
    }
#endif

#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif
    if (p == 0)
    {
        if (pivot)
        {
            //! Solve U1*X1=inv(L1)*(F1-B1*G2{t})
            trans = 'N';
            dgbtrsu_(&trans, &mySize, &kl, &ku, &nrhs, localA + TC(klu - kl, 0, lda), &lda, ipiv + partitionsSize[p], localF, &ldf, &info);
        }
        else
        {
            //! Solve U1*X1=inv(L1)*(F1-B1*G2{t})
            uplo = 'U', trans = 'N', diag = 'N';
            dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &ku, localA + TC(startAiU, 0, lda), &lda, localF, &ldf);
        }
    }
    else if (p == nPart - 1)
    {
        if (pivot)
        {
            //! Solve Lp*Xp=inv(Up)*(Fp-Cp*Gp-1{b})
            trans = 'N';
            dgbtrsu_(&trans, &mySize, &ku, &kl, &nrhs, localA + TC(klu - ku, 0, lda), &lda, ipiv + partitionsSize[p], localF, &ldf, &info);

            for (int j = 0; j < nrhs / 2; j++)
            {
                incx = 1, incy = -1;
                dswap_(&mySize, localF + TC(0, j, ldf), &incx, localF + TC(0, nrhs - j - 1, ldf), &incy);
            }

            if (nrhs % 2 == 1)
            {
                _mySize = mySize / 2, incx = 1, incy = -1;
                dswap_(&_mySize, localF + TC(0, nrhs / 2, ldf), &incx, localF + TC(0 + (mySize + 1) / 2, nrhs / 2, ldf), &incy);
            }
        }
        else
        {
            //! Solve Lp*Xp=inv(Up)*(Fp-Cp*Gp-1{b})
            uplo = 'L', trans = 'N', diag = 'N';
            dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &ku, localA + TC(ku, 0, lda), &lda, localF, &ldf);
        }
    }
    else
    {
        if (pivot)
        {
            trans = 'N';
            dgbtrs_(&trans, &mySize, &kl, &ku, &nrhs, localA + TC(klu - kl, 0, lda), &lda, ipiv + partitionsSize[p], localF, &ldf, &info);
        }
        else
        {
            //! Solve Lp*Up*Xp=Fp-Bp*Gp+1{t}-Cp*Gp-1{b}
            uplo = 'L', trans = 'N', diag = 'U';
            dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &kl, localA + TC(ku, 0, lda), &lda, localF, &ldf);
            //! Solve Up*Xp=inv(Lp)*(Fp-Bp*Gp+1{t}-Cp*Gp-1{b})
            uplo = 'U', trans = 'N', diag = 'N';
            dtbsm_(&uplo, &trans, &diag, &mySize, &nrhs, &ku, localA + TC(0, 0, lda), &lda, localF, &ldf);
        }
    }
#ifdef STAGE_TIME
    stageTime = MPI_Wtime() - stageTime;
    stageTimes[0] = stageTime;
    if (myId != 0)
    {
        MPI_Send(&stageTime, 1, MPI_DOUBLE, 0, 100 + myId, mpiInfo->comm);
    }
    else
    {
        for (int i = 1; i < nPart; i++)
        {
            MPI_Recv(&stageTimes[i], 1, MPI_DOUBLE, i, 100 + i, mpiInfo->comm, &status);
        }
    }
    if (myId == 0)
    {
        for (int i = 0; i < nPart; i++)
        {
            std::cout << "Process " << i << ",time of retrieval stage 2: " << stageTimes[i] << std::endl;
        }
    }
#endif

    IDeallocate_1D(partitionsSize);
    DDeallocate_2D(Gr, nPart);
}