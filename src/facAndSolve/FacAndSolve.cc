#include "facandsolve.h"

void FacAndSolve(BandedMatrix *A, double *localF, int nrhs, double ***V, double ***W, double **R,
                 int *ipiv, Options_t *options, MpiInfo_t *mpiInfo)
{
    int myId, nPart, nLevel,
        p,
        n, kl, ku, klu, kd, lda,
        ldv, ldw, ldgr, ldf,
        mySize,
        _lda, _nrhs, _mySize, _n,
        info, allocInfo, max_info,
        incx, incy, inc,
        startAiL, startAiU,
        startBi, startBj,
        startCi, startCj,
        startFi, startGri;
    int *partitionsSize, *ipivAux;
    YesOrNo_t pivot;
    double norma,
        nzero,
        alpha;
    double *localA, *localAT,
        *normWork,
        *BCi,
        **Gr,
        *G,
        *aux;
    char norm, uplo, trans, diag, side, transa;
    MPI_Status status;

#ifdef COMM_TIME
    double commTime, commMaxTime;
#endif

#ifdef STAGE_TIME
    double stageTime, stageTimes[mpiInfo->nProcess];
#endif

#ifdef DETAIL_TIME
    double detailTime;
#endif

    /* ************************
      * Preprocessing Stage！* 
      ************************ */

    pivot = options->IsPivot;
    nzero = NZERO;
    n = A->n;
    kl = A->kl;
    ku = A->ku;
    klu = std::max(kl, ku);
    if (options->IsPivot)
    {
        lda = klu + kl + ku + 1;
    }
    else
    {
        lda = kl + ku + 1;
    }

    ldv = 2 * klu, ldw = 2 * klu;

    myId = mpiInfo->myId;
    nLevel = options->LevelsCnt;
    nPart = options->PartitionCnt;
    localA = A->storageLoc;

    partitionsSize = IAllocate_1D(nPart + 1, &allocInfo);
    if (options->IsRatioEqual)
    {
        CaculEqPartitionsSize(partitionsSize, n, options);
    }
    else
    {
        CaculPartitionsSize(partitionsSize, n, options);
    }
    p = myId;
    info = 0, max_info = 0;
    mySize = partitionsSize[p + 1] - partitionsSize[p];

    localAT = DAllocate_1D(mySize * lda, &allocInfo);
    uplo = 'F';
    dlacpy_(&uplo, &lda, &mySize, localA, &lda, localAT, &lda);

    if (pivot)
    {
        startBi = klu;
        startCi = klu + kl + ku;
    }
    else
    {
        startBi = 0;
        startCi = kl + ku;
    }

#ifdef COMM_TIME
    commTime = MPI_Wtime();
#endif
    /* 
        !Copy Bp+1 to Bp,copy Cp to Cp+1!!! 
    */
    for (int i = 0; i < nPart; i++)
    {
        if (p == i + 1)
        {
            for (int j = 0; j < klu; j++)
            {
                MPI_Send(localAT + TC(startBi, j, lda), klu - j, MPI_DOUBLE, p - 1, 100 + p - 1, mpiInfo->comm);
            }
        }
        else if (p == i && p != nPart - 1)
        {
            for (int j = 0; j < klu; j++)
            {
                MPI_Recv(localA + TC(startBi, j, lda), klu - j, MPI_DOUBLE, p + 1, 100 + p, mpiInfo->comm, &status);
            }
        }
        if (p == i && i < nPart - 1)
        {
            for (int j = 0; j < klu; j++)
            {
                MPI_Send(localAT + TC(startCi - j, partitionsSize[p + 1] - partitionsSize[p] - (klu - j), lda), j + 1, MPI_DOUBLE, p + 1, 100 + p + 1, mpiInfo->comm);
            }
        }
        else if (p == i + 1 && p != 0)
        {
            for (int j = 0; j < klu; j++)
            {
                MPI_Recv(localA + TC(startCi - j, partitionsSize[p + 1] - partitionsSize[p] - (klu - j), lda), j + 1, MPI_DOUBLE, p - 1, 100 + p, mpiInfo->comm, &status);
            }
        }
    }
    DDeallocate_1D(localAT);
#ifdef COMM_TIME
    commTime = MPI_Wtime() - commTime;
    MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
    if (myId == 0)
    {
        mpiInfo->commTime["Time of moving matrix B and C"] = commMaxTime;
    }
#endif

    /* 
        !Get norm of Ap 
    */
    if (pivot)
    {
        ipivAux = IAllocate_1D(2 * klu, &allocInfo);
        startCi = klu + kl + ku;
        startAiL = klu - kl;
    }
    else
    {
        if (options->Norm == INFININORM)
        {
            norm = 'f';
            normWork = DAllocate_1D(partitionsSize[p + 1] - partitionsSize[p], &allocInfo);
        }
        else
        {
            norm = 'o';
            normWork = DAllocate_1D(1, &allocInfo);
        }
        norma = dlangb_(&norm, &mySize, &kl, &ku, localA + TC(0, 0, lda), &lda, normWork);
        DDeallocate_1D(normWork);
        startBi = 0;
        startCi = kl + ku;
        startAiL = ku;
        startAiU = 0;
        kd = ku;
    }

    /* ************************
       *DS Factorization Stage*
       *Solve DS = A          *   
       ************************ */

    /* 
        !LU factorization of A1 and get V1{bot,1}(half sweep)
    */
#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif
    info = 0, max_info = 0;
    if (p == 0)
    {
        if (pivot)
        {
            startAiU = klu - kl;
            kd = kl + ku;

#ifdef DETAIL_TIME
            detailTime = MPI_Wtime();
#endif
            //! LU factorize A1
            dgbtrf_(&mySize, &mySize, &kl, &ku, localA + TC(startAiL, 0, lda), &lda, ipiv + partitionsSize[p], &info);
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime() - detailTime;
            std::cout << "Time of LU factorization for A in process " << p << std::endl;
#endif

            int idx = 0;
            for (int i = partitionsSize[p + 1] - 2 * klu; i < partitionsSize[p + 1]; i++)
                ipivAux[idx++] = ipiv[i] - (mySize - 2 * klu);

            uplo = 'L', _lda = lda - 1;
            dlacpy_(&uplo, &ku, &ku, localA + TC(startBi, 0, lda), &_lda, V[0][p] + 2 * klu - ku, &ldv);
            //! Solve L1*U1*V1=B1
            trans = 'N';
            dgbtrsl_(&trans, &ldv, &kl, &ku, &klu, localA + TC(startAiL, mySize - 2 * klu, lda), &lda, ipivAux, V[0][p], &ldv, &info);
        }
        else
        {
            uplo = 'L', _lda = lda - 1;
            dlacpy_(&uplo, &ku, &ku, localA + TC(startBi, 0, lda), &_lda, V[0][p] + 2 * klu - ku, &ldv);
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime();
#endif
            //! LU factorize A1
            dgbalu_(&mySize, &kl, &ku, localA + TC(0, 0, lda), &lda, &nzero, &norma, &info);
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime() - detailTime;
            std::cout << "Time of LU factorization for A in process " << p << std::endl;
#endif
            //! Solve L1*U1*V1=B1
            uplo = 'L', trans = 'N', diag = 'U';
            dtbsm_(&uplo, &trans, &diag, &klu, &ku, &kl, localA + TC(startAiL, mySize - klu, lda), &lda, V[0][p] + klu, &ldv);
        }
        // !Get the copy of U1*V1{b,0}
        uplo = 'L';
        dlacpy_(&uplo, &klu, &klu, V[0][p] + 2 * klu - klu, &ldv, V[0][p], &ldv);
        // !Solve U1*V1=inv(L1)*B1
        uplo = 'U', trans = 'N', diag = 'N';
        dtbsm_(&uplo, &trans, &diag, &klu, &ku, &kd, localA + TC(startAiU, mySize - klu, lda), &lda, V[0][p] + klu, &ldv);
    }
    /* 
        ! LU factorization of An and get Wn{top,1}(half sweep)
    */
    else if (p == nPart - 1)
    {
        startAiL = klu - ku;
        if (pivot == YES)
        {
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime();
#endif
            //! UL factorize An
            dgbtrful_(&mySize, &mySize, &kl, &ku, localA + TC(0, 0, lda), &lda, ipiv + partitionsSize[p], &info);
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime() - detailTime;
            std::cout << "Time of UL factorization for A in process " << p << std::endl;
#endif

            int idx = 0;
            for (int i = partitionsSize[p + 1] - 2 * klu; i < partitionsSize[p + 1]; i++)
                ipivAux[idx++] = ipiv[i] - (mySize - 2 * klu);

            uplo = 'L', _lda = lda - 1;
            dlacpy_(&uplo, &kl, &kl, localA + TC(startBi, 0, lda), &_lda, W[0][p] + 2 * klu - kl, &ldw);
            //! Solve Un*Ln*Wn=Cn
            trans = 'N';
            dgbtrsl_(&trans, &ldw, &ku, &kl, &klu, localA + TC(startAiL, mySize - 2 * klu, lda), &lda, ipivAux, W[0][p], &ldw, &info);

            uplo = 'L';
            dlacpy_(&uplo, &kl, &kl, W[0][p] + 2 * klu - kl, &ldw, W[0][p] + klu - kl, &ldw);
            //! Solve Ln*Wn=inv(Un)*Cn
            trans = 'N';
            dgbtrsu_(&trans, &klu, &ku, &kl, &klu, localA + TC(startAiL, mySize - klu, lda), &lda, ipivAux, W[0][p] + klu, &ldw, &info);
            //! ???
            for (int j = 0; j < klu / 2; j++)
            {
                incx = 1, incy = -1;
                dswap_(&ldw, W[0][p] + TC(0, j, ldw), &incx, W[0][p] + TC(0, klu - j - 1, ldw), &incy);
            }
            if (klu % 2 == 1)
            {
                incx = 1, incy = -1;
                dswap_(&klu, W[0][p] + TC(0, klu / 2, ldw), &incx, W[0][p] + TC(klu, klu / 2, ldw), &incy);
            }
        }
        else
        {
            uplo = 'U', _lda = lda - 1;
            dlacpy_(&uplo, &kl, &kl, localA + TC(kl + ku, mySize - kl, lda), &_lda, W[0][p] + TC(0, klu - kl, ldw), &ldw);

#ifdef DETAIL_TIME
            detailTime = MPI_Wtime();
#endif
            //! UL factorize An
            dgbaul_(&mySize, &kl, &ku, localA + TC(0, 0, lda), &lda, &nzero, &norma, &info);
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime() - detailTime;
            if (myId == 1)
            {
                std::cout << "Time of LU factorization for A in process " << p << std::endl;
            }
#endif

            //! Solve Un*Ln*Wn=Cn
            uplo = 'U', trans = 'N', diag = 'U';
            dtbsm_(&uplo, &trans, &diag, &klu, &kl, &ku, localA + TC(0, 0, lda), &lda, W[0][p] + TC(0, klu - kl, ldw), &ldw);
            //! Get a copy of Ln*Wn{t,0}
            uplo = 'U';
            dlacpy_(&uplo, &klu, &klu, W[0][p] + TC(0, klu - kl, ldw), &ldw, W[0][p] + TC(klu, klu - kl, ldw), &ldw);
            //! Solve Ln*Wn=inv(Un)*Cn
            uplo = 'L', trans = 'N', diag = 'N';
            dtbsm_(&uplo, &trans, &diag, &klu, &kl, &kl, localA + TC(ku, 0, lda), &lda, W[0][p] + TC(0, klu - kl, ldw), &ldw);
        }
    }
    /* 
        ! LU factorization of Ai and get Vi{top&bot,1}(half sweep),Wi{top&bot,1}(full sweep) 
    */
    else
    {
        BCi = DZAllocate_1D(mySize * (ku + kl), &allocInfo);
        if (pivot)
        {
            startCi = klu + kl + ku;

#ifdef DETAIL_TIME
            detailTime = MPI_Wtime();
#endif
            //! LU factorize Ap with pivotiong
            dgbtrf_(&mySize, &mySize, &kl, &ku, localA + TC(startAiL, 0, lda), &lda, ipiv + partitionsSize[p], &info);
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime() - detailTime;
            if (myId == 1)
            {
                std::cout << "Time of LU factorization for A in process " << p << std::endl;
            }
#endif

            //! Copy Bi
            uplo = 'L', _lda = lda - 1;
            dlacpy_(&uplo, &ku, &ku, localA + TC(startBi, 0, lda), &_lda, BCi + TC(mySize - ku, 0, mySize), &mySize);
            //! Copy Ci
            uplo = 'U', _lda = lda - 1;
            dlacpy_(&uplo, &kl, &kl, localA + TC(startCi, mySize - kl, lda), &_lda, BCi + TC(0, ku, mySize), &mySize);

            int idx = 0;
            for (int i = partitionsSize[p + 1] - 2 * klu; i < partitionsSize[p + 1]; i++)
                ipivAux[idx++] = ipiv[i] - (mySize - 2 * klu);

            //! Solve Lp*Up*Vp=Bp(half sweep)
            trans = 'N';
            dgbtrsl_(&trans, &ldv, &kl, &ku, &ku, localA + TC(startAiL, mySize - 2 * klu, lda), &lda, ipivAux, BCi + TC(mySize - 2 * klu, 0, mySize), &mySize, &info);
            //! Solve Lp*Up*Wp=Cp(full sweep)
            trans = 'N';
            dgbtrsl_(&trans, &mySize, &kl, &ku, &kl, localA + TC(startAiL, 0, lda), &lda, ipiv + partitionsSize[p], BCi + TC(0, ku, mySize), &mySize, &info);
            //! Solve Up*[Vp,Wp]=inv(Lp)*[Bp,Cp]
            trans = 'N', _nrhs = kl + ku;
            dgbtrsu_(&trans, &mySize, &kl, &ku, &_nrhs, localA + TC(klu - kl, 0, lda), &lda, ipiv + partitionsSize[p], BCi, &mySize, &info);
            IDeallocate_1D(ipivAux);
        }
        else
        {
            //! LU factorize Ap
            dgbalu_(&mySize, &kl, &ku, localA + TC(0, 0, lda), &lda, &nzero, &norma, &info);
            //! Copy Bi
            uplo = 'L', _lda = lda - 1;
            dlacpy_(&uplo, &ku, &ku, localA + TC(0, 0, lda), &_lda, BCi + TC(mySize - ku, 0, mySize), &mySize);
            //! Copy Ci
            uplo = 'U', _lda = lda - 1;
            dlacpy_(&uplo, &kl, &kl, localA + TC(kl + ku, mySize - kl, lda), &_lda, BCi + TC(0, ku, mySize), &mySize);
            //! Solve Lp*Up*Vp=Bp(half sweep)
            uplo = 'L', trans = 'N', diag = 'U';
            dtbsm_(&uplo, &trans, &diag, &klu, &ku, &kl, localA + TC(ku, mySize - klu, lda), &lda, BCi + TC(mySize - klu, 0, mySize), &mySize);
            //! Solve Lp*Up*Wp=Cp(full sweep)
            uplo = 'L', trans = 'N', diag = 'U', startAiL = ku;
            dtbsm_(&uplo, &trans, &diag, &mySize, &kl, &kl, localA + TC(ku, 0, lda), &lda, BCi + TC(0, ku, mySize), &mySize);
            //! Solve Up*[Vp,Wp]=inv(Lp)*[Bp,Cp]
            uplo = 'U', trans = 'N', diag = 'N', _nrhs = ku + kl;
            dtbsm_(&uplo, &trans, &diag, &mySize, &_nrhs, &ku, localA + TC(0, 0, lda), &lda, BCi, &mySize);
        }
        //! Get Vp and Wp
        uplo = 'F';
        dlacpy_(&uplo, &klu, &ku, BCi, &mySize, V[0][p], &ldv);
        dlacpy_(&uplo, &klu, &kl, BCi + mySize * ku, &mySize, W[0][p], &ldw);
        dlacpy_(&uplo, &klu, &ku, BCi + mySize - klu, &mySize, V[0][p] + klu, &ldv);
        dlacpy_(&uplo, &klu, &kl, BCi + mySize * (ku + 1) - klu, &mySize, W[0][p] + klu, &ldw);
        DDeallocate_1D(BCi);
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
            std::cout << "Process " << i << ",time of DS factorization stage: " << stageTimes[i] << std::endl;
        }
    }
#endif

    if (!pivot)
    {
        if (info == 0)
        {
            if (myId == 0)
            {
                std::cout << "DGBALU run successfully!" << std::endl;
            }
        }
        else if (info == 1)
        {
            ABORT("DGBALU had to boost!", mpiInfo->comm);
        }
        else
        {
            ABORT("DGBALU found an error/illegal value!", mpiInfo->comm);
        }
    }

    max_info = std::max(info, max_info);
    info = max_info;

#ifdef COMM_TIME
    commTime = MPI_Wtime();
#endif
    for (int i = 0; i < nPart; i++)
    {
        MPI_Bcast(V[0][i], 2 * klu * klu, MPI_DOUBLE, i, mpiInfo->comm);
        MPI_Bcast(W[0][i], 2 * klu * klu, MPI_DOUBLE, i, mpiInfo->comm);
    }
#ifdef COMM_TIME
    commTime = MPI_Wtime() - commTime;
    MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
    if (myId == 0)
    {
        mpiInfo->commTime["Time of boardcast matrix V and W"] = commMaxTime;
    }
#endif

    //! Get reduced system of 1st level
    if (p != nPart - 1)
    {
        GetReducedSys(klu, V[0][p], W[0][p + 1], R[p]);
    }

#ifdef COMM_TIME
    commTime = MPI_Wtime();
#endif

    for (int i = 0; i < nPart; i++)
    {
        MPI_Bcast(R[i], 2 * klu * 2 * klu, MPI_DOUBLE, i, mpiInfo->comm);
        if (options->IsPivot)
        {
            MPI_Bcast(ipiv + partitionsSize[i], partitionsSize[i + 1] - partitionsSize[i], MPI_INT, i, mpiInfo->comm);
        }
    }

#ifdef COMM_TIME
    commTime = MPI_Wtime() - commTime;
    MPI_Reduce(&commTime, &commMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiInfo->comm);
    if (myId == 0)
    {
        mpiInfo->commTime["Time of boardcasting reduced system R and vector ipiv"] = commMaxTime;
    }
#endif

    Gr = DZAllocate_2D(nPart, 2 * klu * nrhs, &allocInfo);
    ldgr = 2 * klu;
    ldf = mySize;

    if (pivot)
    {
        startAiU = klu - kl;
        kd = kl + ku;
    }
    else
    {
        startAiL = ku;
        startAiU = 0;
        kd = ku;
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
            G = DZAllocate_1D(mySize * nrhs, &allocInfo);
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

    //! Until now,F1 and Fn have been changed,F1=inv(L1)*F1=U1*G1,Fn=inv(Un)*Fn=Ln*Gn

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

    /* ******************
       * Recursive Stage*
       ****************** */

#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif
    RecursiveStage(kl, ku, nrhs, V, W, R, Gr, nPart, nLevel, mpiInfo);
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
            std::cout << "Process " << i << ",time of recursive stage: " << stageTimes[i] << std::endl;
        }
    }
#endif

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
    DDeallocate_2D(Gr, nPart);
    IDeallocate_1D(partitionsSize);
}
