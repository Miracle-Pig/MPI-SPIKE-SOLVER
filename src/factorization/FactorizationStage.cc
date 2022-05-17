#include "factorization.h"
#include "macro.h"

#include <iostream>
#include <string.h>

void FactorizationStage(BandedMatrix *A, double ***V, double ***W, double **R,
                        int *ipiv, Options_t *options, MpiInfo_t *mpiInfo)
{
    int n, kl, ku, klu, kd, lda,
        p,
        _lda, ldv, ldw, _nrhs,
        incx, incy,
        myId, nPart, nLevel,
        info, allocInfo, max_info,
        mySize, startAiL, startAiU,
        startBi, startCi;
    int *partitionsSize, *ipivAux;
    YesOrNo_t pivot;
    double norma, nzero;
    double *normWork,
        *localA, *localAT,
        *BCi;
    char norm, uplo, trans, diag;
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

#ifdef INFO_PRINT
    if (myId == 0)
    {
        std::cout << "分区大小: ";
        for (int i = 0; i < nPart; i++)
        {
            std::cout << partitionsSize[i + 1] - partitionsSize[i] << "\t";
        }
        std::cout << std::endl;
    }
#endif

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
    // ! Copy Bp+1 to Bp,copy Cp to Cp+1!!!
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

    /* **************************
       * DS Factorization Stage *
       * Solve DS = A           *   
       ************************** */

    /* 
        ! LU factorization of A1 and get V1{bot,1}(half sweep)
    */
#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif
    info = 0;
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
        //! Get the copy of U1*V1{b,0}
        uplo = 'L';
        dlacpy_(&uplo, &klu, &klu, V[0][p] + 2 * klu - klu, &ldv, V[0][p], &ldv);
        //! Solve U1*V1=inv(L1)*B1
        uplo = 'U', trans = 'N', diag = 'N';
        dtbsm_(&uplo, &trans, &diag, &klu, &ku, &kd, localA + TC(startAiU, mySize - klu, lda), &lda, V[0][p] + klu, &ldv);
    }
    /* 
        ! LU factorization of Anpart and get W1{top,1}(half sweep)
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

            trans = 'N';
            dgbtrsl_(&trans, &ldw, &ku, &kl, &klu, localA + TC(startAiL, mySize - 2 * klu, lda), &lda, ipivAux, W[0][p], &ldw, &info);

            uplo = 'L';
            dlacpy_(&uplo, &kl, &kl, W[0][p] + 2 * klu - kl, &ldw, W[0][p] + klu - kl, &ldw);

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
            std::cout << "Time of UL factorization for A in process " << p << std::endl;
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
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime();
#endif
            //! LU factorize Ap
            dgbalu_(&mySize, &kl, &ku, localA + TC(0, 0, lda), &lda, &nzero, &norma, &info);
#ifdef DETAIL_TIME
            detailTime = MPI_Wtime() - detailTime;
            if (myId == 1)
            {
                std::cout << "Time of LU factorization for A in process " << p << std::endl;
            }
#endif

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

    max_info = std::max(info, max_info);
    info = max_info;

    /* **********************************
   * Get reduced system of 1st level*
   ********************************** */

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

    /* ***************************************
       * Recursive spike factorization stage *
       * Get S[nLevel]                       *
       *************************************** */

#ifdef STAGE_TIME
    stageTime = MPI_Wtime();
#endif

    RecursiveSpike(kl, ku, V, W, R, nPart, nLevel, mpiInfo);

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
            std::cout << "Process " << i << ",time of recursive factorization stage: " << stageTimes[i] << std::endl;
        }
    }
#endif

    IDeallocate_1D(partitionsSize);
}