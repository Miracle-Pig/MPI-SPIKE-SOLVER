#include "util.h"
#include "macro.h"
#include "MpiSpikeSolver.h"

#include <algorithm>
#include <math.h>
#include <iostream>
#include <new>
#include <string.h>
#include <string>
#include <sstream>

void ABORT(const std::string &s, MPI_Comm &comm)
{
    std::cout << s << std::endl;
    MPI_Abort(comm, MPI_ERR_INFO);
}

/* 
  !Memory allocate or deallocate 
*/

int *IAllocate_1D(size_t len, int *allocInfo)
{
    int *array;
    *allocInfo = 0;
    try
    {
        array = new int[len];
    }
    catch (std::bad_alloc &memExp)
    {
        *allocInfo = -1;
        std::cerr << memExp.what() << std::endl;
    }
    return array;
}

void IDeallocate_1D(int *array)
{
    delete[] array;
}

double *DAllocate_1D(size_t len, int *allocInfo)
{
    double *array;
    *allocInfo = 0;
    try
    {
        array = new double[len];
    }
    catch (std::bad_alloc &memExp)
    {
        *allocInfo = -1;
        std::cerr << memExp.what() << std::endl;
    }
    return array;
}

void DDeallocate_1D(double *array)
{
    delete[] array;
}

double *DZAllocate_1D(size_t len, int *allocInfo)
{
    double *array;
    *allocInfo = 0;
    try
    {
        array = new double[len];
        for (size_t i = 0; i < len; i++)
            array[i] = 0.0;
    }
    catch (std::bad_alloc &memExp)
    {
        *allocInfo = -1;
        std::cerr << memExp.what() << std::endl;
    }
    return array;
}

double **DAllocate_2D(size_t r, size_t c, int *allocInfo)
{
    double **array;
    *allocInfo = 0;
    try
    {
        array = new double *[r];
        for (size_t i = 0; i < r; i++)
        {
            array[i] = new double[c];
        }
    }
    catch (std::bad_alloc &memExp)
    {
        *allocInfo = -1;
        std::cerr << memExp.what() << std::endl;
    }

    return array;
}
double **DZAllocate_2D(size_t r, size_t c, int *allocInfo)
{
    double **array;
    *allocInfo = 0;
    try
    {
        array = new double *[r];
        for (size_t i = 0; i < r; i++)
        {
            array[i] = new double[c];
            for (size_t j = 0; j < c; j++)
                array[i][j] = 0.0;
        }
    }
    catch (std::bad_alloc &memExp)
    {
        *allocInfo = -1;
        std::cerr << memExp.what() << std::endl;
    }

    return array;
}
void DDeallocate_2D(double **array, int r)
{
    for (int i = 0; i < r; i++)
    {
        delete[] array[i];
    }
}

double ***DAllocate_3D(size_t r, size_t c, size_t l, int *allocInfo)
{
    double ***array;
    *allocInfo = 0;
    try
    {
        array = new double **[r];
        for (size_t i = 0; i < r; i++)
        {
            array[i] = new double *[c];
            for (size_t j = 0; j < c; j++)
            {
                array[i][j] = new double[l];
                memset(array[i][j], 0, sizeof(double) * l);
            }
        }
    }
    catch (std::bad_alloc &memExp)
    {
        *allocInfo = -1;
        std::cerr << memExp.what() << std::endl;
    }
    return array;
}
void DDeallocate_3D(double ***array, size_t r, size_t c)
{
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            delete[] array[i][j];
        }
        delete[] array[i];
    }
}

/* 
    !Set default Option 
*/
void SetDefaultOptions(Options_t *options, MpiInfo_t *mpiInfo)
{
    int curParCnt;
    options->OptimizeFlag = 1;
    options->IsPivot = NO;
    options->HasB = YES;
    options->IsSym = NO;
    options->LargeRatio = 18;
    options->SmallRatio = 35;
    options->EqualRatio = 10;
    options->IsRatioEqual = NO;
    options->SelfMakeA = NO;

    if (options->OptimizeFlag == 1)
    {
        options->LargeRatio = 10;
    }
    options->K = 1.6;
    options->IterativeMax = 3;
    options->DResidualTol = 12;
    options->SResidualTol = 5;
    options->IfPrint = 0;
    options->Norm = NORM1;
    options->NProcesses = mpiInfo->nProcess;
    options->PartitionCnt = mpiInfo->nProcess;
    options->LevelsCnt = 0;
    options->NPar2Proc = 0;
    curParCnt = options->PartitionCnt;
    while (curParCnt != 1)
    {
        curParCnt /= 2;
        options->LevelsCnt++;
    }
    options->lwork = 4 * (options->LevelsCnt * options->PartitionCnt + options->PartitionCnt - 1);
    options->NPar1Proc = options->NProcesses - 2 * options->NPar2Proc;
}

/* Read matrix from files */
void ReadAndBoradcastMatrix(BandedMatrix *A, int *nrhs, double *&localF, char **argv, Options_t *options, MpiInfo_t *mpiInfo)
{
    int n, nnz, ku, kl, klu, lda,
        allocInfo,
        myId, nPart, mySize;
    int *partitionsSize;
    char **cpp;
    bool flag;
    double *tmpA, *tmp, *globalF;
    FILE *file[2];
    MPI_Status status;

    myId = mpiInfo->myId;
    nPart = mpiInfo->nProcess;

    if (myId == 0)
    {
        if (!options->SelfMakeA)
        {
            /*
                ! Parse argv
            */
            cpp = argv + 1;
            if (!(file[0] = fopen(*cpp, "r")))
            {
                ABORT("File A does not exist", mpiInfo->comm);
            }
            if (*(cpp + 1) == NULL)
            {
                options->HasB = NO;
            }
            else
            {
                options->HasB = YES;
            }

            flag = false;

            BandWidth(file[0], &n, &nnz, &ku, &kl);

            if (options->IsSym)
            {
                if (ku > kl)
                {
                    flag = true;
                }
                kl = std::max(kl, ku);
                ku = kl;
            }

            A->kl = kl;
            A->ku = ku;
            A->n = n;
            klu = std::max(kl, ku);

            A->storageGlob = DZAllocate_1D(n * (kl + ku + 1), &allocInfo);

            if (!(file[0] = fopen(*cpp, "r")))
            {
                ABORT("File A does not exist", mpiInfo->comm);
            }
            ReadBigBandA(file[0], A->storageGlob, n, nnz, ku, kl);

            if (options->IsSym)
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if ((flag && abs(i - j) <= klu && i > j) || (!flag && abs(i - j) <= klu && i < j))
                        {
                            int x = DenseToBand1D(i, j, kl, ku), y = DenseToBand1D(j, i, kl, ku);
                            A->storageGlob[x] = A->storageGlob[y];
                        }
                    }
                }
            }
            if (options->HasB)
            {
                ++cpp;
                if (!(file[1] = fopen(*cpp, "r")))
                {
                    ABORT("File A does not exist", mpiInfo->comm);
                }
                ReadB(file[1], globalF, nrhs);
            }
            else
            {
                *nrhs = 3;
                globalF = DAllocate_1D(n * (*nrhs), &allocInfo);
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < n; j++)
                    {
                        if (abs(i - j) <= klu)
                        {
                            sum += A->storageGlob[TC(ku + i - j, j, kl + ku + 1)];
                        }
                    }
                    for (int j = 0; j < *nrhs; j++)
                    {
                        globalF[TC(i, j, n)] = sum;
                    }
                }
            }
        }
        else
        {
            n = 128000;
            kl = 300;
            ku = 300;
            klu = std::max(kl, ku);
            lda = kl + ku + 1;
            nnz = n * lda - (1 + ku) * ku;
            *nrhs = 200;

            A->kl = kl;
            A->ku = ku;
            A->n = n;
            A->storageGlob = DAllocate_1D(lda * n, &allocInfo);
            for (int i = 0; i < lda; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == ku)
                    {
                        A->storageGlob[TC(i, j, lda)] = 10;
                    }
                    else
                    {
                        A->storageGlob[TC(i, j, lda)] = 1;
                    }
                }
            }
            globalF = DAllocate_1D(n * (*nrhs), &allocInfo);
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    if (abs(i - j) <= klu)
                    {
                        sum += A->storageGlob[DenseToBand1D(i, j, kl, ku)];
                    }
                }
                for (int j = 0; j < *nrhs; j++)
                {
                    globalF[TC(i, j, n)] = sum;
                }
            }
        }

        lda = kl + ku + 1;
        if (options->IsPivot)
        {
            lda = klu + kl + ku + 1;
            tmpA = DZAllocate_1D(n * lda, &allocInfo);
            for (int i = klu; i < lda; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    tmpA[TC(i, j, lda)] = A->storageGlob[TC(i - klu, j, kl + ku + 1)];
                }
            }
            tmp = A->storageGlob;
            A->storageGlob = tmpA;
            DDeallocate_1D(tmp);
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&kl, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&ku, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&nnz, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(nrhs, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&lda, 1, MPI_INT, 0, mpiInfo->comm);
    }
    else
    {

        MPI_Bcast(&n, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&kl, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&ku, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&nnz, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(nrhs, 1, MPI_INT, 0, mpiInfo->comm);
        MPI_Bcast(&lda, 1, MPI_INT, 0, mpiInfo->comm);

        A->kl = kl;
        A->ku = ku;
        A->n = n;
        klu = std::max(kl, ku);
    }

    partitionsSize = IAllocate_1D(nPart + 1, &allocInfo);
    if (options->IsRatioEqual)
    {
        CaculEqPartitionsSize(partitionsSize, n, options);
    }
    else
    {
        CaculPartitionsSize(partitionsSize, n, options);
    }
    mySize = partitionsSize[myId + 1] - partitionsSize[myId];

    if (mySize < 2 * klu)
    {
        ABORT("Bandwidth dose not satisfy the condition!!!",mpiInfo->comm);
    }

    A->storageLoc = DAllocate_1D(mySize * lda, &allocInfo);
    localF = DAllocate_1D(mySize * (*nrhs), &allocInfo);
    if (myId == 0)
    {
        for (int i = 1; i < nPart; i++)
        {
            MPI_Send(A->storageGlob + TC(0, partitionsSize[i], lda),
                     (partitionsSize[i + 1] - partitionsSize[i]) * lda,
                     MPI_DOUBLE, i, 100 + i, mpiInfo->comm);
            for (int j = 0; j < *nrhs; j++)
            {
                MPI_Send(globalF + TC(partitionsSize[i], j, n),
                         partitionsSize[i + 1] - partitionsSize[i], MPI_DOUBLE, i,
                         1000 + 100 * i + j, mpiInfo->comm);
            }
        }
        char uplo = 'F';

        dlacpy_(&uplo, &lda, &mySize, A->storageGlob, &lda, A->storageLoc, &lda);
        dlacpy_(&uplo, &mySize, &(*nrhs), globalF, &n, localF, &mySize);
        DDeallocate_1D(A->storageGlob);
        DDeallocate_1D(globalF);
    }
    else
    {
        MPI_Recv(A->storageLoc, mySize * lda,
                 MPI_DOUBLE, 0, 100 + myId, mpiInfo->comm, &status);
        for (int j = 0; j < *nrhs; j++)
        {
            MPI_Recv(localF + TC(0, j, mySize), mySize * (*nrhs), MPI_DOUBLE,
                     0, 1000 + 100 * myId + j, mpiInfo->comm, &status);
        }
    }

    IDeallocate_1D(partitionsSize);
}

/* Read banded storageGlob of A */
void ReadBigBandA(FILE *file, double *A, int dim, int len, int supDiag, int subDiag)
{
    int i, m, n;
    double value;
    char line[512], banner[64];

    banner[0] = '%';
    while (banner[0] == '%')
    {
        fgets(line, 512, file);
        sscanf(line, "%s", banner);
    }

    for (i = 0; i < len; i++)
    {
        fscanf(file, "%d %d  %lf\n", &m, &n, &value);
        A[DenseToBand1D((m - 1), (n - 1), subDiag, supDiag)] = value;
    }
    fclose(file);
}

void ReadB(FILE *file, double *&B, int *nrhs)
{
    int i, j, len,
        allocInfo;
    char *s, line[512], banner[64];
    double value;

    /* 2/ Skip comments */
    banner[0] = '%';
    while (banner[0] == '%')
    {
        s = fgets(line, 512, file);
        sscanf(line, "%s", banner);
    }
    std::istringstream iss(s);
    iss >> len, iss >> *nrhs;
    B = DAllocate_1D(len * (*nrhs), &allocInfo);
    for (i = 0; i < len; i++)
    {
        s = fgets(line, 512, file);
        std::istringstream iss(s);
        for (j = 0; j < *nrhs; j++)
        {
            iss >> value;
            B[TC(i, j, len)] = value;
        }
    }
    fclose(file);
}

/* Determine the number of the subDiagonals and supDiagonals */
void BandWidth(FILE *file, int *dim, int *len, int *supDiag, int *subDiag)
{
    int i, m, n, tmp;
    double value;
    char *s, line[512], banner[64];

    banner[0] = '%';
    while (banner[0] == '%')
    {
        s = fgets(line, 512, file);
        sscanf(line, "%s", banner);
    }
    std::istringstream iss(s);
    iss >> *dim, iss >> tmp, iss >> *len;
    if (*dim != tmp)
    {
        printf("This is not a squared matrix!");
        exit(1);
    }
    *subDiag = 0;
    *supDiag = 0;
    for (i = 0; i < *len; i++)
    {
        fscanf(file, "%d %d  %lg\n", &m, &n, &value);
        *supDiag = std::max(*supDiag, n - m);
        *subDiag = std::max(*subDiag, m - n);
    }
    fclose(file);
}

/* Caculate the size of partitions */
void CaculEqPartitionsSize(int *partitionsSize, int n, Options_t *options)
{
    int midNums, midSize, firstSize;
    int rSmall; //rLarge,

    midNums = options->PartitionCnt - 2;

    // rLarge = options->LargeRatio;
    rSmall = 0.1 * options->EqualRatio;

    partitionsSize[0] = 0;

    midSize = int((1.0 * n) / (rSmall * (2 + midNums / rSmall)));
    firstSize = (n - midNums * midSize) / 2;

    partitionsSize[1] = firstSize;

    for (int i = 2; i < 2 + midNums; i++)
    {
        partitionsSize[i] = partitionsSize[i - 1] + midSize;
    }

    partitionsSize[options->PartitionCnt] = n;
}

void CaculPartitionsSize(int *partitionsSize, int n, Options_t *options)
{
    int midNums,
        sizeSmall, sizeHuge, //sizeLarge,
        rSmall;              //rLarge;

    midNums = options->PartitionCnt - 2;

    // rLarge = 0.1 * options->LargeRatio;
    rSmall = 0.2 * options->LargeRatio;
    // rSmall = options->SmallRatio;

    partitionsSize[0] = 0;

    // sizeLarge = (int)((1.0 * n) / (rLarge * (2 + midNums / rSmall)));
    // sizeLarge += sizeLarge % 2;
    sizeSmall = (int)((1.0 * n) / (rSmall * (2 + midNums / rSmall)));
    sizeHuge = (n - (midNums * sizeSmall)) / 2;

    partitionsSize[1] = sizeHuge;

    for (int i = 2; i < options->PartitionCnt; i++)
    {
        partitionsSize[i] = partitionsSize[i - 1] + sizeSmall;
    }

    partitionsSize[options->PartitionCnt] = n;
}
