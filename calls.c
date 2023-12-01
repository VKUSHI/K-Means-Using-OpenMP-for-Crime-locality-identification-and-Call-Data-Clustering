#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include<float.h>
#include <time.h>

#define MAX_CALLS 1000
#define MAX_ITERATIONS 100
double euclideanDistance(const double* a, const double* b, int numFeatures)
{
    double sum = 0.0;
    for (int i = 0; i < numFeatures; ++i)
    {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

void generate_samples(int num_samples, FILE *file) {
    srand(time(NULL));

    for (int i = 0; i < num_samples; ++i) {
        double duration = ((double)rand() / RAND_MAX) * (16.0 - 6.5) + 6.5;
        int calls = rand() % 5 + 1;
        fprintf(file, "%.1f %d\n", duration, calls);
    }
}


void assignToClusters(const double calls[MAX_CALLS][2], const double centroids[MAX_CALLS][2], int clusters[MAX_CALLS], int numCalls, int numClusters)
{
    #pragma omp parallel for
    for (int i = 0; i < numCalls; ++i)
    {
        double minDistance = DBL_MAX;
        int clusterIndex = -1;
        for (int j = 0; j < numClusters; ++j)
        {
            double distance = euclideanDistance(calls[i], centroids[j], 2);
            if (distance < minDistance)
            {
                minDistance = distance;
                clusterIndex = j;
            }
        }

        clusters[i] = clusterIndex;
    }
}


void SassignToClusters(const double Scalls[MAX_CALLS][2], const double Scentroids[MAX_CALLS][2], int Sclusters[MAX_CALLS], int SnumCalls, int SnumClusters)
{

    for (int i = 0; i < SnumCalls; ++i)
    {
        double minDistance = DBL_MAX;
        int clusterIndex = -1;
        for (int j = 0; j < SnumClusters; ++j)
        {
            double distance = euclideanDistance(Scalls[i], Scentroids[j], 2);
            if (distance < minDistance)
            {
                minDistance = distance;
                clusterIndex = j;
            }
        }

        Sclusters[i] = clusterIndex;
    }
}


void updateCentroids(const double calls[MAX_CALLS][2], const int clusters[MAX_CALLS], double centroids[MAX_CALLS][2], int numCalls, int numFeatures, int numClusters)
{
    double clusterSums[MAX_CALLS][2] = {0.0};
    int clusterCounts[MAX_CALLS] = {0};

    #pragma omp parallel for
    for (int i = 0; i < numCalls; ++i)
    {
        int clusterIndex = clusters[i];
        #pragma omp critical
        {
            for (int j = 0; j < numFeatures; ++j)
            {
                clusterSums[clusterIndex][j] += calls[i][j];
            }
            clusterCounts[clusterIndex]++;
        }
    }
    for (int i = 0; i < numClusters; ++i)
    {
        for (int j = 0; j < numFeatures; ++j)
        {
            centroids[i][j] = (clusterCounts[i] == 0) ? 0.0 : (clusterSums[i][j] / clusterCounts[i]);
        }
    }
}


void SupdateCentroids(const double Scalls[MAX_CALLS][2], const int Sclusters[MAX_CALLS], double Scentroids[MAX_CALLS][2], int SnumCalls, int numFeatures, int SnumClusters)
{
    double clusterSums[MAX_CALLS][2] = {0.0};
    int clusterCounts[MAX_CALLS] = {0};


    for (int i = 0; i < SnumCalls; ++i)
    {
        int clusterIndex = Sclusters[i];
        #pragma omp critical
        {
            for (int j = 0; j < numFeatures; ++j)
            {
                clusterSums[clusterIndex][j] += Scalls[i][j];
            }
            clusterCounts[clusterIndex]++;
        }
    }
    for (int i = 0; i < SnumClusters; ++i)
    {
        for (int j = 0; j < numFeatures; ++j)
        {
            Scentroids[i][j] = (clusterCounts[i] == 0) ? 0.0 : (clusterSums[i][j] / clusterCounts[i]);
        }
    }
}


void serial(){


    int SnumClusters = 3;
    int SnumThreads = 4;


    FILE *Sfile = fopen("call_records.txt", "r");
    if (Sfile == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int SnumCalls;
    fscanf(Sfile, "%d", &SnumCalls);

    double Scalls[MAX_CALLS][2];
    for (int i = 0; i < SnumCalls; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            fscanf(Sfile, "%lf", &Scalls[i][j]);
        }
    }

    srand(time(0));
    double Scentroids[MAX_CALLS][2];
    for (int i = 0; i < SnumClusters; ++i)
    {
        int randomIndex = rand() % SnumCalls;
        for (int j = 0; j < 2; ++j)
        {
            Scentroids[i][j] = Scalls[randomIndex][j];
        }
    }
    int SmaxIterations = MAX_ITERATIONS;
    int Sclusters[MAX_CALLS];

    for (int iter = 0; iter < SmaxIterations; ++iter)
    {
        SassignToClusters(Scalls, Scentroids, Sclusters, SnumCalls, SnumClusters);
        SupdateCentroids(Scalls, Sclusters, Scentroids, SnumCalls, 2, SnumClusters);
    }
    double SmaxDuration = -1.0;
    int SmaxDurationCluster = -1;

    for (int i = 0; i < SnumClusters; ++i)
    {
        double clusterDuration = 0.0;
        int count = 0;

        for (int j = 0; j < SnumCalls; ++j)
        {
            if (Sclusters[j] == i)
            {
                clusterDuration += Scalls[j][0];
                count++;
            }
        }

        if (count > 0)
        {
            clusterDuration /= count;
            if (clusterDuration > SmaxDuration)
            {
                SmaxDuration = clusterDuration;
                SmaxDurationCluster = i;
            }
        }
    }

    for (int i = 0; i < SnumCalls; ++i)
    {
        printf("Call %d assigned to cluster %d\n", i + 1, Sclusters[i] + 1);
    }

    if (SmaxDurationCluster != -1)
    {
        printf("\nCluster with maximum average call duration: %d\n", SmaxDurationCluster + 1);
    }
    else
    {
        printf("\nNo clusters found\n");
    }

    fclose(Sfile);


}

void parallel(){


    int numClusters = 3;
    int numThreads = 4;
    omp_set_num_threads(numThreads);


    FILE *file = fopen("call_records.txt", "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int numCalls;
    fscanf(file, "%d", &numCalls);

    double calls[MAX_CALLS][2];
    for (int i = 0; i < numCalls; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            fscanf(file, "%lf", &calls[i][j]);
        }
    }

    srand(time(0));
    double centroids[MAX_CALLS][2];
    for (int i = 0; i < numClusters; ++i)
    {
        int randomIndex = rand() % numCalls;
        for (int j = 0; j < 2; ++j)
        {
            centroids[i][j] = calls[randomIndex][j];
        }
    }
    int maxIterations = MAX_ITERATIONS;
    int clusters[MAX_CALLS];

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        assignToClusters(calls, centroids, clusters, numCalls, numClusters);
        updateCentroids(calls, clusters, centroids, numCalls, 2, numClusters);
    }
    double maxDuration = -1.0;
    int maxDurationCluster = -1;

    for (int i = 0; i < numClusters; ++i)
    {
        double clusterDuration = 0.0;
        int count = 0;

        for (int j = 0; j < numCalls; ++j)
        {
            if (clusters[j] == i)
            {
                clusterDuration += calls[j][0];
                count++;
            }
        }

        if (count > 0)
        {
            clusterDuration /= count;
            if (clusterDuration > maxDuration)
            {
                maxDuration = clusterDuration;
                maxDurationCluster = i;
            }
        }
    }

    for (int i = 0; i < numCalls; ++i)
    {
        printf("Call %d assigned to cluster %d\n", i + 1, clusters[i] + 1);
    }

    if (maxDurationCluster != -1)
    {
        printf("\nCluster with maximum average call duration: %d\n", maxDurationCluster + 1);
    }
    else
    {
        printf("\nNo clusters found\n");
    }

    fclose(file);

}

int main()
{

    double serialStart = omp_get_wtime();
    printf("\nSerial Execution started at time %lf\n", serialStart);
    serial();
    double serialEnd = omp_get_wtime();
    printf("\nSerial Execution started at time %lf\n", serialEnd);
    double totalSerial = serialEnd - serialStart;
    printf("Total Serial Execution time is %lf\n", totalSerial);


    double parallelStart = omp_get_wtime();
    printf("\nParallel Execution started at time %lf\n", parallelStart);
    parallel();
    double parallelEnd = omp_get_wtime();
    printf("\nParallel Execution started at time %lf\n", parallelEnd);
    double totalParallel = parallelEnd - parallelStart;
    printf("Total Parallel Execution time is %lf\n", totalParallel);


    printf("\nTotal Serial Execution time is %lf\n", totalSerial);
    printf("Total Parallel Execution time is %lf\n", totalParallel);


    return 0;
}
