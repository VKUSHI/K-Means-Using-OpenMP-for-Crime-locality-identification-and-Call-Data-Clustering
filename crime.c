#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <float.h>
#include <time.h>
#define MAX_LOCATIONS 1000
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


void assignToClusters(const double locations[MAX_LOCATIONS][3], const double centroids[MAX_LOCATIONS][3], int clusters[MAX_LOCATIONS], int numLocations, int numClusters)
{
    #pragma omp parallel for
    for (int i = 0; i < numLocations; ++i)
    {
        double minDistance = DBL_MAX;
        int clusterIndex = -1;
        for (int j = 0; j < numClusters; ++j)
        {
            double distance = euclideanDistance(locations[i], centroids[j], 3);
            if (distance < minDistance)
            {
                minDistance = distance;
                clusterIndex = j;
            }
        }
        clusters[i] = clusterIndex;
    }
}


void SassignToClusters(const double Slocations[MAX_LOCATIONS][3], const double Scentroids[MAX_LOCATIONS][3], int Sclusters[MAX_LOCATIONS], int SnumLocations, int SnumClusters)
{

    for (int i = 0; i < SnumLocations; ++i)
    {
        double minDistance = DBL_MAX;
        int clusterIndex = -1;
        for (int j = 0; j < SnumClusters; ++j)
        {
            double distance = euclideanDistance(Slocations[i], Scentroids[j], 3);
            if (distance < minDistance)
            {
                minDistance = distance;
                clusterIndex = j;
            }
        }
        Sclusters[i] = clusterIndex;
    }
}

void updateCentroids(const double locations[MAX_LOCATIONS][3], const int clusters[MAX_LOCATIONS], double centroids[MAX_LOCATIONS][3], int numLocations, int numFeatures, int numClusters)
{
    double clusterSums[MAX_LOCATIONS][3] = {0.0};
    int clusterCounts[MAX_LOCATIONS] = {0};

    #pragma omp parallel for
    for (int i = 0; i < numLocations; ++i)
    {
        int clusterIndex = clusters[i];
        #pragma omp critical
        {
            for (int j = 0; j < numFeatures; ++j)
            {
                clusterSums[clusterIndex][j] += locations[i][j];
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

void SupdateCentroids(const double Slocations[MAX_LOCATIONS][3], const int Sclusters[MAX_LOCATIONS], double Scentroids[MAX_LOCATIONS][3], int SnumLocations, int numFeatures, int SnumClusters)
{
    double clusterSums[MAX_LOCATIONS][3] = {0.0};
    int clusterCounts[MAX_LOCATIONS] = {0};


    for (int i = 0; i < SnumLocations; ++i)
    {
        int clusterIndex = Sclusters[i];

        {
            for (int j = 0; j < numFeatures; ++j)
            {
                clusterSums[clusterIndex][j] += Slocations[i][j];
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
     int SnumClusters = 2;
    int SnumThreads = 4;

    FILE* Sfile = fopen("crime_data.txt", "r");
    if (Sfile == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int SnumLocations;
    fscanf(Sfile, "%d", &SnumLocations);

    double Slocations[MAX_LOCATIONS][3];
    for (int i = 0; i < SnumLocations; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            fscanf(Sfile, "%lf", &Slocations[i][j]);
        }
    }

    fclose(Sfile);

    srand(time(0));
    double Scentroids[MAX_LOCATIONS][3];
    for (int i = 0; i < SnumClusters; ++i)
    {
        int randomIndex = rand() % SnumLocations;
        for (int j = 0; j < 3; ++j)
        {
            Scentroids[i][j] = Slocations[randomIndex][j];
        }
    }
    int SmaxIterations = MAX_ITERATIONS;
    int Sclusters[MAX_LOCATIONS];

    for (int iter = 0; iter < SmaxIterations; ++iter)
    {
        SassignToClusters(Slocations, Scentroids, Sclusters, SnumLocations, SnumClusters);
        SupdateCentroids(Slocations, Sclusters, Scentroids, SnumLocations, 3, SnumClusters);
    }

    printf("\nCluster Assignments and Crime Classification:\n");
    for (int i = 0; i < SnumLocations; ++i)
    {
        printf("Location %d assigned to cluster %d - ", i + 1, Sclusters[i] + 1);
        if (Slocations[i][2] > 10)
        {

            printf("Crime-Prone\n");
        }
        else
        {
            printf("Non-Crime\n");
        }
    }

}

void parallel(){
    int numClusters = 2;
    int numThreads = 4;
    omp_set_num_threads(numThreads);

    FILE* file = fopen("crime_data.txt", "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int numLocations;
    fscanf(file, "%d", &numLocations);

    double locations[MAX_LOCATIONS][3];
    for (int i = 0; i < numLocations; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            fscanf(file, "%lf", &locations[i][j]);
        }
    }

    fclose(file);

    srand(time(0));
    double centroids[MAX_LOCATIONS][3];
    for (int i = 0; i < numClusters; ++i)
    {
        int randomIndex = rand() % numLocations;
        for (int j = 0; j < 3; ++j)
        {
            centroids[i][j] = locations[randomIndex][j];
        }
    }
    int maxIterations = MAX_ITERATIONS;
    int clusters[MAX_LOCATIONS];

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        assignToClusters(locations, centroids, clusters, numLocations, numClusters);
        updateCentroids(locations, clusters, centroids, numLocations, 3, numClusters);
    }

    printf("\nCluster Assignments and Crime Classification:\n");
    for (int i = 0; i < numLocations; ++i)
    {
        printf("Location %d assigned to cluster %d - ", i + 1, clusters[i] + 1);
        if (locations[i][2] > 10)
        {

            printf("Crime-Prone\n");
        }
        else
        {
            printf("Non-Crime\n");
        }
    }

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
