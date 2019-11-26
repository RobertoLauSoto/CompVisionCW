#include "3dArray.h"
// Example code for allocating a 3D int array on the heap, many other more efficient solutions are possible

#include <stdio.h>
#include <stdlib.h>

int ***malloc3dArray(int dim1, int dim2, int dim3){
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
        for (j = 0; j < dim2; j++) {
            array[i][j] = (int *) malloc(dim3 * sizeof(int));
        }
    }
    for (i = 0; i < dim1; ++i)
        for (j = 0; j < dim2; ++j)
            for (k = 0; k < dim3; ++k)
                array[i][j][k] = 0;

    return array;
}

int **malloc2dArray(int dim1, int dim2){
    int** array = new int*[dim1];
    for(int i = 0; i < dim1; ++i){
        array[i] = new int[dim2];
    }
    return array;
}