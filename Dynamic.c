#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    int count = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char *argv[]) {
    int image[HEIGHT][WIDTH];
    double AVG = 0;
    int N = 10;
    double total_time[N];
    struct complex c;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        int next_row = 0;
        for (int k = 0; k < N; k++) {
            clock_t start_time = clock();
            int rows_per_worker = HEIGHT / size;
            for (int dest = 1; dest < size; dest++) {
                MPI_Send(&next_row, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                next_row += rows_per_worker;
            }
            int remaining_rows = HEIGHT - next_row;
            for (int i = next_row; i < HEIGHT; i++) {
                for (int j = 0; j < WIDTH; j++) {
                    c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                    c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                    image[i][j] = cal_pixel(c);
                }
            }
            for (int source = 1; source < size; source++) {
                MPI_Recv(&next_row, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                rows_per_worker = remaining_rows / (size - source);
                for (int i = next_row; i < next_row + rows_per_worker; i++) {
                    for (int j = 0; j < WIDTH; j++) {
                        c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                        c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                        image[i][j] = cal_pixel(c);
                    }
                }
                remaining_rows -= rows_per_worker;
            }
            clock_t end_time = clock();
            total_time[k] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
            printf("Execution time of trial [%d]: %f seconds\n", k, total_time[k]);
            AVG += total_time[k];
        }
    } else {
        int start_row;
        MPI_Recv(&start_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int rows_per_worker = HEIGHT / size;
        for (int i = start_row; i < start_row + rows_per_worker; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
        }
        MPI_Send(&start_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    if (rank == 0) {
        save_pgm("mandelbrot.pgm", image);
        printf("The average execution time of %d trials is: %f ms\n", N, AVG / N * 1000);
    }

    return 0;
}
