#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480

int main(int argc, char** argv) {
    double commtime;
    MPI_Init(&argc, &argv);
    double start = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double real_min = -1.0;
    double real_max = 1.0;
    double imag_min = -1.0;
    double imag_max = 1.0;
    int max_counter = 1000;
    int sub_region = HEIGHT / size;
    int start_row = sub_region * rank;
    int end_row = start_row + sub_region;
    if (rank == size - 1) {
        end_row = HEIGHT;
    }

    int* row = (int*)malloc(sizeof(int) * WIDTH);
    int* data = (int*)malloc(sizeof(int) * WIDTH * sub_region);

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double c_real = real_min + (real_max - real_min) * x / WIDTH;
            double c_imag = imag_min + (imag_max - imag_min) * y / HEIGHT;
            double z_real = 0.0;
            double z_imag = 0.0;

            int counter = 0;
            while (z_real * z_real + z_imag * z_imag < 4.0 && counter < max_counter) {
                double next_z_real = z_real * z_real - z_imag * z_imag + c_real;
                double next_z_imag = 2.0 * z_real * z_imag + c_imag;
                z_real = next_z_real;
                z_imag = next_z_imag;
                counter++;
            }

            if (counter == max_counter) {
                row[x] = 0;
            } else {
                row[x] = counter % 256;
            }
        }
        int r_index = (y - start_row) * WIDTH;

        for (int x = 0; x < WIDTH; x++) {
            data[r_index + x] = row[x];
        }
    }

    free(row);
    int* Data = NULL;
    if (rank == 0) {
        Data = (int*)malloc(sizeof(int) * WIDTH * HEIGHT);
    }
    
    double starttime = MPI_Wtime();
    MPI_Gather(data, WIDTH * sub_region, MPI_INT, Data, WIDTH * sub_region, MPI_INT, 0, MPI_COMM_WORLD);
    double endtime = MPI_Wtime();
    commtime = endtime - starttime;
    free(data);

    if (rank == 0) {
        FILE* fp = fopen("mandelbrot.pgm", "wb");
        fprintf(fp, "P5\n%d %d\n255\n", WIDTH, HEIGHT);
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            fputc(Data[i], fp);
        }
        fclose(fp);
        free(Data);
    }

    double end = MPI_Wtime();
    double elapsed = end - start;
    printf("Elapsed time: %f seconds\n", elapsed);
    printf("commtime: %f seconds\n", commtime);
    MPI_Finalize();
    return 0;
}
