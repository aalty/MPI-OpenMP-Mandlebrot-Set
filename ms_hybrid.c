#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
//#include <time.h>
//#define BILLION 1E9

void write_png(const char* filename, const int width, const int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
	int x, y;
    for (y = 0; y < height; ++y) {
        memset(row, 0, row_size);
		//#pragma omp parallel for schedule(dynamic) private(x)
        for (x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            row[x * 3] = ((p & 0xf) << 4);
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* argument parsing */
    assert(argc == 9);
    int threads = strtol(argv[1], 0, 10);
    double left = strtod(argv[2], 0);
    double right = strtod(argv[3], 0);
    double lower = strtod(argv[4], 0);
    double upper = strtod(argv[5], 0);
    int width = strtol(argv[6], 0, 10);
    int height = strtol(argv[7], 0, 10);
    const char* filename = argv[8];
	double vert_int = (upper-lower) / height;
	double hori_int = (right-left) / width;
	
	int rank, proc_size;
	MPI_Comm VALID_COMM, OTHER_COMM;
	
	MPI_Status st;
	MPI_Request send_req, recv_req;
	int flag, master_batch, slave_batch;
	slave_batch = (int)sqrt(width * height)*threads*3;
	master_batch = threads*2;
    int* image_loc;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
	
	if(width*height <= slave_batch) proc_size = 1;
    else if(1 + width*height/slave_batch < proc_size) proc_size = 1 + width*height/slave_batch;

    if(rank >= proc_size) {
        MPI_Comm_split(MPI_COMM_WORLD, rank / proc_size, rank, &OTHER_COMM);
        return 0;
    }
    else MPI_Comm_split(MPI_COMM_WORLD, rank / proc_size, rank, &VALID_COMM);

    /* allocate memory for image */
	if(rank == 0) {
		image_loc =  (int*)malloc(width * height * sizeof(int));
    	assert(image_loc);
	}
	//if(rank == 0) {image = (int*)malloc(width * height * sizeof(int)); assert(image);}
	//assert(image);


	/*struct timespec Comm_start, Comm_end;
	double Comm_time, Comm_time_temp;
	Comm_time = Comm_time_temp = 0;*/

    /* mandelbrot set */
	if(proc_size > 1) {
		if(rank > 0) {
			int *assign_index = (int*)malloc(2*sizeof(int));
			//itialize 
			assign_index[0] = (rank-1)*slave_batch;	//start index
			assign_index[1] = rank*slave_batch;	//end index
			int *slave_buff = (int*)malloc((slave_batch+2)*sizeof(int));	//with start&end index
			while(1){
				//printf("Slave%d - assign_index[0]: %d, assign_index[1]: %d\n", rank, assign_index[0], assign_index[1]);
				if(assign_index[0] == -1) break;			
				slave_buff[0] = assign_index[0];
				slave_buff[1] = assign_index[1];

				int i;
				#pragma omp parallel for schedule(dynamic) private(i)
				for(i = assign_index[0]; i < assign_index[1]; i++) {
					//if(i/width >= height) break;
					double y0 = (i/width)*vert_int + lower;
					double x0 = (i%width)*hori_int + left;

    	    	    int repeats = 0;
    	    	    double x = 0;
    	    	    double y = 0;
    	    	    double length_squared = 0;
    	    	    while (repeats < 100000 && length_squared < 4) {
    	    	        double temp = x * x - y * y + x0;
    	    	        y = 2 * x * y + y0;
    	    	        x = temp;
    	    	        length_squared = x * x + y * y;
    	    	        ++repeats;
    	    	    }
    	    	    slave_buff[2+i-assign_index[0]] = repeats;
    	    	}
				MPI_Isend(slave_buff, 2+assign_index[1]-assign_index[0], MPI_INT, 0, 0, VALID_COMM, &send_req);
				//if(assign_index[1] == width*height) break;
				MPI_Recv(assign_index, 2, MPI_INT, 0, 0, VALID_COMM, &st);
			}
		}
		else if(rank == 0) {
			int current_index, total = 0, end = -1, assign_batch = slave_batch;
			int *send_index = (int*)malloc(2*sizeof(int));
			int *master_buff = (int*)malloc((slave_batch+2)*sizeof(int));
			//int *slave_guide = (int*)malloc(proc_size*sizeof(int));
			//for(int i=0; i<proc_size; i++) slave_guide[i] = 0;
			current_index = (proc_size-1)*slave_batch;
			//printf("Master - current_index: %d\n", current_index);
			MPI_Irecv(master_buff, slave_batch+2, MPI_INT, MPI_ANY_SOURCE, 0, VALID_COMM, &recv_req);
			while(1) {
				MPI_Test(&recv_req, &flag, &st);
               	//printf("flag: %d\n", flag); 
                if(flag != 0) {
					total += master_buff[1]-master_buff[0]; 
					assign_batch = (assign_batch-300 > (int)sqrt(width * height)*threads) ? assign_batch-300 : (int)sqrt(width * height)*threads;
						
					if(current_index+assign_batch >= width*height) {
						send_index[0] = current_index;
						send_index[1] = width*height;
						current_index = width*height;	//finish assign all points
					}		
					else {
						send_index[0] = current_index;
						send_index[1] = current_index + assign_batch;
						current_index += assign_batch;
					}
                    MPI_Isend(send_index, 2, MPI_INT, st.MPI_SOURCE, 0, VALID_COMM, &send_req);
                    //printf("Master - current_index: %d\n", current_index);
                    //std::copy(master_buff+2, master_buff+2+(master_buff[1]-master_buff[0]), image_loc+master_buff[0]);  
					memcpy(image_loc+master_buff[0], master_buff+2, (master_buff[1]-master_buff[0])*sizeof(int));
					MPI_Irecv(master_buff, slave_batch+2, MPI_INT, MPI_ANY_SOURCE, 0, VALID_COMM, &recv_req);
					if(current_index >= width*height) break;
                }
				else {
					int i, master_end;
                    master_end = (current_index+master_batch >= width*height)? width*height : current_index+master_batch;
                    
					#pragma omp parallel for schedule(dynamic) private(i) 
                    for(i = current_index; i < master_end; i++) {
						double y0 = (i/width)*vert_int + lower;
						double x0 = (i%width)*hori_int + left;	
						int repeats = 0;
						double x = 0;
						double y = 0;
						double length_squared = 0;
						while (repeats < 100000 && length_squared < 4) {
							double temp = x * x - y * y + x0;
							y = 2 * x * y + y0;
							x = temp;
							length_squared = x * x + y * y;
							++repeats;
						}
						image_loc[i] = repeats;	
					}
					total += master_end - current_index;
					current_index = master_end;
					//printf("Master - current_index: %d\n", current_index);
					if(current_index >= width * height) break;
				}
			}
			
			//MPI_Irecv(master_buff, slave_batch+2, MPI_INT, MPI_ANY_SOURCE, 0, VALID_COMM, &recv_req);
			send_index[0] = -1; send_index[1] = width*height;
			while(total < width*height-1) {
				//printf("Master - total: %d\n", total);
				MPI_Test(&recv_req, &flag, &st);
				if(flag != 0) {
					total += master_buff[1]-master_buff[0];
					MPI_Isend(send_index, 2, MPI_INT, st.MPI_SOURCE, 0, VALID_COMM, &send_req);
					//std::copy(master_buff+2, master_buff+2+(master_buff[1]-master_buff[0]), image_loc+master_buff[0]);
					memcpy(image_loc+master_buff[0], master_buff+2, (master_buff[1]-master_buff[0])*sizeof(int));
					MPI_Irecv(master_buff, slave_batch+2, MPI_INT, MPI_ANY_SOURCE, 0, VALID_COMM, &recv_req);
				}
			}
			write_png(filename, width, height, image_loc);
			free(image_loc);
			//for(int i=1; i<proc_size; i++) MPI_Isend(send_index, 2, MPI_INT, st.MPI_SOURCE, 0, VALID_COMM, &send_req);

		}
	
	}
	else {
    	for (int j = 0; j < height; ++j) {
    	    double y0 = j * vert_int + lower;
    	    for (int i = rank; i < width; i+=proc_size) {
    	        double x0 = i * hori_int + left;

    	        int repeats = 0;
    	        double x = 0;
    	        double y = 0;
    	        double length_squared = 0;
    	        while (repeats < 100000 && length_squared < 4) {
    	            double temp = x * x - y * y + x0;
    	            y = 2 * x * y + y0;
    	            x = temp;
    	            length_squared = x * x + y * y;
    	            ++repeats;
    	        }
    	        image_loc[j * width + i] = repeats;
    	    }
    	}
		write_png(filename, width, height, image_loc);
		free(image_loc);
	}
	//MPI_Barrier(VALID_COMM);
	//MPI_Reduce(&Comm_time_temp, &Comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, VALID_COMM);
	//if(rank == 0) printf("Comm_time: %lf\n", Comm_time);
	//MPI_Finalize();

    /* draw and cleanup */
    /*if(rank == 0) {
		write_png(filename, width, height, image_loc);
		free(image_loc);
	}*/
}
