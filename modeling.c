#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<dirent.h>
#include<string.h>

void process(char *file, int rows, int cols, int *img, int L, int R){

    int out_degree[rows*cols][R];
    double wout_degree[rows*cols][R];
    double win_degree[rows*cols][R];

    char *out_dir = (char*) malloc(sizeof(char)*50);
    char *wout_dir = (char*) malloc(sizeof(char)*50);
    char *win_dir = (char*) malloc(sizeof(char)*50);

    strcpy(out_dir, "data/1200Tex/net_measures/");
    strcpy(wout_dir, "data/1200Tex/net_measures/");
    strcpy(win_dir, "data/1200Tex/net_measures/");

    char *sout = (char*) malloc(strlen(file)+sizeof(char)*30);
    char *swout = (char*) malloc(strlen(file)+sizeof(char)*30);
    char *swin = (char*) malloc(strlen(file)+sizeof(char)*30);
    snprintf(sout, strlen(file)-3, "%s", file);
    snprintf(swout, strlen(file)-3, "%s", file);
    snprintf(swin, strlen(file)-3, "%s", file);
    strcat(sout, "_out_degree.csv");
    strcat(swout, "_wout_degree.csv");
    strcat(swin, "_win_degree.csv");


    strcat(out_dir, sout);
    strcat(wout_dir, swout);
    strcat(win_dir, swin);

    free(sout);
    free(swout);
    free(swin);

    FILE *out, *wout, *win;

    out = fopen(out_dir, "w");
    wout = fopen(wout_dir, "w");
    win = fopen(win_dir, "w");

    free(out_dir);
    free(wout_dir);
    free(win_dir);


    for(int r=1;r<=R;r++){
        if(r<R){
            fprintf(out, "r=%d, ", r);
            fprintf(wout, "r=%d, ", r);
            fprintf(win, "r=%d, ", r);
        }
        else{
            fprintf(out, "r=%d", r);
            fprintf(wout, "r=%d", r);
            fprintf(win, "r=%d", r);
        }
        
        for (int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                out_degree[j+i*cols][r-1] = 0;
                wout_degree[j+i*cols][r-1] = 0;
                win_degree[j+i*cols][r-1] = 0;

            }
        }
    }

    double weight;
    for (int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            for(int y=i-R;y<=i+R;y++){
                if(y >= 0 && y<rows){
                    for(int x=j-R;x<=i+R;x++){
                        if(x >= 0 && x<cols){
                            for(int r=1;r<=R;r++){
                                weight=0;
                                double d = sqrt(pow(i-y,2)+pow(j-x,2));
                                if(img[j+i*cols] <= img[x+y*cols] && d<=r){
                                    int diff = abs(img[j+i*cols] - img[x+y*cols]);
                                    if(r==1){
                                        weight= (double) diff/L;
                                    }
                                    else{
                                        weight = ((d-1)/(double)(r-1) + diff/(double)L)/2;
                                    }
                                    out_degree[j+i*cols][r-1]+=1;
                                    wout_degree[j+i*cols][r-1]+=weight;
                                    win_degree[x+y*cols][r-1]+=weight;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
        

    fprintf(out, "\n");
    fprintf(wout, "\n");
    fprintf(win, "\n");

    for(int i=0;i<rows*cols;i++){
        for(int r=1;r<=R;r++){
            if(r<R){
                fprintf(out, "%d, ", out_degree[i][r-1]);
                fprintf(wout, "%.5f, ", wout_degree[i][r-1]);
                fprintf(win, "%.5f, ", win_degree[i][r-1]);
            }
            else{
                fprintf(out, "%d\n", out_degree[i][r-1]);
                fprintf(wout, "%.5f\n", wout_degree[i][r-1]);
                fprintf(win, "%.5f\n", win_degree[i][r-1]);
            }
        }
    }
    
    fclose(out);
    fclose(wout);
    fclose(win);

    return;

}

const char *get_filename_ext(const char *filename){
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

int main(void){

    int L=255;
    int R=11;

    struct dirent *dir;
    DIR *d;
    char *sdir = (char*) malloc(sizeof(char)*30);
    sprintf(sdir, "data/1200Tex/matrices/");
    d = opendir(sdir);
    int counter = 1;
    if(d){
        while((dir = readdir(d)) != NULL){
            if(!strcmp(get_filename_ext(dir->d_name), "txt")){
                printf("Iniciando imagem %d\n", counter);
                FILE *matrix;
                char *path = (char*) malloc(sizeof(char)*30);
                strcpy(path, sdir);
                strcat(path, dir->d_name);
                matrix = fopen(path, "r");

                int value;
                int size;

                if(matrix){
                    fscanf(matrix, "%d", &size);
                }

                int *img = (int*)malloc(sizeof(int)*size);
                int i=0;
                while(fscanf(matrix, "%d", &value)!= EOF){
                    img[i]=value;
                    i++;
                }
                
                int rows = sqrt(size);
                int cols = sqrt(size);

                fclose(matrix);


                process(dir->d_name, rows, cols, img,L,R);
                counter++;

                free(path);
            }
            
        }
    }
    closedir(d);

    return 0;
}
