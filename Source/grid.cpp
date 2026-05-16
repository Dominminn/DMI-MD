/* ----------------------------------------------------------------------
DMI-IMD Cube Subroutine. Original Code Modified by Jiseok Oh 
Copyright (C) 2025 M-design group @ KAIST
   
Original Code: grid.cpp for DFT-CES. Written by H.-K. Lim
Copyright (C) 2016 M-design group @ KAIST
------------------------------------------------------------------------- */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "grid.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "fix_cubesave.h"
#include <string.h>

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

Grid::Grid(LAMMPS *lmp, int narg, char **arg) : Pointers(lmp)
{
  int me = comm -> me;
  int i;

  natoms = 0;
  atomns = NULL;
  basis  = NULL;
  gvout0= NULL;
  gvout1= NULL;
  gvout2= NULL;
  gvout3= NULL;
  gvout4= NULL;
  gvout_all0 = NULL;
  gvout_all1 = NULL;
  gvout_all2 = NULL;
  gvout_all3 = NULL;
  gvout_all4 = NULL;
  savedcube = NULL;

  if (narg < 4) error->all(FLERR,"Illegal grid command");

  if (strcmp(arg[0],"empty") != 0 ){ 
    // read header until atom information
    read_header(arg[0]);

    if(gx[1]+gx[2]+gy[0]+gy[2]+gz[0]+gz[1] != 0) error->all(FLERR,"Grid file is not orthorhombic");
    if(me == 0){
      printf("#### DMI-IMD: Grid File Info ####\n");
      printf("- Grid file has been parsed: %s\n",arg[0]);
      printf("- # of atoms in grid: %d\n", natoms);
      printf("- # of grid points: %d %d %d\n", gnx, gny, gnz);
      printf("- grid spacing: %f %f %f Angs\n", gx[0], gy[1], gz[2]);
    }
    memory->grow(atomns, natoms, "grid:atomns");
    memory->grow(basis, natoms, 3, "grid:basis");
    memory->grow(gvout0, gnx*gny*gnz, "grid:gvout0");
    memory->grow(gvout1, gnx*gny*gnz, "grid:gvout1");
    memory->grow(gvout2, gnx*gny*gnz, "grid:gvout2");
    memory->grow(gvout3, gnx*gny*gnz, "grid:gvout3");
    memory->grow(gvout4, gnx*gny*gnz, "grid:gvout4");
    if(me==0)memory->grow(gvout_all0, gnx*gny*gnz, "grid:gvout_all0");
    if(me==0)memory->grow(gvout_all1, gnx*gny*gnz, "grid:gvout_all1");
    if(me==0)memory->grow(gvout_all2, gnx*gny*gnz, "grid:gvout_all2");
    if(me==0)memory->grow(gvout_all3, gnx*gny*gnz, "grid:gvout_all3");
    if(me==0)memory->grow(gvout_all4, gnx*gny*gnz, "grid:gvout_all4");

    // read atom and grid value information
    read_content(arg[0]);
  }
  else{
    return;
  }
  
  savetag = 1; //default is save
  if (strcmp(arg[1],"no") == 0 ){ 
    savetag = 0;//don't save
  }
  if ( savetag==0 ){ if(me == 0) printf("Grid file will not saved\n"); }
  //else { if(me == 0) printf("Grid file will be saved\n"); }


  ncubes = atoi(arg[2]); 
  savedcube = new char*[5];         // cubefile for save

  for(int temp=0; temp<5;temp++){
    savedcube[temp] = (char *) "empty";
  }
  for(int temp=0; temp<ncubes;temp++){ 
    int n = strlen(arg[temp+3]) + 1;
    savedcube[temp] = new char[n];
    strcpy(savedcube[temp],arg[temp+3]);
    if(me == 0) printf("grid will be saved in %s\n",savedcube[temp]);
  }
}
/* ---------------------------------------------------------------------- */

Grid::~Grid()
{
  memory->destroy(atomns);
  memory->destroy(basis);
  memory->destroy(gvout0);
  memory->destroy(gvout1);
  memory->destroy(gvout2);
  memory->destroy(gvout3);
  memory->destroy(gvout4);
  memory->destroy(gvout_all0);
  memory->destroy(gvout_all1);
  memory->destroy(gvout_all2);
  memory->destroy(gvout_all3);
  memory->destroy(gvout_all4);
}

/* ---------------------------------------------------------------------- */

void Grid::read_header(char *filename)
{
  int me = comm->me;
  FILE *fptr;
  char line[MAXLINE];
  double temp[4];

  if (me == 0) {
    fptr = fopen(filename,"r");
    if (fptr == NULL) {
      char str[128];
      sprintf(str,"Cannot open grid file %s",filename);
      error->one(FLERR,str);
    }

    fgets(line,MAXLINE,fptr);
    fgets(line,MAXLINE,fptr); // skip first two rows
    fgets(line,MAXLINE,fptr);
    sscanf(line,"%d %lf %lf %lf", &natoms, &temp[0], &temp[1], &temp[2]);
    fgets(line,MAXLINE,fptr);
    sscanf(line,"%d %lf %lf %lf", &gnx, &gx[0], &gx[1], &gx[2]);
    gx[0] *= 0.52917721;
    fgets(line,MAXLINE,fptr);
    sscanf(line,"%d %lf %lf %lf", &gny, &gy[0], &gy[1], &gy[2]);
    gy[1] *= 0.52917721;
    fgets(line,MAXLINE,fptr);
    sscanf(line,"%d %lf %lf %lf", &gnz, &gz[0], &gz[1], &gz[2]);
    gz[2] *= 0.52917721;
    fclose(fptr);
  }
  MPI_Bcast(&natoms,1,MPI_INT,0,world);
  MPI_Bcast(&gnx,1,MPI_INT,0,world);
  MPI_Bcast(&gny,1,MPI_INT,0,world);
  MPI_Bcast(&gnz,1,MPI_INT,0,world);
  MPI_Bcast(&gx[0],3,MPI_DOUBLE,0,world);
  MPI_Bcast(&gy[0],3,MPI_DOUBLE,0,world);
  MPI_Bcast(&gz[0],3,MPI_DOUBLE,0,world);
}
void Grid::read_content(char *filename)
{
  int me = comm->me;
  FILE *fptr;
  char line[MAXLINE], *str_ptr;
  double temp;
  int i, cnt;

  if (me == 0) {
    fptr = fopen(filename,"r");
    for(i=0; i<6; i++) fgets(line,MAXLINE,fptr);
    for(i=0; i<natoms; i++){
      fgets(line,MAXLINE,fptr);
      sscanf(line,"%d %lf %lf %lf %lf", &atomns[i], &temp, &basis[i][0], &basis[i][1], &basis[i][2]);
      basis[i][0] *= 0.52917721;
      basis[i][1] *= 0.52917721;
      basis[i][2] *= 0.52917721;
    }
    fclose(fptr);
  }
  MPI_Bcast(&atomns[0],natoms,MPI_INT,0,world);
  MPI_Bcast(&basis[0][0],3*natoms,MPI_DOUBLE,0,world);
}
void Grid::save_grid(char** filename, int nsteps)
{
  int me = comm->me;
  FILE *fptr;
  int i, j, k, cnt;

  for (int temp=0;temp<ncubes;temp++){
    if(temp ==0) MPI_Reduce(gvout0,gvout_all0,gnx*gny*gnz,MPI_DOUBLE,MPI_SUM,0,world);
    else if(temp ==1) MPI_Reduce(gvout1,gvout_all1,gnx*gny*gnz,MPI_DOUBLE,MPI_SUM,0,world);
    else if(temp ==2) MPI_Reduce(gvout2,gvout_all2,gnx*gny*gnz,MPI_DOUBLE,MPI_SUM,0,world);
    else if(temp ==3) MPI_Reduce(gvout3,gvout_all3,gnx*gny*gnz,MPI_DOUBLE,MPI_SUM,0,world);
    else if(temp ==4) MPI_Reduce(gvout4,gvout_all4,gnx*gny*gnz,MPI_DOUBLE,MPI_SUM,0,world);

    if (me == 0 ){
      fptr = fopen(filename[temp], "w");
      fprintf(fptr,"MDrho from DMI-IMD\n");
      fprintf(fptr,"ZYX, Bohr unit\n");
      fprintf(fptr,"% 5d    0.000000    0.000000    0.000000\n",natoms);
      fprintf(fptr,"% 5d% 12.6lf    0.000000    0.000000\n",gnx,gx[0]/0.52917721);
      fprintf(fptr,"% 5d    0.000000% 12.6lf    0.000000\n",gny,gy[1]/0.52917721);
      fprintf(fptr,"% 5d    0.000000    0.000000% 12.6lf\n",gnz,gz[2]/0.52917721);
      for(i=0; i<natoms; i++) {
        fprintf(fptr,"% 5d% 12.6lf% 12.6lf% 12.6lf% 12.6lf\n",atomns[i],(double)atomns[i],basis[i][0]/0.52917721,basis[i][1]/0.52917721,basis[i][2]/0.52917721);
      }
      for(i=0;i<gnx;i++){
        for(j=0;j<gny;j++){
          cnt = 0;
          for(k=0;k<gnz;k++){
            if(temp ==0) fprintf(fptr,"% 13.5lE",gvout_all0[k+j*gnz+i*gnz*gny]*pow(0.52917721,3)/(nsteps+1));
            else if(temp ==1) fprintf(fptr,"% 13.5lE",gvout_all1[k+j*gnz+i*gnz*gny]*pow(0.52917721,3)/(nsteps+1));
            else if(temp ==2) fprintf(fptr,"% 13.5lE",gvout_all2[k+j*gnz+i*gnz*gny]*pow(0.52917721,3)/(nsteps+1));
            else if(temp ==3) fprintf(fptr,"% 13.5lE",gvout_all3[k+j*gnz+i*gnz*gny]*pow(0.52917721,3)/(nsteps+1));
            else if(temp ==4) fprintf(fptr,"% 13.5lE",gvout_all4[k+j*gnz+i*gnz*gny]*pow(0.52917721,3)/(nsteps+1));
            if(cnt%6 == 5 && k < gnz-1) fprintf(fptr,"\n");
            cnt++;
          }
          fprintf(fptr,"\n");
        }
      }
      fclose(fptr);
      printf("\n#######\nDMI-IMD: %s has been successfully saved\n\n",filename[temp]);
    }
  }
}

