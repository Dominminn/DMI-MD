/* -*- c++ -*- ----------------------------------------------------------
   DFT-CES core subroutines. Written by H.-K. Lim
   Copyright (C) 2016 M-design group @ KAIST
------------------------------------------------------------------------- */

#ifndef LMP_GRID_H
#define LMP_GRID_H

#include "pointers.h"

namespace LAMMPS_NS {

class Grid : protected Pointers {
 public:
  int ncubes;    		                                     // number of output cubefiles
  int savetag;			                                     // save CES grid or not	
  char **savedcube;                                                  // cubefile for save
  int gnx, gny, gnz;                                                 // # of grid points in 3 dim
  double gx[3],gy[3],gz[3];                                          // grid spacing vectors of (unit: Ang)
  int natoms;                                                        // # of atoms
  int *atomns;                                                       // atomic numbers for each atom
  double **basis;                                                    // cartesian coords of each atom (unit: Ang)
  double *gvout0,*gvout1,*gvout2,*gvout3,*gvout4;                    // grid values for MD rho  (unit: Ry)
  double *gvout_all0,*gvout_all1,*gvout_all2,*gvout_all3,*gvout_all4;// grid values for MD rho  (unit: Ry)
  Grid(class LAMMPS *, int, char **);
  ~Grid();

  void read_header(char *);
  void read_content(char*);
  void save_grid(char **, int);
};

}

#endif

