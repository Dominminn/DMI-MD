/* -*- c++ -*- ----------------------------------------------------------
   DFT-CES core subroutines. Written by H.-K. Lim
   Copyright (C) 2016 M-design group @ KAIST
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(cubesave,FixCubeSave)

#else

#ifndef LMP_FIX_CUBESAVE_H
#define LMP_FIX_CUBESAVE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixCubeSave : public Fix {
 public:
  FixCubeSave(class LAMMPS *, int, char **);
  ~FixCubeSave();
  int setmask();
  void init();
  void setup(int);
  void post_force(int);
  void triInter(double, double, double, double, int);
 private:
  int sfactor;        // supercell factor
  int cubeID;        // cubefile id   
  int tip4p_CES; //DFT-CES with tip4p tag 

 protected:
  int typeH,typeO;             // atom types of TIP4P water H and O atoms
  double theta,blen;             // angle and bond types of TIP4P water
  double alpha;                // geometric constraint parameter for TIP4P
  double qdist;

  void compute_newsite(double *, double *, double *, double *);

};

}

#endif
#endif
