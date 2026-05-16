/* ----------------------------------------------------------------------
DMI-IMD Cube Subroutine. Original Code Modified by Jiseok Oh 
Copyright (C) 2025 M-design group @ KAIST
   
Original Code: fix_gridforce.cpp for DFT-CES. Written by H.-K. Lim
Copyright (C) 2016 M-design group @ KAIST
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "fix_cubesave.h"
#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "atom_masks.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "grid.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCubeSave::FixCubeSave(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 11&& narg != 6 ) error->all(FLERR,"Illegal fix cubesave command: weight,superfactor,cubeID,tip4p?,then typeO,H,theta(degree),Blen(Angs),qdist(Angs)");
  else if (narg == 11&& strcmp(arg[5],"tip4p") != 0 ) error->all(FLERR,"Use tip4p for input character");
  else if (narg == 6&& strcmp(arg[5],"tip4p") == 0 ) error->all(FLERR,"need more args or don't use tip4p for character unless don't simulate tip4p");
  
  sfactor = atoi(arg[3]);
  cubeID = atoi(arg[4]);
  if(comm->me == 0) printf("#### DMI-IMD: Cubefile Saving Information ####\n");
  if(comm->me == 0) printf("- supercell factor = %d\n", sfactor);
  if(comm->me == 0) printf("- cubefile ID      = %d\n", cubeID);

  if (strcmp(arg[5],"tip4p") == 0){ 
    tip4p_CES = 1;
    typeO = atoi(arg[6]);
    typeH = atoi(arg[7]); 
    theta = atof(arg[8]); 
    blen = atof(arg[9]); 
    qdist = atof(arg[10]); 
  }
  else {
    tip4p_CES = 0;
    typeO = 0;
    typeH = 0; 
    theta = 0.0; 
    blen =  0.0; 
    qdist = 0.0; 
  }
  // set alpha parameter

  alpha = qdist / (cos(0.5*theta*M_PI/180) * blen);
}

/* ---------------------------------------------------------------------- */

FixCubeSave::~FixCubeSave()
{
  return;
}

/* ---------------------------------------------------------------------- */

int FixCubeSave::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCubeSave::init()
{
  // check variables
  if (domain->grid->natoms == 0) error->all(FLERR,"Grid data is unavilable");

}

/* ---------------------------------------------------------------------- */

void FixCubeSave::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixCubeSave::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type; //TIP4P
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  double fx, fy, fz;
  int iH1, iH2;
  double newsite[3];//TIP4P massless site

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
        if ( tip4p_CES == 1 ) {
           if (type[i] == typeO) {
		   iH1 = atom->map(tag[i] + 1);
		   iH2 = atom->map(tag[i] + 2);
		   iH1 = domain->closest_image(i, iH1);
		   iH2 = domain->closest_image(i, iH2);
		   compute_newsite(x[i],x[iH1],x[iH2],newsite);
		   triInter(q[i], newsite[0], newsite[1], newsite[2],cubeID);
           
	   } 
           else {
              triInter(q[i], x[i][0], x[i][1], x[i][2],cubeID);
           } 
        }
        else {
           triInter(q[i], x[i][0], x[i][1], x[i][2],cubeID);
        }
    }
  }
}

void FixCubeSave::triInter(double q, double x, double y, double z, int CUBEID){
  int i;
  double px, py, pz, xd, yd, zd;
  double p000, p100, p010, p001, p110, p101, p011, p111;
  int gnx = domain->grid->gnx;
  int gny = domain->grid->gny;
  int gnz = domain->grid->gnz;
  double gsx = domain->grid->gx[0];
  double gsy = domain->grid->gy[1];
  double gsz = domain->grid->gz[2];
  double gvol;
  if(x<0){
    px =(fmod(x,gnx*gsx)+gnx*gsx)/gsx;
  }else{
    px = fmod(x,gnx*gsx)/gsx;
  }
  if(y<0){
    py =(fmod(y,gny*gsy)+gny*gsy)/gsy;
  }else{
    py = fmod(y,gny*gsy)/gsy;
  }
  if(z<0){
    pz =(fmod(z,gnz*gsz)+gnz*gsz)/gsz;
  }else{
    pz = fmod(z,gnz*gsz)/gsz;
  }
  xd=(double)(px-(int)px);
  yd=(double)(py-(int)py);
  zd=(double)(pz-(int)pz);

  // inverse trilinear interpolation for saving MD rho
  p000=(1-xd)*(1-yd)*(1-zd)*q;
  p100=xd*(1-yd)*(1-zd)*q;
  p010=(1-xd)*yd*(1-zd)*q;
  p110=xd*yd*(1-zd)*q;
  p001=(1-xd)*(1-yd)*zd*q;
  p101=xd*(1-yd)*zd*q;
  p011=(1-xd)*yd*zd*q;
  p111=xd*yd*zd*q;

  gvol = gsx*gsy*gsz;

  if(CUBEID ==0){ double* gvout = domain->grid->gvout0;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p000/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p100/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p010/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p001/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p110/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p101/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p011/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p111/gvol/sfactor;
  }
  else if(CUBEID ==1){ double* gvout = domain->grid->gvout1;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p000/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p100/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p010/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p001/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p110/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p101/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p011/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p111/gvol/sfactor;
  }
  else if(CUBEID ==2){ double* gvout = domain->grid->gvout2;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p000/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p100/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p010/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p001/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p110/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p101/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p011/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p111/gvol/sfactor;
  }
  else if(CUBEID ==3){ double* gvout = domain->grid->gvout3;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p000/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p100/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p010/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p001/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p110/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p101/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p011/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p111/gvol/sfactor;
  }
  else if(CUBEID ==4){ double* gvout = domain->grid->gvout4;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p000/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p100/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p010/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p001/gvol/sfactor;
    gvout[((int)pz)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p110/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p101/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px)%gnx)*gnz*gny] += p011/gvol/sfactor;
    gvout[((int)pz+1)%gnz+(((int)py+1)%gny)*gnz+(((int)px+1)%gnx)*gnz*gny] += p111/gvol/sfactor;
  }
  return;
}


/* ----------------------------------------------------------------------
  compute position xM of fictitious charge site for O atom and 2 H atoms
  return it as xM
------------------------------------------------------------------------- */

void FixCubeSave::compute_newsite(double *xO,  double *xH1,
                                        double *xH2, double *xM)
{
  double delx1 = xH1[0] - xO[0];
  double dely1 = xH1[1] - xO[1];
  double delz1 = xH1[2] - xO[2];

  double delx2 = xH2[0] - xO[0];
  double dely2 = xH2[1] - xO[1];
  double delz2 = xH2[2] - xO[2];

  xM[0] = xO[0] + alpha * 0.5 * (delx1 + delx2);
  xM[1] = xO[1] + alpha * 0.5 * (dely1 + dely2);
  xM[2] = xO[2] + alpha * 0.5 * (delz1 + delz2);
}
