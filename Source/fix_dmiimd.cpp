/*-------------------------------------------------------------------------
 DMI-IMD: Dynamic Mirror Image charge - Interfacial Molecular Dynamics. 
 Image Particle Subroutine with Plane-Wave Based Lattice Mirror for Ag(111)
 Copyright (C) 2025 M-Design Group @ KAIST. Jiseok Oh and Yoosang Son
--------------------------------------------------------------------------*/

#include <string.h>
#include <stdlib.h>
#include "fix_dmiimd.h"
#include "fix.h"
#include "atom.h"
#include "atom_masks.h"
#include "error.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "math.h"
#include "memory.h"
#include "neighbor.h"
#include "comm.h"
#include "domain.h"
#include <cmath>
#include <unordered_set>
#include <algorithm>

using namespace LAMMPS_NS;
using namespace std;
using namespace FixConst;

/*--------------------------------------------------------------*/
FixDmiImd::FixDmiImd(LAMMPS *lmp, int narg, char **arg) : 
        Fix(lmp, narg, arg)
{
        group_id = -1;
        group_id = group->find(arg[1]);
        if (group_id == -1) error->all(FLERR, "Invalid group ID");
        
	// Mirror parameters: f(z) = c - exp(a-bz), Top/Hollow
	a_1 = atof(arg[3]);
        b_1 = atof(arg[4]);
        c_1 = atof(arg[5]);
        a_2 = atof(arg[6]);
        b_2 = atof(arg[7]);
        c_2 = atof(arg[8]);

	// Ag xy coordinates
        x_ag = atof(arg[9]);
        y_ag = atof(arg[10]);

        spacing = atoi(arg[11]);
        imagemin = spacing - 1; // Assumes that data ids are arrranged as: {real} -> {slab} -> {image} (N_slab > 0) 
        
	// Postforce Routine. -1.0 for full force subtraction. 
	multiplier = atof(arg[12]);
	if (multiplier == 0.0) postforce_flag = 0;
	else postforce_flag = 1;

	if (narg == 18 && strcmp(arg[13],"tip4p") == 0) {
		tip4p_flag = 1;
        	//TIP4p parameters
        	typeO = atoi(arg[14]);
        	blen = atof(arg[15]);
        	theta = atof(arg[16]);
        	qdist = atof(arg[17]);
		numpermol = 0;
	}
	else if (narg == 14) {
		tip4p_flag = 0;
		numpermol = atof(arg[13]);
		typeO = 0;
		blen = 0.0;
		theta = 0.0;
		qdist = 0.0;
	}
	else error->all(FLERR, "DMI-IMD: invalid number of arguments");
	if(comm->me == 0) printf("#### DMI-IMD: Group %s ####\n", arg[1]);
	if (tip4p_flag == 1) {
        	if(comm->me == 0) printf("- Top mirror: -exp(%f - %f*z) + %f and Hollow mirror: -exp(%f - %f*z) + %f\n ", a_1, b_1, c_1, a_2, b_2, c_2);
        	if(comm->me == 0 && postforce_flag == 0) printf("- No force subtraction with group %s\n", arg[1]);
		if(comm->me == 0 && postforce_flag == 1) printf("- Force subtraction with group %s weight %f\n", arg[1], multiplier);
		//TIP4P alpha parameter
        	alpha = qdist/(std::cos(0.5*theta*3.14159265358979323846/180)*blen);
		if(comm->me == 0) printf("- TIP4P water model with alpha =  %f\n", alpha);
	}
	else {
                if(comm->me == 0) printf("- Top mirror: -exp(%f - %f*z) + %f and Hollow mirror: -exp(%f - %f*z) + %f\n ", a_1, b_1, c_1, a_2, b_2, c_2);
		if(comm->me == 0 && postforce_flag == 0) printf("- No force subtraction with group %s\n", arg[1]);
		if(comm->me == 0 && postforce_flag == 1) printf("- Force subtraction with group %s weight %f\n", arg[1], multiplier);
        }
	if(comm->me == 0) printf("####\n");
}

/* ---------------------------------------------------------------------- */

FixDmiImd::~FixDmiImd() {
        return;
}

/* ---------------------------------------------------------------------- */

int FixDmiImd::setmask() {
        int mask = 0;
        mask |= POST_INTEGRATE; //Image atom adjustment before force calculation
        mask |= POST_FORCE; //Post_force adjustments, if needed.
        return mask;
}

/* ---------------------------------------------------------------------- */

void FixDmiImd::init() {
        if (a_1 < 0.0 || a_2 < 0.0 || b_1 < 0.0 || b_2 < 0.0 || c_1 < 0.0 || c_2 < 0.0) error->all(FLERR, "All Mirror Parameters should not be negative.");
}

/* ---------------------------------------------------------------------- */

void FixDmiImd::setup(int vflag)
{
        post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixDmiImd::compute_geometric_mir(double x, double y, double z) {
        //Assume the mirror fitting is done as: top = c_1 - exp(a_1-b_1*z), hol = c_2 - exp(a_2-b_2*z),where z is explicit coordinate(not relative to Ag layer).
        double L = 2.930375*std::sqrt(3.0)/2.0;
        
        double xlo = domain->boxlo[0];
        double xhi = domain->boxhi[0];
        double ylo = domain->boxlo[1];
        double yhi = domain->boxhi[1];

        double Lx = xhi - xlo;
        double Ly = yhi - ylo;

        // Move coordinates inside the simulation box (Seems not necessary, LAMMPS automatically invokes in-box values. just in case..)
        double x_shift = x - Lx*std::floor((x - xlo)/Lx) - x_ag;
        double y_shift = y - Ly*std::floor((y - ylo)/Ly) - y_ag;

        double pi = 3.14159265358979323846;
        double k = 2*pi/L;
        // Plane-wave interpolation: gamma =  Re(e^ik_1r + e^ik_2r + e^ik_3r), k_1 = (1, 0), k_2 = (1/2, sqrt(3.0)/2), k_3 = (1/2, -sqrt(3.0)/2)
        double gamma = (2.0/3.0)*(0.5+(1.0/3.0)*(std::cos(k*x_shift) 
                                + std::cos(0.5*k*x_shift+(std::sqrt(3.0)/2.0)*k*y_shift) 
                                + std::cos(0.5*k*x_shift-(std::sqrt(3.0)/2.0)*k*y_shift))); 
        double mir = -std::exp(gamma*a_1+(1.0-gamma)*a_2-(gamma*b_1+(1.0-gamma)*b_2)*z) + gamma*c_1 + (1.0-gamma)*c_2;
        return mir;
}

/* ---------------------------------------------------------------------- */

double FixDmiImd::compute_integral(double x, double y, double z) {
        double L = 2.930375*std::sqrt(3.0)/2.0;
        double integral = 0.0;
        double dz = 0.002;
        double tol = 0.00001;
        double prev = 0.0;

        double xlo = domain->boxlo[0];
        double xhi = domain->boxhi[0];
        double ylo = domain->boxlo[1];
        double yhi = domain->boxhi[1];
        
        double Lx = xhi - xlo;
        double Ly = yhi - ylo;

        double x_shift = x - Lx*std::floor((x - xlo)/Lx) - x_ag;
        double y_shift = y - Ly*std::floor((y - ylo)/Ly) - y_ag;
        
        double pi = 3.14159265358979323846;
        double k = 2*pi/L;

        // Plane-wave interpolation: gamma =  Re(e^ik_1r + e^ik_2r + e^ik_3r), k_1 = (1, 0), k_2 = (1/2, sqrt(3.0)/2), k_3 = (1/2, -sqrt(3.0)/2)
        double gamma = (2.0/3.0)*(0.5+(1.0/3.0)*(std::cos(k*x_shift) 
                                + std::cos(0.5*k*x_shift+(std::sqrt(3.0)/2.0)*k*y_shift) 
                                + std::cos(0.5*k*x_shift-(std::sqrt(3.0)/2.0)*k*y_shift)));

        for (int i = 0; i < 50001; i++) {
                double z_i = z + 0.5*dz*(i+1);
                double mir = -std::exp(gamma*a_1+(1.0-gamma)*a_2-(gamma*b_1+(1.0-gamma)*b_2)*z_i) + gamma*c_1 + (1.0-gamma)*c_2;
                double val = (a_1-a_2-z_i*(b_1-b_2))*std::exp(gamma*a_1+(1.0-gamma)*a_2-(gamma*b_1+(1.0-gamma)*b_2)*z_i); 
                integral += val*(std::pow((z_i-mir), -3.0))*dz;
                //if (std::abs(integral - prev) < tol) {
                        //break;
                //}
                //if (i == 50000) error->all(FLERR, "z-integral for xy force does not converge");
                //prev = integral;
        }
        return integral;
}

/* ---------------------------------------------------------------------- */

void FixDmiImd::post_integrate() {
        double dt = update->dt;
        double **x = atom->x;
        double **f = atom->f;
        double **v = atom->v;
        double *q = atom->q;
        double *mass = atom->mass;
        int *type = atom->type;
        int *tag = atom->tag;
        int nlocal = atom->nlocal;
        int nghost = atom->nghost;
        int *mask = atom->mask;

	// For each processor, pick an image atom, search for the paired real atom, and adjust image atom position.
        for (int idx2 = 0; idx2 < nlocal; idx2++) {
                int idx1 = -1;
		int idx3 = 0;
                if (tag[idx2] < imagemin) {
                        continue;
                }
                if (mask[idx2] & groupbit) {
			idx1 = atom->map(tag[idx2]-spacing);	
			idx1 = domain->closest_image(idx2, idx1);
			if (idx1 == -1) {
                                error->all(FLERR, "Paired Real Atom not found. Check the processor or datafile.");
                        }
                        if (std::abs(q[idx1]+q[idx2]) > 0.000001) error->all(FLERR, "q_real + q_image should be 0.");
                        if (tip4p_flag == 1 && type[idx1]==typeO) {
                                int iH1, iH2;
                                iH1 = atom->map(tag[idx1] + 1); //assumes {O,H,H} ordering of the datafile.
                                iH2 = atom->map(tag[idx1] + 2);
                                iH1 = domain->closest_image(idx1,iH1); // move H atoms as the closest image to O atoms.
                                iH2 = domain->closest_image(idx1,iH2);
                                if (iH1 == -1 || iH2 == -1) {
                                        error->all(FLERR,"TIP4P hydrogen is missing");
                                }
                                double x_M = 0.5*alpha*(x[iH1][0]+x[iH2][0])+(1.0-alpha)*(x[idx1][0]);
                                double y_M = 0.5*alpha*(x[iH1][1]+x[iH2][1])+(1.0-alpha)*(x[idx1][1]);
                                double z_M = 0.5*alpha*(x[iH1][2]+x[iH2][2])+(1.0-alpha)*(x[idx1][2]);
                                double cur_mirror = compute_geometric_mir(x_M, y_M, z_M);
                                x[idx2][2] = 2*cur_mirror-z_M;
                                x[idx2][0] = x_M;
                                x[idx2][1] = y_M;
                                // move the image O as the mirror image of massless site of tip4p
                                v[idx2][0] = v[idx1][0];
                                v[idx2][1] = v[idx1][1];
                                f[idx2][0] = f[idx1][0];
                                f[idx2][1] = f[idx1][1];
                                f[idx2][2] = -f[idx1][2];
                                v[idx2][2] = -v[idx1][2];
                        }
                        else {
                                double z_real = x[idx1][2];
                                double x_real = x[idx1][0];
                                double y_real = x[idx1][1];
                                double cur_mirror = compute_geometric_mir(x_real, y_real, z_real);
                                // Mirror Reflection before force calculation
                                x[idx2][2] = 2*cur_mirror - z_real; 
                                x[idx2][0] = x[idx1][0];
                                x[idx2][1] = x[idx1][1];
                                // Adjust Image atom velocity and force in order to maintain proximity, and prevent elimination from neighbor list (in principle not required)
                                v[idx2][0] = v[idx1][0];
                                v[idx2][1] = v[idx1][1];
                                f[idx2][0] = f[idx1][0];
                                f[idx2][1] = f[idx1][1];
                                f[idx2][2] = -f[idx1][2];
                                v[idx2][2] = -v[idx1][2];
                        }
                        // The image atoms are moved according to DMI-MD theory before force calculation. 
                }
        }
}
     
/* ---------------------------------------------------------------------- */

void FixDmiImd::post_force(int vflag) {
	if (postforce_flag == 0) {
                double **x = atom->x;
                double **f = atom->f;
                double *q = atom->q;
                double *mass = atom->mass;
                int *type = atom->type;
                int *tag = atom->tag;
                int nlocal = atom->nlocal;
                int nghost = atom->nghost;
                int *mask = atom->mask;
                int *mol = atom->molecule;

                double L = 2.930375*std::sqrt(3.0)/2.0;
                double pi = 3.14159265358979323846;
                double k = 2*pi/L;

                double xlo = domain->boxlo[0];
                double xhi = domain->boxhi[0];
                double ylo = domain->boxlo[1];
                double yhi = domain->boxhi[1];

                double Lx = xhi - xlo;
                double Ly = yhi - ylo;

                for (int idx = 0; idx < nlocal; idx++) {
                        if (tag[idx] > imagemin) {
                                continue;
                        }
			if (mask[idx] & groupbit) {
                        	double x_1 = x[idx][0];
                        	double y_1 = x[idx][1];
                        	double z_1 = x[idx][2];
                        	double q_1 = q[idx];
                        	double z_integral = compute_integral(x_1, y_1, z_1);

                        	double x_shift = x_1 - Lx*std::floor((x_1 - xlo)/Lx) - x_ag;
                        	double y_shift = y_1 - Ly*std::floor((y_1 - ylo)/Ly) - y_ag;

                        	// (F_x, F_y) = -del(V)
                        	double d_gamma_x = (-2.0/9.0)*(k*std::sin(k*x_shift)
                                           + 0.5*k*std::sin(0.5*k*x_shift+(std::sqrt(3.0)/2.0)*k*y_shift)
                                           + 0.5*k*std::sin(0.5*k*x_shift-(std::sqrt(3.0)/2.0)*k*y_shift));
                        	double d_gamma_y = (-2.0/9.0)*((std::sqrt(3.0)/2.0)*k*std::sin(0.5*k*x_shift+(std::sqrt(3.0)/2.0)*k*y_shift)
                                           - (std::sqrt(3.0)/2.0)*k*std::sin(0.5*k*x_shift-(std::sqrt(3.0)/2.0)*k*y_shift));

                        	double fx_add = -0.5*q_1*q_1*z_integral*d_gamma_x;
                        	double fy_add = -0.5*q_1*q_1*z_integral*d_gamma_y;

                        	f[idx][0] += 332.06371*fx_add;
                        	f[idx][1] += 332.06371*fy_add;
                	}
		}
	}
	else if (postforce_flag == 1) {
		// postforce_flag = 1 (needs force subtraction)
        	double **x = atom->x;
        	double **f = atom->f;
       	 	double *q = atom->q;
        	double *mass = atom->mass;
        	int *type = atom->type;
        	int *tag = atom->tag;
        	int nlocal = atom->nlocal;
        	int nghost = atom->nghost;
        	int *mask = atom->mask;
        	int *mol = atom->molecule;
        	for (int idx1 = 0; idx1 < nlocal; idx1++) {
                	int molid = mol[idx1];
                	if (tag[idx1] >= imagemin) {
                        	continue;
                	}
                	if (mask[idx1] & groupbit) {
                        	if (tip4p_flag == 1 && type[idx1] == typeO) {
                                	int iH1 = -1;
                                	int iH2 = -1;
                                	iH1 = atom->map(tag[idx1] + 1);
                                	iH2 = atom->map(tag[idx1] + 2);
                                	iH1 = domain->closest_image(idx1,iH1);
                                	iH2 = domain->closest_image(idx1,iH2);
                                	if (iH1 == -1 || iH2 == -1) {
                                        	error->all(FLERR,"TIP4P hydrogen is missing");
                                	}
					double x_M = 0.5*alpha*(x[iH1][0]+x[iH2][0])+(1.0-alpha)*(x[idx1][0]);
                                	double y_M = 0.5*alpha*(x[iH1][1]+x[iH2][1])+(1.0-alpha)*(x[idx1][1]);
                                	double z_M = 0.5*alpha*(x[iH1][2]+x[iH2][2])+(1.0-alpha)*(x[idx1][2]);
					// Calculate ImageM -> realM force by image reconstruction
                                	double f_ex = 0.0, f_ey = 0.0, f_ez = 0.0;
                                	double mir_H1 = compute_geometric_mir(x[iH1][0], x[iH1][1], x[iH1][2]);
                                	double z_i_H1 = 2.0*mir_H1 - x[iH1][2];
                                	double mir_H2 = compute_geometric_mir(x[iH2][0], x[iH2][1], x[iH2][2]);
					double z_i_H2 = 2.0*mir_H2 - x[iH2][2];
                                	double mir_M = compute_geometric_mir(x_M, y_M, z_M);
                                	double R_H1 = std::sqrt((x[iH1][0]-x_M)*(x[iH1][0]-x_M)+(x[iH1][1]-y_M)*(x[iH1][1]-y_M)+(z_i_H1-z_M)*(z_i_H1-z_M));
                                	double R_H2 = std::sqrt((x[iH2][0]-x_M)*(x[iH2][0]-x_M)+(x[iH2][1]-y_M)*(x[iH2][1]-y_M)+(z_i_H2-z_M)*(z_i_H2-z_M));
                                	double R_M = 2.0*(z_M - mir_M);
                                	double q_H = q[iH1];
                                	double q_O = q[idx1];
                                	if (q_H < 0.0 || q_O > 0.0) {
                                        	error->all(FLERR,"q_H < 0 or q_O > 0. check datafile.");
                                	}
					//qqe2r = 332.06371 for units real.
                                	f_ex -= 332.06371*(q_O*q_H*(x_M - x[iH1][0])/(R_H1*R_H1*R_H1) + q_O*q_H*(x_M - x[iH2][0])/(R_H2*R_H2*R_H2));
                                	f_ey -= 332.06371*(q_O*q_H*(y_M - x[iH1][1])/(R_H1*R_H1*R_H1) + q_O*q_H*(y_M - x[iH2][1])/(R_H2*R_H2*R_H2));
                                	f_ez -= 332.06371*(q_O*q_H*(z_M - z_i_H1)/(R_H1*R_H1*R_H1) + q_O*q_H*(z_M - z_i_H2)/(R_H2*R_H2*R_H2) + q_O*q_O/(R_M*R_M));
                                	// Only apply to O 
                                	f[idx1][0] += multiplier*(1.0-alpha)*f_ex;
                                	f[idx1][1] += multiplier*(1.0-alpha)*f_ey;
                                	f[idx1][2] += multiplier*(1.0-alpha)*f_ez;
                        	}
                        	else if (tip4p_flag == 1) {
                                	int iH = -1;
                                	int iO = -1;
                                	int idx_prev = atom->map(tag[idx1]-1);
                                	if (type[idx_prev]==typeO) {
                                        	iH = atom->map(tag[idx1]+1);
                                        	iO = idx_prev;
                                	}
                                	else {
                                        	iO = atom->map(tag[idx1]-2);
                                        	iH = idx_prev;
                                	}
                                	if (iH == -1 || iO == -1) {
                                        	error->all(FLERR,"TIP4P hydrogen cannot find its friends :( ");
                                	}
                                	iH = domain->closest_image(idx1,iH);
                                	iO = domain->closest_image(idx1,iO);
                                	double x_M = 0.5*alpha*(x[iH][0]+x[idx1][0])+(1.0-alpha)*(x[iO][0]);
                                	double y_M = 0.5*alpha*(x[iH][1]+x[idx1][1])+(1.0-alpha)*(x[iO][1]);
                                	double z_M = 0.5*alpha*(x[iH][2]+x[idx1][2])+(1.0-alpha)*(x[iO][2]);
                                	// Compute Force directly applied to H.
					double f_ex = 0.0, f_ey = 0.0, f_ez = 0.0;
                                	double mir_H1 = compute_geometric_mir(x[idx1][0], x[idx1][1], x[idx1][2]);
                                	double z_i_H1 = 2.0*mir_H1 - x[idx1][2];
                                	double mir_H2 = compute_geometric_mir(x[iH][0], x[iH][1], x[iH][2]);
                                	double z_i_H2 = 2.0*mir_H2 - x[iH][2];
                                	double mir_M = compute_geometric_mir(x_M, y_M, z_M);
                                	double z_i_M = 2.0*mir_M - z_M;
                                	double q_H = q[idx1];
                                	double q_O = q[iO];
                                	if (q_H < 0.0 || q_O > 0.0) {
                                        	error->all(FLERR,"q_H < 0 or q_O > 0. check datafile.");
                                	}
                                	double R_H2 = std::sqrt((x[idx1][0]-x[iH][0])*(x[idx1][0]-x[iH][0])+(x[idx1][1]-x[iH][1])*(x[idx1][1]-x[iH][1])+(x[idx1][2]-z_i_H2)*(x[idx1][2]-z_i_H2));
                                	double R_M = std::sqrt((x[idx1][0]-x_M)*(x[idx1][0]-x_M)+(x[idx1][1]-y_M)*(x[idx1][1]-y_M)+(x[idx1][2]-z_i_M)*(x[idx1][2]-z_i_M));
                                	f_ex -= 332.06371*(q_O*q_H*(x[idx1][0]-x_M)/(R_M*R_M*R_M)+q_H*q_H*(x[idx1][0]-x[iH][0])/(R_H2*R_H2*R_H2));
                                	f_ey -= 332.06371*(q_O*q_H*(x[idx1][1]-y_M)/(R_M*R_M*R_M)+q_H*q_H*(x[idx1][1]-x[iH][1])/(R_H2*R_H2*R_H2));
                                	f_ez -= 332.06371*(q_O*q_H*(x[idx1][2]-z_i_M)/(R_M*R_M*R_M)+q_H*q_H*(x[idx1][2]-z_i_H2)/(R_H2*R_H2*R_H2)+q_H*q_H/((x[idx1][2]-z_i_H1)*(x[idx1][2]-z_i_H1)));
                                	f[idx1][0] += multiplier*f_ex;
                                	f[idx1][1] += multiplier*f_ey;
                                	f[idx1][2] += multiplier*f_ez;
                                	// Compute Force due to massless site contribution.
                                	double f_mx = 0.0, f_my = 0.0, f_mz = 0.0;
                                	double R_MH1 = std::sqrt((x[idx1][0]-x_M)*(x[idx1][0]-x_M)+(x[idx1][1]-y_M)*(x[idx1][1]-y_M)+(z_i_H1-z_M)*(z_i_H1-z_M));
                                	double R_MH2 = std::sqrt((x[iH][0]-x_M)*(x[iH][0]-x_M)+(x[iH][1]-y_M)*(x[iH][1]-y_M)+(z_i_H2-z_M)*(z_i_H2-z_M));
                                	f_mx -= 332.06371*(q_O*q_H*(x_M - x[idx1][0])/(R_MH1*R_MH1*R_MH1) + q_O*q_H*(x_M - x[iH][0])/(R_MH2*R_MH2*R_MH2));
                                	f_my -= 332.06371*(q_O*q_H*(y_M - x[idx1][1])/(R_MH1*R_MH1*R_MH1) + q_O*q_H*(y_M - x[iH][1])/(R_MH2*R_MH2*R_MH2));
                                	f_mz -= 332.06371*(q_O*q_H*(z_M - z_i_H1)/(R_MH1*R_MH1*R_MH1)+q_O*q_H*(z_M - z_i_H2)/(R_MH2*R_MH2*R_MH2) + q_O*q_O/((z_M-z_i_M)*(z_M-z_i_M)));
                                	f[idx1][0] += 0.5*alpha*multiplier*f_mx;
                                	f[idx1][1] += 0.5*alpha*multiplier*f_my;
                                	f[idx1][2] += 0.5*alpha*multiplier*f_mz;
                        	}
				else {
					if (tip4p_flag == 1) error->all(FLERR,"TIP4P water but wrong treatment");
					std::unordered_set<int> seen;
                			int molcount = 0;

                			double f_ex = 0.0, f_ey = 0.0, f_ez = 0.0;

                			for (int j = 0; j < nlocal+nghost; j++) {
                        			if (mol[j] == molid) {
                                			if (tag[j] - tag[idx1] > numpermol || tag[idx1] - tag[j] > numpermol) {
                                        			char err_msg[256];
                                        			sprintf(err_msg, "matching not good: atoms belonging to same molecule should possess consecutive ids, tag[j]=%d, tag[idx1]=%d, numpermol=%d, molid=%d", tag[j], tag[idx1], numpermol, molid);
                                        			error->all(FLERR, err_msg);
                                			}
                                			if (seen.find(tag[j]) != seen.end()) {
                                        			continue;
                                			}
                                			seen.insert(tag[j]);

                                			int idx2 = j;
							idx2 = domain->closest_image(idx1,idx2);

                                			double q_1 = q[idx1];
                                			double q_2 = q[idx2];

                                	// Mirror Calculation -> reconstruction of image atom idx2' from real atom idx2
                                			double cur_mirror = 0.0;
                                			cur_mirror = compute_geometric_mir(x[idx2][0], x[idx2][1], x[idx2][2]);
                                			if (cur_mirror == 0.0) error->all(FLERR,"Mirror Calculation Error");

                                			double z_image = 2*cur_mirror - x[idx2][2];
                                			double d_x = x[idx1][0] - x[idx2][0];
                                			double d_y = x[idx1][1] - x[idx2][1];
                                			double d_z = x[idx1][2] - z_image;

                                			double r_sq = d_x * d_x + d_y * d_y + d_z * d_z;
                                			double r = std::sqrt(r_sq);

                                	// sum image forces: q_im = -q_2
                                	// conversion factor: convert to atomic force unit(*(0.52917721092)^2) and use 1Hartree = 627.5095 kcal/mol
                                			f_ex -= (332.06371 * q_1 * q_2 * d_x)/(r_sq * r);
                                			f_ey -= (332.06371 * q_1 * q_2 * d_y)/(r_sq * r);
                                			f_ez -= (332.06371 * q_1 * q_2 * d_z)/(r_sq * r);

                                			molcount ++;

                                			if (molcount == numpermol) {
                                        			break;
                                			}
						}
					}
					if (molcount != numpermol) error->all(FLERR, "Entire atoms of each molecule should be found within local atoms and ghost atoms");
                        		f[idx1][0] += multiplier*f_ex;
					f[idx1][1] += multiplier*f_ey;
					f[idx1][2] += multiplier*f_ez;
				}

                	}
		}
        }
	else {
		error->all(FLERR, "postforce_flag not set");
	}
}

