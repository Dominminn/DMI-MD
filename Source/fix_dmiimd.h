#ifdef FIX_CLASS

FixStyle(dmiimd,FixDmiImd)

#else

#ifndef LMP_FIX_DMIIMD_H
#define LMP_FIX_DMIIMD_H

#include "fix.h"
#include <vector>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class FixDmiImd : public Fix {
	public:
		FixDmiImd(class LAMMPS *, int, char**);
		~FixDmiImd();
		int setmask();
		void init();
		void setup(int);
		void post_integrate();
		void post_force(int);
	private:
		double compute_geometric_mir(double, double, double);
		double compute_integral(double, double, double);
		int group_id;
		int numpermol;
		int spacing;
		int imagemin;
		double multiplier;
		int postforce_flag, tip4p_flag;
		double blen, theta, qdist, alpha;
		int typeO;
		double a_1, b_1, c_1, a_2, b_2, c_2, x_ag, y_ag;
};
}

#endif
#endif
