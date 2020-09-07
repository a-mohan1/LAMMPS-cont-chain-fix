/* header file for fix_gcvv */

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator

------------------------------------------------------------------------- */

#ifndef FIX_GCVV_H
#define FIX_GCVV_H

#include "fix.h"
#include "fftw3.h"

namespace LAMMPS_NS {

class FixGCVV : public Fix {
 public:
  FixGCVV(class LAMMPS *, int, char **);
  ~FixGCVV();
  int setmask();
  void init();
  void setup(int);
  void initial_integrate(int);
  void final_integrate();
  void reset_dt();

  double memory_usage();
  void create_arrays(int);

 private:
  // processor
  int me;
  // number of monomers per chain
  int Nmon;
  // number of chains, number of points per chain
  int nchains, ncoll;
  // number of procs
  int nprocs;
  // chains per proc for fft and time stepping
  int ncperproc, nrem, ncproc;
  // bond type = 1
  int btype;
  // time step size, force coeff in time stepping, drag coeff 
  double dt, dtf, gamma;
  // nondimensional temperature = k_B*T/eps
  double temp;
  // coeffs in time stepping
  double kfac;
  // random force coeff
  double ranfac;
  // real and fourier space coord, force and velocity arrays
  double **xr_loc, **xr_all, **xf_proc;
  double **fr_loc, **fr_all, **ff_proc;
  double **vf_proc;
  // spring constant from BondHarmonic::sprconst(btype) function
  double kspr;
  // RNG pointer
  class RanMars *random;
  // FFTW array and plan
  double *inout;
  fftw_plan fftwp;
  // PI
  double PI;
};

}

#endif
