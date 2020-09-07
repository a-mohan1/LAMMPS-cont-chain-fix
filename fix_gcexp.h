/* header file for fix_gcexp */

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator

------------------------------------------------------------------------- */

#ifndef FIX_GCEXP_H
#define FIX_GCEXP_H

#include "fix.h"
#include "fftw3.h"

namespace LAMMPS_NS {

class FixGCEXP : public Fix {
 public:
  FixGCEXP(class LAMMPS *, int, char **);
  ~FixGCEXP();
  int setmask();
  void init();
  void setup(int);
  void initial_integrate(int);
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
  double gfac1, gfac2, kgfac;
  // random force coeff
  double ranfac;
  // real and fourier space coord and force arrays
  double **xfpproc, **xr_loc, **xr_all, **xf_proc;
  double **fr_loc, **fr_all, **ff_proc;
  // temp var for new coords in fourier space
  double xfn[3];
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
