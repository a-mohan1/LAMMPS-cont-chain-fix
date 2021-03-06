/* fix_gc.cpp
fix for gaussian chains, time stepping in Fourier space
langevin thermostat
1st order (weak) implicit time-stepping
*/

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   
------------------------------------------------------------------------- */

#include "stdio.h"
#include "string.h"
#include "fix_gc.h"
#include "atom.h"
#include "force.h"
#include "bond.h"
#include "update.h"
#include "error.h"
#include "mpi.h"
#include "math.h"
#include "comm.h"
#include "stdlib.h"
#include "random_mars.h"
#include "error.h"
#include "memory.h"
#include "domain.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixGC::FixGC(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

  if (narg < 7) error->all("Illegal fix gc command");
  // arg[0]=fix ID, arg[1]=group ID(all), arg[2]=fix style,
  // arg[3]=gamma, arg[4]=k_B*T/eps, arg[5]=seed
  // arg[6]=Nmon=#monomers per chain
  gamma = atof(arg[3]); // nondimensional drag coeff
  temp = atof(arg[4]); // nondimensional temperature
  int nseed = atoi(arg[5]); // seed
  if(nseed <= 0) error->all("Illegal seed");
  Nmon = atoi(arg[6]);

  /* restart_global=0, fix does not write to restart files, 
     restart() and write_restart() functions not implemented */

  // process
  me = comm->me;

  // Initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp, nseed + me);

  time_integrate = 1; // 1 means fix performs time integration

  /* initial allocation 
     xfpproc = coords in Fourier space at previous time step 
       for chains on this proc
     xr_loc = coords in real space for local points
     xr_all = coords in real space for all points
     xf_proc = coords in Fourier space for points on chains on this proc
     fr_loc = forces in real space for local points
     fr_all = forces in real space for all points
     ff_proc = forces in fourier space for chains on this proc */

  xfpproc = NULL;
  xr_loc = NULL;
  xr_all = NULL;
  xf_proc = NULL;
  fr_loc = NULL;
  fr_all = NULL;
  ff_proc = NULL;

  // PI
  PI = 3.1416;

  create_arrays(int(atom->natoms)); // allocate memory
  // number of chains
  nchains = (int) (atom->natoms - atom->nbonds);
  // number of points per chain (atoms per molecule)
  ncoll = (int) (atom->natoms / nchains);

  // fftw DCT-I
  inout = (double*) fftw_malloc(sizeof(double)*ncoll);
  fftwp = fftw_plan_r2r_1d(ncoll, inout, inout, FFTW_REDFT00, FFTW_MEASURE);
}

/* ---------------------------------------------------------------------- */

FixGC::~FixGC() 
{
  // delete locally stored arrays
  memory->destroy_2d_double_array(xfpproc);
  memory->destroy_2d_double_array(xr_loc);
  memory->destroy_2d_double_array(xr_all);
  memory->destroy_2d_double_array(xf_proc);
  memory->destroy_2d_double_array(fr_loc);
  memory->destroy_2d_double_array(fr_all);
  memory->destroy_2d_double_array(ff_proc);
  // delete random object
  delete random;

  // delete fftw vars
  fftw_destroy_plan(fftwp);
  fftw_free(inout);

}

/* ---------------------------------------------------------------------- */

int FixGC::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  // final_integrate is not needed for this scheme
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGC::init()
{
  dt = update->dt;
  //btype=bond type=1 (only 1 type of bond)
  btype = 1;
  kspr = force->bond->sprconst(btype);
  // time stepping coeffs
  gfac = gamma/2.0*dt - 1.0;
  dtf = dt*dt;
  kgfac1 = 1.0 + gamma/2.0*dt;
  kgfac2 = 2.0*kspr*PI*PI*dtf/Nmon/Nmon;
  ranfac = sqrt(48.0/Nmon*temp*gamma/dt);

  // chains per proc for fft and time stepping
  nprocs = comm->nprocs;
  ncperproc = (int) nchains/nprocs; 
  nrem = nchains%nprocs;
  // procs 0 to nprocs-2 take ncperproc chains
  // proc nprocs-1 takes (ncperproc+nrem) chains
  ncproc = ncperproc;
  if(me == nprocs-1) ncproc += nrem;
}

/*-----------------------------------------------------------------------*/
/* called by verlet::setup() after computing pair forces from initial coords 
   before time stepping
   verlet::setup() is called by run::command() using update->integrate->setup()
   before update->integrate->iterate(nsteps) */

void FixGC::setup(int vflag) 
{
  double **x = atom->x;
  int *tag = atom->tag;
  int *image = atom->image;
  int nlocal = atom->nlocal;
  int i, j;
  double xtemp[3];
  double *xdata, *ydata, *zdata;
  xdata = (double*) fftw_malloc(sizeof(double)*ncoll);
  ydata = (double*) fftw_malloc(sizeof(double)*ncoll);
  zdata = (double*) fftw_malloc(sizeof(double)*ncoll);

  // xr_loc contains coords of local atoms
  // xf_proc = fourier transforms for chains on this proc
  for(i = 0; i < nchains; i++) {
    for(j = 0; j < ncoll; j++) {
      xr_loc[i*ncoll+j][0] = 0.0;
      xr_loc[i*ncoll+j][1] = 0.0;
      xr_loc[i*ncoll+j][2] = 0.0;
      xf_proc[i*ncoll+j][0] = 0.0;
      xf_proc[i*ncoll+j][1] = 0.0;
      xf_proc[i*ncoll+j][2] = 0.0;
    }
  }

  // tag[i] = global id, runs from 1 to nchains*ncoll
  // array indices = 0 to nchains*ncoll-1
  for(i = 0; i < nlocal; i++) {
    // unwrap initial coords
    domain->unmap(x[i], image[i], xtemp);
    xr_loc[tag[i]-1][0] = xtemp[0];
    xr_loc[tag[i]-1][1] = xtemp[1];
    xr_loc[tag[i]-1][2] = xtemp[2];
  }
  
  MPI_Allreduce(&xr_loc[0][0], &xr_all[0][0], 3*ncoll*nchains, 
		MPI_DOUBLE, MPI_SUM, world);

  // xr_all contains coords of all atoms in real space
  // each proc has a copy of xr_all
  // calculate fourier transform
  
  for(i = me*ncperproc; i < me*ncperproc+ncproc; i++) {
    for(j = 0; j < ncoll; j++) {
      xdata[j] = xr_all[i*ncoll+j][0]/2.0/(ncoll-1.0);
      ydata[j] = xr_all[i*ncoll+j][1]/2.0/(ncoll-1.0);
      zdata[j] = xr_all[i*ncoll+j][2]/2.0/(ncoll-1.0);
    }

    // DCT-I
    fftw_execute_r2r(fftwp, xdata, xdata);
    fftw_execute_r2r(fftwp, ydata, ydata);
    fftw_execute_r2r(fftwp, zdata, zdata);
 
    for(j = 1; j < ncoll-1; j++) {
      xf_proc[i*ncoll+j][0] = 2.0*xdata[j];
      xf_proc[i*ncoll+j][1] = 2.0*ydata[j];
      xf_proc[i*ncoll+j][2] = 2.0*zdata[j];
    }   
    j = 0;
    xf_proc[i*ncoll+j][0] = xdata[j];
    xf_proc[i*ncoll+j][1] = ydata[j];
    xf_proc[i*ncoll+j][2] = zdata[j];
    j = ncoll-1;
    xf_proc[i*ncoll+j][0] = xdata[j];
    xf_proc[i*ncoll+j][1] = ydata[j];
    xf_proc[i*ncoll+j][2] = zdata[j];
  }
  
  // now each proc has initial coords in fourier space for its chains
  fftw_free(xdata); fftw_free(ydata); fftw_free(zdata);
 
}

/* ----------------------------------------------------------------------
   mass is defined in input file as 1.0, so is ignored here
   group=all
------------------------------------------------------------------------- */

void FixGC::initial_integrate(int vflag)
{

  double **x = atom->x; // atom positions at start of time step
  /* atom velocities at start of time step, 
     initialized to 0 at start of simulation */
  double **v = atom->v; 
  // forces (pair only, computed by verlet/ contc) at start of time step
  double **f = atom->f; 
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  int *image = atom->image;
  int i, j;
  double xtemp[3];

  // send forces to all procs and calculate fourier transform

  double *fxdat, *fydat, *fzdat;
  fxdat = (double*) fftw_malloc(sizeof(double)*ncoll);
  fydat = (double*) fftw_malloc(sizeof(double)*ncoll);
  fzdat = (double*) fftw_malloc(sizeof(double)*ncoll);
  double *xdata, *ydata, *zdata;
  xdata = (double*) fftw_malloc(sizeof(double)*ncoll);
  ydata = (double*) fftw_malloc(sizeof(double)*ncoll);
  zdata = (double*) fftw_malloc(sizeof(double)*ncoll);

  // fr_loc contains forces on local atoms
  // ff_proc = fourier transforms for chains on this proc
  for(i = 0; i < nchains; i++) {
    for(j = 0; j < ncoll; j++) {
      fr_loc[i*ncoll+j][0] = 0.0;
      fr_loc[i*ncoll+j][1] = 0.0;
      fr_loc[i*ncoll+j][2] = 0.0;
      ff_proc[i*ncoll+j][0] = 0.0;
      ff_proc[i*ncoll+j][1] = 0.0;
      ff_proc[i*ncoll+j][2] = 0.0;
    }
  }
  for(i = 0; i < nlocal; i++) {
    fr_loc[tag[i]-1][0] = f[i][0] * double(Nmon)/double(ncoll-1.0);
    fr_loc[tag[i]-1][1] = f[i][1] * double(Nmon)/double(ncoll-1.0);
    fr_loc[tag[i]-1][2] = f[i][2] * double(Nmon)/double(ncoll-1.0);
  }

  MPI_Allreduce(&fr_loc[0][0], &fr_all[0][0], 3*ncoll*nchains,
		MPI_DOUBLE, MPI_SUM, world);


  // fr_all contains forces on all atoms in real space
  // each proc has a copy of fr_all
  // calculate fourier transform for chains on this proc

  for(i = me*ncperproc; i < me*ncperproc+ncproc; i++) {
    for(j = 0; j < ncoll; j++) {
      fxdat[j] = fr_all[i*ncoll+j][0]/2.0/(ncoll-1.0);
      fydat[j] = fr_all[i*ncoll+j][1]/2.0/(ncoll-1.0);
      fzdat[j] = fr_all[i*ncoll+j][2]/2.0/(ncoll-1.0);
    }

    // DCT-I
    fftw_execute_r2r(fftwp, fxdat, fxdat);
    fftw_execute_r2r(fftwp, fydat, fydat);
    fftw_execute_r2r(fftwp, fzdat, fzdat);

    for(j = 1; j < ncoll-1; j++) {
      ff_proc[i*ncoll+j][0] = 2.0*fxdat[j];
      ff_proc[i*ncoll+j][1] = 2.0*fydat[j];
      ff_proc[i*ncoll+j][2] = 2.0*fzdat[j];
    }
    j = 0;
    ff_proc[i*ncoll+j][0] = fxdat[j];
    ff_proc[i*ncoll+j][1] = fydat[j];
    ff_proc[i*ncoll+j][2] = fzdat[j];
    j = ncoll-1;
    ff_proc[i*ncoll+j][0] = fxdat[j];
    ff_proc[i*ncoll+j][1] = fydat[j];
    ff_proc[i*ncoll+j][2] = fzdat[j];

    // add thermal force for chain i
    for(j = 1; j < ncoll; j++) {
      ff_proc[i*ncoll+j][0] += ranfac*(random->uniform()-0.5);
      ff_proc[i*ncoll+j][1] += ranfac*(random->uniform()-0.5);
      ff_proc[i*ncoll+j][2] += ranfac*(random->uniform()-0.5);
    }
    // j = 0 mode
    j = 0;
    ff_proc[i*ncoll+j][0] += ranfac/sqrt(2.0)*(random->uniform()-0.5);
    ff_proc[i*ncoll+j][1] += ranfac/sqrt(2.0)*(random->uniform()-0.5);
    ff_proc[i*ncoll+j][2] += ranfac/sqrt(2.0)*(random->uniform()-0.5);
  }

  fftw_free(fxdat); fftw_free(fydat); fftw_free(fzdat);

  // time stepping for chains on this proc
  // this proc operates on chains me*ncperproc through me*ncperproc+ncproc-1

  for(i = me*ncperproc; i < me*ncperproc+ncproc; i++) {
   if(update->ntimestep > 1) {
     // j=0 mode
     j = 0;
     xfn[0] = (2.0*xf_proc[i*ncoll+j][0] + gfac*xfpproc[i*ncoll+j][0] + 
	    ff_proc[i*ncoll+j][0]*dtf)/kgfac1;
     xfn[1] = (2.0*xf_proc[i*ncoll+j][1] + gfac*xfpproc[i*ncoll+j][1] +
	    ff_proc[i*ncoll+j][1]*dtf)/kgfac1;
     xfn[2] = (2.0*xf_proc[i*ncoll+j][2] + gfac*xfpproc[i*ncoll+j][2] +
	    ff_proc[i*ncoll+j][2]*dtf)/kgfac1;     
     xfpproc[i*ncoll+j][0] = xf_proc[i*ncoll+j][0];
     xfpproc[i*ncoll+j][1] = xf_proc[i*ncoll+j][1];
     xfpproc[i*ncoll+j][2] = xf_proc[i*ncoll+j][2];
     xf_proc[i*ncoll+j][0] = xfn[0];
     xf_proc[i*ncoll+j][1] = xfn[1];
     xf_proc[i*ncoll+j][2] = xfn[2];

     // remaining modes
     for(j = 1; j < ncoll; j++) {
       xfn[0] = (2.0*xf_proc[i*ncoll+j][0] + gfac*xfpproc[i*ncoll+j][0] +
		 ff_proc[i*ncoll+j][0]*dtf)/(kgfac1 + kgfac2*j*j);
       xfn[1] = (2.0*xf_proc[i*ncoll+j][1] + gfac*xfpproc[i*ncoll+j][1] +
		 ff_proc[i*ncoll+j][1]*dtf)/(kgfac1 + kgfac2*j*j);
       xfn[2] = (2.0*xf_proc[i*ncoll+j][2] + gfac*xfpproc[i*ncoll+j][2] +
		 ff_proc[i*ncoll+j][2]*dtf)/(kgfac1 + kgfac2*j*j);     
       xfpproc[i*ncoll+j][0] = xf_proc[i*ncoll+j][0];
       xfpproc[i*ncoll+j][1] = xf_proc[i*ncoll+j][1];
       xfpproc[i*ncoll+j][2] = xf_proc[i*ncoll+j][2];
       xf_proc[i*ncoll+j][0] = xfn[0];
       xf_proc[i*ncoll+j][1] = xfn[1];
       xf_proc[i*ncoll+j][2] = xfn[2];
     }

   } else { // first time step
     // j=0 mode
     j = 0;
     xfpproc[i*ncoll+j][0] = xf_proc[i*ncoll+j][0];
     xfpproc[i*ncoll+j][1] = xf_proc[i*ncoll+j][1];
     xfpproc[i*ncoll+j][2] = xf_proc[i*ncoll+j][2];
     xf_proc[i*ncoll+j][0] += ff_proc[i*ncoll+j][0]*dtf;
     xf_proc[i*ncoll+j][1] += ff_proc[i*ncoll+j][1]*dtf;
     xf_proc[i*ncoll+j][2] += ff_proc[i*ncoll+j][2]*dtf;
     // remaining modes
     for(j = 1; j < ncoll; j++) {
       xfpproc[i*ncoll+j][0] = xf_proc[i*ncoll+j][0];
       xfpproc[i*ncoll+j][1] = xf_proc[i*ncoll+j][1];
       xfpproc[i*ncoll+j][2] = xf_proc[i*ncoll+j][2];
       xf_proc[i*ncoll+j][0] += (ff_proc[i*ncoll+j][0]*dtf);
       xf_proc[i*ncoll+j][0] /= (1.0 + kgfac2*j*j);
       xf_proc[i*ncoll+j][1] += (ff_proc[i*ncoll+j][1]*dtf);
       xf_proc[i*ncoll+j][1] /= (1.0 + kgfac2*j*j);
       xf_proc[i*ncoll+j][2] += (ff_proc[i*ncoll+j][2]*dtf);
       xf_proc[i*ncoll+j][2] /= (1.0 + kgfac2*j*j);
     }
   }
  }

  /* this proc now has updated values of xf_proc and xfpproc
   for its chains. apply inverse transform to xf_proc to get
   coords in real space for chains on this proc
   and update atom->x, atom->v and atom->image for owned atoms */

  // xr_loc are now coords in real space for chains on this proc

  for(i = 0; i < nchains; i++) {
    for(j = 0; j < ncoll; j++) {
      xr_loc[i*ncoll+j][0] = 0.0;
      xr_loc[i*ncoll+j][1] = 0.0;
      xr_loc[i*ncoll+j][2] = 0.0;
    }
  }

  for(i = me*ncperproc; i < me*ncperproc+ncproc; i++) {
    for(j = 1; j < ncoll-1; j++) {
      xdata[j] = xf_proc[i*ncoll+j][0]/2.0;
      ydata[j] = xf_proc[i*ncoll+j][1]/2.0;
      zdata[j] = xf_proc[i*ncoll+j][2]/2.0;
    }
    j = 0;
    xdata[j] = xf_proc[i*ncoll+j][0];
    ydata[j] = xf_proc[i*ncoll+j][1];
    zdata[j] = xf_proc[i*ncoll+j][2];
    j = ncoll - 1;
    xdata[j] = xf_proc[i*ncoll+j][0];
    ydata[j] = xf_proc[i*ncoll+j][1];
    zdata[j] = xf_proc[i*ncoll+j][2];   

    // DCT-I
    fftw_execute_r2r(fftwp, xdata, xdata);
    fftw_execute_r2r(fftwp, ydata, ydata);
    fftw_execute_r2r(fftwp, zdata, zdata);
 
    for(j = 0; j < ncoll; j++) {
      xr_loc[i*ncoll+j][0] = xdata[j];
      xr_loc[i*ncoll+j][1] = ydata[j];
      xr_loc[i*ncoll+j][2] = zdata[j];
    }   
  }
 
  fftw_free(xdata); fftw_free(ydata); fftw_free(zdata);

  // send real space coords to all procs  
  MPI_Allreduce(&xr_loc[0][0], &xr_all[0][0], 3*ncoll*nchains, 
		MPI_DOUBLE, MPI_SUM, world);
  
  // now each proc has a copy of updated real space coords xr_all
  // update velocities, positions and images of owned atoms

  for(i = 0; i < nlocal; i++) {
    // unwrap old coords before calculating velocity
    domain->unmap(x[i], image[i], xtemp);
    v[i][0] = (xr_all[tag[i]-1][0] - xtemp[0])/dt;
    v[i][1] = (xr_all[tag[i]-1][1] - xtemp[1])/dt;
    v[i][2] = (xr_all[tag[i]-1][2] - xtemp[2])/dt;
    x[i][0] = xr_all[tag[i]-1][0];
    x[i][1] = xr_all[tag[i]-1][1];
    x[i][2] = xr_all[tag[i]-1][2];
  
    // set image to nx,ny,nz=0 since we have unwrapped coords
    // remap into simulation box and adjust image

    image[i] = (512 << 20) | (512 << 10) | 512;
    domain->remap(x[i], image[i]);  
  }
  // end of time step

}

/* ---------------------------------------------------------------------- */
 
void FixGC::reset_dt()
{
  dt = update->dt;
  // time stepping coeffs
  gfac = gamma/2.0*dt - 1.0;
  dtf = dt*dt;
  kgfac1 = 1.0 + gamma/2.0*dt;
  kgfac2 = 2.0*kspr*PI*PI*dtf/Nmon/Nmon;
  ranfac = sqrt(48.0/Nmon*temp*gamma/dt);
}

/* ----------------------------------------------------------------------
   memory usage of local arrays
------------------------------------------------------------------------- */

double FixGC::memory_usage()
{
  double bytes = atom->natoms*3*7 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate arrays
------------------------------------------------------------------------- */

void FixGC::create_arrays(int natoms)
{
  xfpproc =
    memory->create_2d_double_array(natoms,3,"fix_gc:xfpproc");
  xr_loc =
    memory->create_2d_double_array(natoms,3,"fix_gc:xr_loc");
  xr_all =
    memory->create_2d_double_array(natoms,3,"fix_gc:xr_all");
  xf_proc =
    memory->create_2d_double_array(natoms,3,"fix_gc:xf_proc");
  fr_loc =
    memory->create_2d_double_array(natoms,3,"fix_gc:fr_loc");
  fr_all =
    memory->create_2d_double_array(natoms,3,"fix_gc:fr_all");
  ff_proc =
    memory->create_2d_double_array(natoms,3,"fix_gc:ff_proc");
} 

