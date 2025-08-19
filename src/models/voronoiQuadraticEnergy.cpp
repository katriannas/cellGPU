#include "voronoiQuadraticEnergy.h"
#include "voronoiQuadraticEnergy.cuh"
#include "cuda_profiler_api.h"
/*! \file voronoiQuadraticEnergy.cpp */

/*!
\param n number of cells to initialize
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post initializeVoronoiQuadraticEnergy(n,initGPURNcellsG) is called, as is setCellPreferenceUniform(1.0,4.0)
*/
VoronoiQuadraticEnergy::VoronoiQuadraticEnergy(int n, bool reprod, bool usegpu)
{
//    printf("Initializing %i cells with random positions in a square box... \n",n);
Reproducible = reprod;
initializeVoronoiQuadraticEnergy(n, usegpu);
setCellPreferencesUniform(1.0,4.0);
};

/*!
\param n number of cells to initialize
\param A0 set uniform preferred area for all cells
\param P0 set uniform preferred perimeter for all cells
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post initializeVoronoiQuadraticEnergy(n,initGPURNG) is called
*/
VoronoiQuadraticEnergy::VoronoiQuadraticEnergy(int n,double A0, double P0,bool reprod, bool gpu)
{
//    printf("Initializing %i cells with random positions in a square box...\n ",n);
Reproducible = reprod;
initializeVoronoiQuadraticEnergy(n, gpu);
setCellPreferencesUniform(A0,P0);
setv0Dr(0.05,1.0);
};

/*!
\param  n Number of cells to initialized
\post all GPUArrays are set to the correct size, v0 is set to 0.05, Dr is set to 1.0, the
Hilbert sorting period is set to -1 (i.e. off), the moduli are set to KA=KP=1.0, voronoiModelBase is
initialized (initializeVoronoiModelBase(n) gets called), particle exclusions are turned off, and auxiliary
data structures for the topology are set
*/
//take care of all class initialization functions
void VoronoiQuadraticEnergy::initializeVoronoiQuadraticEnergy(int n, bool gpu)
{
initializeVoronoiModelBase(n, gpu);
Timestep = 0;
setDeltaT(0.01);
};

/*!
goes through the process of computing the forces on either the CPU or GPU, either with or without
exclusions, as determined by the flags. Assumes the geometry has NOT yet been computed.
\post the geometry is computed, and force per cell is computed.
*/
void VoronoiQuadraticEnergy::computeForces()
{
if(forcesUpToDate)
return; 
forcesUpToDate = true;
computeGeometry();
if (GPUcompute)
{
ComputeForceSetsGPU();
SumForcesGPU();
}
else
{
for (int ii = 0; ii < Ncells; ++ii)
    computeVoronoiForceCPU(ii);
};
};

/*!
\pre The geoemtry (area and perimeter) has already been calculated
\post calculate the contribution to the net force on every particle from each of its voronoi vertices
via a cuda call
*/
void VoronoiQuadraticEnergy::ComputeForceSetsGPU()
{
computeVoronoiForceSetsGPU();
};

/*!
\pre forceSets are already computed
\post call the right routine to add up forceSets to get the net force per cell
*/
void VoronoiQuadraticEnergy::SumForcesGPU()
{
if(!particleExclusions)
sumForceSets();
else
sumForceSetsWithExclusions();
};

/*!
\pre forceSets are already computed,
\post The forceSets are summed to get the net force per particle via a cuda call
*/
void VoronoiQuadraticEnergy::sumForceSets()
{

ArrayHandle<int> d_nn(neighborNum,access_location::device,access_mode::read);
ArrayHandle<double2> d_forceSets(forceSets,access_location::device,access_mode::read);
ArrayHandle<double2> d_forces(cellForces,access_location::device,access_mode::overwrite);

gpu_sum_force_sets(
	    d_forceSets.data,
	    d_forces.data,
	    d_nn.data,
	    Ncells,n_idx);
};

/*!
\pre forceSets are already computed, some particle exclusions have been defined.
\post The forceSets are summed to get the net force per particle via a cuda call, respecting exclusions
*/
void VoronoiQuadraticEnergy::sumForceSetsWithExclusions()
{

ArrayHandle<int> d_nn(neighborNum,access_location::device,access_mode::read);
ArrayHandle<double2> d_forceSets(forceSets,access_location::device,access_mode::read);
ArrayHandle<double2> d_forces(cellForces,access_location::device,access_mode::overwrite);
ArrayHandle<double2> d_external_forces(external_forces,access_location::device,access_mode::overwrite);
ArrayHandle<int> d_exes(exclusions,access_location::device,access_mode::read);

gpu_sum_force_sets_with_exclusions(
	    d_forceSets.data,
	    d_forces.data,
	    d_external_forces.data,
	    d_exes.data,
	    d_nn.data,
	    Ncells,n_idx);
};

/*!
Calculate the contributions to the net force on particle "i" from each of particle i's voronoi
vertices
*/
void VoronoiQuadraticEnergy::computeVoronoiForceSetsGPU()
{
ArrayHandle<double2> d_p(cellPositions,access_location::device,access_mode::read);
ArrayHandle<double2> d_AP(AreaPeri,access_location::device,access_mode::read);
ArrayHandle<double2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
ArrayHandle<int2> d_delSets(delSets,access_location::device,access_mode::read);
ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
ArrayHandle<double2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);
ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::read);
ArrayHandle<double2> d_vc(voroCur,access_location::device,access_mode::read);
ArrayHandle<double4> d_vln(voroLastNext,access_location::device,access_mode::read);

gpu_force_sets(
	    d_p.data,
	    d_AP.data,
	    d_APpref.data,
	    d_delSets.data,
	    d_delOther.data,
	    d_vc.data,
	    d_vln.data,
	    d_forceSets.data,
	    d_nidx.data,
	    KA,
	    KP,
	    NeighIdxNum,n_idx,*(Box));
};

/*!
\param i The particle index for which to compute the net force, assuming addition tension terms between unlike particles
\post the net force on cell i is computed
*/
void VoronoiQuadraticEnergy::computeVoronoiForceCPU(int i)
{
double Pthreshold = THRESHOLD;

//read in all the data we'll need
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<double2> h_f(cellForces,access_location::host,access_mode::readwrite);
ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);

ArrayHandle<double2> h_external_forces(external_forces,access_location::host,access_mode::overwrite);
ArrayHandle<int> h_exes(exclusions,access_location::host,access_mode::read);

//get Delaunay neighbors of the cell
int neigh = h_nn.data[i];
vector<int> ns(neigh);
for (int nn = 0; nn < neigh; ++nn)
{
ns[nn]=h_n.data[n_idx(nn,i)];
};

//compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
vector<double2> voro(neigh);
vector<Matrix2x2> dhdri(neigh);
Matrix2x2 Id;
double2 circumcent;
double2 rij,rik;
double2 nnextp,nlastp;
double2 rjk;
double2 pi = h_p.data[i];

nlastp = h_p.data[ns[ns.size()-1]];
Box->minDist(nlastp,pi,rij);
for (int nn = 0; nn < neigh;++nn)
{
int id = n_idx(nn,i);
nnextp = h_p.data[ns[nn]];
Box->minDist(nnextp,pi,rik);
voro[nn] = h_v.data[id];
rjk.x =rik.x-rij.x;
rjk.y =rik.y-rij.y;

double2 dbDdri,dgDdri,dDdriOD,z;
double betaD = -dot(rik,rik)*dot(rij,rjk);
double gammaD = dot(rij,rij)*dot(rik,rjk);
double cp = rij.x*rjk.y - rij.y*rjk.x;
double D = 2*cp*cp;

z.x = betaD*rij.x+gammaD*rik.x;
z.y = betaD*rij.y+gammaD*rik.y;

dbDdri.x = 2*dot(rij,rjk)*rik.x+dot(rik,rik)*rjk.x;
dbDdri.y = 2*dot(rij,rjk)*rik.y+dot(rik,rik)*rjk.y;

dgDdri.x = -2*dot(rik,rjk)*rij.x-dot(rij,rij)*rjk.x;
dgDdri.y = -2*dot(rik,rjk)*rij.y-dot(rij,rij)*rjk.y;

dDdriOD.x = (-2.0*rjk.y)/cp;
dDdriOD.y = (2.0*rjk.x)/cp;

dhdri[nn] = Id+1.0/D*(dyad(rij,dbDdri)+dyad(rik,dgDdri)-(betaD+gammaD)*Id-dyad(z,dDdriOD));

rij=rik;
};

double2 vlast,vnext,vother;

//start calculating forces
double2 forceSum;
forceSum.x=0.0;forceSum.y=0.0;

double Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
double Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);

double2 vcur;
vlast = voro[neigh-1];
for(int nn = 0; nn < neigh; ++nn)
{
//first, let's do the self-term, dE_i/dr_i
vcur = voro[nn];
vnext = voro[(nn+1)%neigh];
int baseNeigh = ns[nn];
int other_idx = nn - 1;
if (other_idx < 0) other_idx += neigh;
int otherNeigh = ns[other_idx];

double2 dAidv,dPidv;
dAidv.x = 0.5*(vlast.y-vnext.y);
dAidv.y = 0.5*(vnext.x-vlast.x);

double2 dlast,dnext;
dlast.x = vlast.x-vcur.x;
dlast.y=vlast.y-vcur.y;

double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

dnext.x = vcur.x-vnext.x;
dnext.y = vcur.y-vnext.y;
double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
if(dnnorm < Pthreshold)
    dnnorm = Pthreshold;
if(dlnorm < Pthreshold)
    dlnorm = Pthreshold;
dPidv.x = dlast.x/dlnorm - dnext.x/dnnorm;
dPidv.y = dlast.y/dlnorm - dnext.y/dnnorm;

//
//now let's compute the other terms...first we need to find the third voronoi
//position that v_cur is connected to
//
int neigh2 = h_nn.data[baseNeigh];
int DT_other_idx=-1;
for (int n2 = 0; n2 < neigh2; ++n2)
    {
    int testPoint = h_n.data[n_idx(n2,baseNeigh)];
    if(testPoint == otherNeigh) DT_other_idx = h_n.data[n_idx((n2+1)%neigh2,baseNeigh)];
    };
if(DT_other_idx == otherNeigh || DT_other_idx == baseNeigh || DT_other_idx == -1)
    {
    printf("Triangulation problem %i\n",DT_other_idx);
    throw std::exception();
    };
double2 nl1 = h_p.data[otherNeigh];
double2 nn1 = h_p.data[baseNeigh];
double2 no1 = h_p.data[DT_other_idx];

double2 r1,r2,r3;
Box->minDist(nl1,pi,r1);
Box->minDist(nn1,pi,r2);
Box->minDist(no1,pi,r3);

Circumcenter(r1,r2,r3,vother);

double Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
double Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
double Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
double Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

double2 dAkdv,dPkdv;
dAkdv.x = 0.5*(vnext.y-vother.y);
dAkdv.y = 0.5*(vother.x-vnext.x);

dlast.x = vnext.x-vcur.x;
dlast.y=vnext.y-vcur.y;
dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
dnext.x = vcur.x-vother.x;
dnext.y = vcur.y-vother.y;
dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
if(dnnorm < Pthreshold)
    dnnorm = Pthreshold;
if(dlnorm < Pthreshold)
    dlnorm = Pthreshold;

dPkdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
dPkdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

double2 dAjdv,dPjdv;
dAjdv.x = 0.5*(vother.y-vlast.y);
dAjdv.y = 0.5*(vlast.x-vother.x);

dlast.x = vother.x-vcur.x;
dlast.y=vother.y-vcur.y;
dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
dnext.x = vcur.x-vlast.x;
dnext.y = vcur.y-vlast.y;
dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
if(dnnorm < Pthreshold)
    dnnorm = Pthreshold;
if(dlnorm < Pthreshold)
    dlnorm = Pthreshold;

dPjdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
dPjdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

double2 dEdv;

dEdv.x = 2.0*Adiff*dAidv.x + 2.0*Pdiff*dPidv.x;
dEdv.y = 2.0*Adiff*dAidv.y + 2.0*Pdiff*dPidv.y;
dEdv.x += 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x;
dEdv.y += 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y;
dEdv.x += 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x;
dEdv.y += 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y;

double2 temp = dEdv*dhdri[nn];
forceSum.x += temp.x;
forceSum.y += temp.y;

vlast=vcur;
};

h_f.data[i].x=forceSum.x;
h_f.data[i].y=forceSum.y;
if(particleExclusions)
{
if(h_exes.data[i] != 0)
    {
    h_f.data[i].x = 0.0;
    h_f.data[i].y = 0.0;
    h_external_forces.data[i].x=-forceSum.x;
    h_external_forces.data[i].y=-forceSum.y;
    };
}
};

/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A_i-A_i,0)^2 + K_P(P_i-P_i,0)^2
*/
double VoronoiQuadraticEnergy::computeEnergy()
{
if(!forcesUpToDate)
computeForces();
ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
ArrayHandle<double2> h_APP(AreaPeriPreferences,access_location::host,access_mode::read);
Energy = 0.0;
for (int nn = 0; nn  < Ncells; ++nn)
{
Energy += KA * (h_AP.data[nn].x-h_APP.data[nn].x)*(h_AP.data[nn].x-h_APP.data[nn].x);
Energy += KP * (h_AP.data[nn].y-h_APP.data[nn].y)*(h_AP.data[nn].y-h_APP.data[nn].y);
};
return Energy;
};

/*!
a utility function...output some information assuming the system is uniform
*/
void VoronoiQuadraticEnergy::reportCellInfo()
{
printf("Ncells=%i\tv0=%f\tDr=%f\n",Ncells,v0,Dr);
};

/*!
This function calculates
\sigma_{xy} = 1/Area_{total}*(dE/d\gamma), the normalized change in energy when deforming the box with a strain tensor given by
0   \gamma
0   1
. Notably, this is done by taking analytic derivatives, not by doing a finite-difference computed
by actually deforming the box a bit and recomputing the geometry.
*/
double VoronoiQuadraticEnergy::getSigmaXY()
{
computeGeometry();
double sigmaXY = 0.0;
double Pthreshold = THRESHOLD;

//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

ArrayHandle<double4> h_vln(voroLastNext,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

//compute the contribution from each cell
for (int i = 0; i < Ncells; ++i)
{
double2 pi = h_p.data[i];
//get Delaunay neighbors of the cell
int neigh = h_nn.data[i];
vector<int> ns(neigh);
vector<double2> voro(neigh);
vector<double2> voroLast(neigh);
vector<double2> voroNext(neigh);
for (int nn = 0; nn < neigh; ++nn)
    {
    //There is an indexing offset in the relevant voroCur, LastNext data structures between GPU and CPU
    int id = n_idx(nn,i);
    ns[nn]=h_n.data[id];
    int newIndex = nn;
    if(GPUcompute)
	{
	newIndex = nn+1;
	if(newIndex == neigh)
	    newIndex = 0;
	id =n_idx(newIndex,i);
	}

    voro[nn] = h_v.data[id];
    }
for (int nn = 0; nn < neigh; ++nn)
    {
    //There is an indexing offset in the relevant voroCur, LastNext data structures between GPU and CPU
    int loopOffset = 0;
    if(GPUcompute)
	loopOffset = 1;
    int newIndex = nn + loopOffset;
    if(newIndex == neigh)
	newIndex = 0;
    int id = n_idx(newIndex,i);

    if(!GPUcompute)
	{
	voroNext[nn] = voro[(newIndex+1)%neigh];
	if(newIndex>0)
	    voroLast[nn] = voro[newIndex-1];
	else
	    voroLast[nn] = voro[neigh-1];
	}
    else
	{
	voroLast[nn].x = h_vln.data[id].x;
	voroLast[nn].y = h_vln.data[id].y;
	int id2;
	if (newIndex+1 == neigh)
	    id2 = n_idx(0,i);
	else
	    id2 = n_idx(newIndex+1,i);
	voroNext[nn].x = h_vln.data[id].z;
	voroNext[nn].y = h_vln.data[id].w;
	};
    };
//loop through the Delaunay neighbors, computing dA/d\gamma and dP/d\gamma
double2 rij,rik,dhdg;
double2 nnextp,nlastp;
double2 vlast,vnext,vcur;
nlastp = h_p.data[ns[neigh-1]];
Box->minDist(nlastp,pi,rij);
double Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
double Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);
double dAdg = 0.0;
double dPdg = 0.0;
vlast = voro[neigh-1];
double2 dAidv,dPidv;
double2 dlast,dnext;
for (int nn = 0; nn < neigh; ++nn)
    {
    vlast = voroLast[nn];
    vcur = voro[nn];
    vnext = voroNext[nn];
    nnextp = h_p.data[ns[nn]];
    Box->minDist(nnextp,pi,rik);
    getdhdgamma(dhdg,rij,rik);

    //get area and perimeter derivatives from force calculation
    //note that in the force calculation we adopted a sign convention to avoid computing the
    //final minus sign in f= - \nabla E
    //We'll compensate by writing sigmaXY -= (blah blah) instead of the more natural +=
    dAidv.x = 0.5*(vlast.y-vnext.y);
    dAidv.y = 0.5*(vnext.x-vlast.x);
    dlast.x = vlast.x-vcur.x;
    dlast.y=vlast.y-vcur.y;
    double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    if(dnnorm < Pthreshold)
	    dnnorm = Pthreshold;
    if(dlnorm < Pthreshold)
	    dlnorm = Pthreshold;
    dPidv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPidv.y = dlast.y/dlnorm - dnext.y/dnnorm;

    dAdg += dot(dAidv,dhdg);
    dPdg += dot(dPidv,dhdg);

    rij=rik;
    vlast=vcur;
    };
sigmaXY -= 2.0*Adiff*dAdg + 2.0*Pdiff*dPdg;
};

double b1,b2,b3,b4;
Box->getBoxDims(b1,b2,b3,b4);
double area = b1*b4;
return sigmaXY/area;
};

/*!
This function calculates
\sigma_{xy} = 1/Area_{total}*(dE/d\gamma), the normalized change in energy when deforming the box with a strain tensor given by
0   \gamma
0   1
. Notably, this is done by taking analytic derivatives, not by doing a finite-difference computed
by actually deforming the box a bit and recomputing the geometry.
*/
double VoronoiQuadraticEnergy::getSigmaXY(vector<double> &sigmai)
{
computeGeometry();
sigmai.reserve(Ncells);
double sigmaXY = 0.0;
double Pthreshold = THRESHOLD;

//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

ArrayHandle<double4> h_vln(voroLastNext,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

//compute the contribution from each cell
for (int i = 0; i < Ncells; ++i)
{
double2 pi = h_p.data[i];
//get Delaunay neighbors of the cell
int neigh = h_nn.data[i];
vector<int> ns(neigh);
vector<double2> voro(neigh);
for (int nn = 0; nn < neigh; ++nn)
    {
    ns[nn]=h_n.data[n_idx(nn,i)];
    int id = n_idx(nn,i);
    voro[nn] = h_v.data[id];
    };
//loop through the Delaunay neighbors, computing dA/d\gamma and dP/d\gamma
double2 rij,rik,dhdg;
double2 nnextp,nlastp;
double2 vlast,vnext,vcur;
nlastp = h_p.data[ns[ns.size()-1]]; 
Box->minDist(nlastp,pi,rij);
double Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
double Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);
double dAdg = 0.0;
double dPdg = 0.0;
vlast = voro[neigh-1];
for (int nn = 0; nn < neigh; ++nn)
    {
    vcur = voro[nn];
    vnext = voro[(nn+1)%neigh];
    nnextp = h_p.data[ns[nn]];
    Box->minDist(nnextp,pi,rik);
    getdhdgamma(dhdg,rij,rik);

    //get area and perimeter derivatives from force calculation
    //note that in the force calculation we adopted a sign convention to avoid computing the
    //final minus sign in f= - \nabla E
    //We'll compensate by writing sigmaXY -= (blah blah) instead of the more natural +=
    double2 dAidv,dPidv;
    dAidv.x = 0.5*(vlast.y-vnext.y);
    dAidv.y = 0.5*(vnext.x-vlast.x);
    double2 dlast,dnext;
    dlast.x = vlast.x-vcur.x;
    dlast.y=vlast.y-vcur.y;
    double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    if(dnnorm < Pthreshold)
	    dnnorm = Pthreshold;
    if(dlnorm < Pthreshold)
	    dlnorm = Pthreshold;
    dPidv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPidv.y = dlast.y/dlnorm - dnext.y/dnnorm;

    dAdg += dot(dAidv,dhdg);
    dPdg += dot(dPidv,dhdg);

    rij=rik;
    vlast=vcur;
    };
sigmaXY -= 2.0*Adiff*dAdg + 2.0*Pdiff*dPdg;
sigmai.push_back(-2.0*Adiff*dAdg - 2.0*Pdiff*dPdg);
};

double b1,b2,b3,b4;
Box->getBoxDims(b1,b2,b3,b4);
double area = b1*b4;
return sigmaXY/area;
};

/*!
\param rcs a vector of (row,col) locations
\param vals a vector of the corresponding value of the dynamical matrix
*/




void VoronoiQuadraticEnergy::getDynMatEntries(vector<int2> &rcs, vector<double> &vals,double unstress, double stress)
{
printf("evaluating dynamical matrix\n");
ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);

neighborType nt0 = neighborType::self;
neighborType nt1 = neighborType::first;
neighborType nt2 = neighborType::second;
rcs.reserve(10*Ncells);
vals.reserve(10*Ncells);
Matrix2x2 d2E;
int C1x, C1y,C2x,C2y;
int2 loc;
for (int cell = 0; cell < Ncells; ++cell)
{
vector<int> firstNeighs, secondNeighs;
firstNeighs.reserve(6);
secondNeighs.reserve(15);
//always calculate the self term
d2E = d2Edridrj(cell,cell,nt0,unstress,stress);
C1x = 2*cell;
C1y = 2*cell+1;
loc.x= C1x; loc.y = C1x;
rcs.push_back(loc);
vals.push_back(d2E.x11);
loc.x= C1x; loc.y = C1y;
rcs.push_back(loc);
vals.push_back(d2E.x12);
loc.x= C1y; loc.y = C1x;
rcs.push_back(loc);
vals.push_back(d2E.x21);
loc.x= C1y; loc.y = C1y;
rcs.push_back(loc);
vals.push_back(d2E.x22);

//how many neighbors does cell i have?
int neigh = h_nn.data[cell];
vector<int> ns(neigh);
for (int nn = 0; nn < neigh; ++nn)
    {
    ns[nn] = h_n.data[n_idx(nn,cell)];
    firstNeighs.push_back(ns[nn]);
    };
//find the second neighbors
int lastCell = ns[neigh-2];
int curCell = ns[neigh-1];
for (int nn = 0; nn < neigh; ++nn)
    {
    int nextCell = ns[nn];

    int neigh2 = h_nn.data[curCell];
    for (int n2 = 0; n2 < neigh2; ++n2)
	{
	int potentialNeighbor = h_n.data[n_idx(n2,curCell)];
	if (potentialNeighbor != cell && potentialNeighbor != lastCell && potentialNeighbor != nextCell)
	    secondNeighs.push_back(potentialNeighbor);
	};
    lastCell = curCell;
    curCell = nextCell;
    };

//evaluate partial derivatives w/r/t first and second neighbors, but only once each
//(i.e., respect equality of mixed partials)
for (int ff = 0; ff < firstNeighs.size(); ++ff)
    {
    int cellG = firstNeighs[ff];
    //if cellG > cell, calculate those entries and add to vectors
    if (cellG > cell)
	{
	d2E = d2Edridrj(cell,cellG,nt1,unstress,stress);
	C2x = 2*cellG;
	C2y = 2*cellG+1;
	loc.x= C1x; loc.y = C2x;
	rcs.push_back(loc);
	vals.push_back(d2E.x11);
	loc.x= C1x; loc.y = C2y;
	rcs.push_back(loc);
	vals.push_back(d2E.x12);
	loc.x= C1y; loc.y = C2x;
	rcs.push_back(loc);
	vals.push_back(d2E.x21);
	loc.x= C1y; loc.y = C2y;
	rcs.push_back(loc);
	vals.push_back(d2E.x22);
	};
    };
for (int ss = 0; ss < secondNeighs.size(); ++ss)
    {
    int cellD = secondNeighs[ss];
    //if cellD > cell, calculate those entries and add to vectors
    if (cellD > cell)
	{
	d2E = d2Edridrj(cell,cellD,nt2,unstress,stress);
	C2x = 2*cellD;
	C2y = 2*cellD+1;
	loc.x= C1x; loc.y = C2x;
	rcs.push_back(loc);
	vals.push_back(d2E.x11);
	loc.x= C1x; loc.y = C2y;
	rcs.push_back(loc);
	vals.push_back(d2E.x12);
	loc.x= C1y; loc.y = C2x;
	rcs.push_back(loc);
	vals.push_back(d2E.x21);
	loc.x= C1y; loc.y = C2y;
	rcs.push_back(loc);
	vals.push_back(d2E.x22);
	};
    };

};//end loop over cells
printf("finished building dynamical Matrix\n");
};

/*!
\param i The index of cell i
\param j The index of cell j
\pre Requires that computeGeometry is current
The goal is to return a matrix (x11,x12,x21,x22) with
x11 = d^2 / dr_{i,x} dr_{j,x}
x12 = d^2 / dr_{i,x} dr_{j,y}
x21 = d^2 / dr_{i,y} dr_{j,x}
x22 = d^2 / dr_{i,y} dr_{j,y}
*/
Matrix2x2 VoronoiQuadraticEnergy::d2Edridrj(int i, int j, neighborType neighbor,double unstress, double stress)
{
Matrix2x2  answer;
answer.x11 = 0.0; answer.x12=0.0; answer.x21=0.0;answer.x22=0.0;
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::readwrite);
ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

//how many neighbors does cell i have?
int neigh = h_nn.data[i];
vector<int> ns(neigh);
for (int nn = 0; nn < neigh; ++nn)
{
ns[nn] = h_n.data[n_idx(nn,i)];
};
//the saved voronoi positions
double2 vlast, vcur,vnext;
Matrix2x2 zeroMat(0.0,0.0,0.0,0.0);
//the cell indices
int cellG, cellB,cellGp1,cellBm1;
cellB = ns[neigh-1];
cellBm1 = ns[neigh-2];
vlast = h_v.data[n_idx(neigh-1,i)];
double dEdA = 2*KA*(h_AP.data[i].x - h_APpref.data[i].x);
double dEdP = 2*KP*(h_AP.data[i].y - h_APpref.data[i].y);
double2 dAadrj = dAidrj(i,j);
double2 dPadrj = dPidrj(i,j);

//For debugging, multiply all of the area or perimeter terms by the values here
double area = 1.0;
double peri = 1.0;

answer += area*unstress*2.0*KA*dyad(dAidrj(i,i),dAadrj);
answer += peri*unstress*2.0*KP*dyad(dPidrj(i,i),dPadrj);
for (int vv = 0; vv < neigh; ++vv)
{
cellG = ns[vv];
if (vv+1 == neigh)
    cellGp1 = ns[0];
else
    cellGp1 = ns[vv+1];

//What is the index and relative position of cell delta (which forms a vertex with gamma and beta connect by an edge to v_i)?
int neigh2 = h_nn.data[cellG];
int cellD=-1;
for (int n2 = 0; n2 < neigh2; ++n2)
    {
    int testPoint = h_n.data[n_idx(n2,cellG)];
    if(testPoint == cellB) cellD = h_n.data[n_idx((n2+1)%neigh2,cellG)];
    };
if(cellD == cellB || cellD  == cellG || cellD == -1)
    {
    printf("Triangulation problem %i\n",cellD);
    throw std::exception();
    };

double2 rB,rG;
Box->minDist(h_p.data[cellB],h_p.data[i],rB);
Box->minDist(h_p.data[cellG],h_p.data[i],rG);
double2 rD;
Box->minDist(h_p.data[cellD],h_p.data[i],rD);
double2 vother;
Circumcenter(rB,rG,rD,vother);

vcur = h_v.data[n_idx(vv,i)];
vnext = h_v.data[n_idx((vv+1)%neigh,i)];

Matrix2x2 dvidri = dHdri(h_p.data[i],h_p.data[cellB],h_p.data[cellG]);
Matrix2x2 dvidrj(0.0,0.0,0.0,0.0);
Matrix2x2 dvip1drj(0.0,0.0,0.0,0.0);
Matrix2x2 dvim1drj(0.0,0.0,0.0,0.0);
Matrix2x2 dvodrj(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrix(0.0,0.0,0.0,0.0);
vector<double> d2vidridrj(8,0.0);
if (neighbor ==neighborType::second)
    {
    if (j == cellD)
	dvodrj = dHdri(h_p.data[cellD],h_p.data[cellG],h_p.data[cellB]);
    };
if (neighbor == neighborType::self)
    {
    dvidrj = dvidri;
    dvip1drj = dHdri(h_p.data[i],h_p.data[cellGp1],h_p.data[cellG]);
    dvim1drj = dHdri(h_p.data[i],h_p.data[cellBm1],h_p.data[cellB]);
    d2vidridrj = d2Hdridrj(rB,rG,1);
    };
if (neighbor == neighborType::first)
    {
    if (j == cellG)
	{
	dvidrj = dHdri(h_p.data[cellG],h_p.data[i],h_p.data[cellB]);
	dvip1drj = dHdri(h_p.data[cellG],h_p.data[i],h_p.data[cellGp1]);
	dvodrj = dHdri(h_p.data[cellG],h_p.data[cellD],h_p.data[cellB]);
	d2vidridrj = d2Hdridrj(rG,rB,2);
	};
    if (j == cellB)
	{
	dvidrj = dHdri(h_p.data[cellB],h_p.data[i],h_p.data[cellG]);
	dvim1drj = dHdri(h_p.data[cellB],h_p.data[i],h_p.data[cellBm1]);
	dvodrj = dHdri(h_p.data[cellB],h_p.data[cellD],h_p.data[cellG]);
	d2vidridrj = d2Hdridrj(rB,rG,2);
	};
    if (j == cellBm1)
	dvim1drj = dHdri(h_p.data[cellBm1],h_p.data[i],h_p.data[cellB]);
    if (j == cellGp1)
	dvip1drj = dHdri(h_p.data[cellGp1],h_p.data[i],h_p.data[cellG]);
    };

//
//cell alpha terms
//
//Area part
double2 dAdv;
dAdv.x = 0.5*(vnext.y-vlast.y);
dAdv.y = 0.5*(vlast.x-vnext.x);
//first of three area terms... now done as a simple dyadic product outside the loop
//answer += 2.*KA*dyad(dAdv*dvidri,dAadrj);

//second of three area terms
Matrix2x2 d2Advidrj; //Get in form M_{rb, psi}
d2Advidrj = d2Areadvdr(dvip1drj,dvim1drj);
tempMatrix=d2Advidrj*dvidri;
tempMatrix.transpose();
answer += area*stress*dEdA*(tempMatrix);

//third of three area terms
tempMatrix.x11 =dAdv.x*d2vidridrj[0]+dAdv.y*d2vidridrj[1];
tempMatrix.x21 =dAdv.x*d2vidridrj[2]+dAdv.y*d2vidridrj[3];
tempMatrix.x12 =dAdv.x*d2vidridrj[4]+dAdv.y*d2vidridrj[5];
tempMatrix.x22 =dAdv.x*d2vidridrj[6]+dAdv.y*d2vidridrj[7];
answer += area*stress*dEdA*tempMatrix;

//perimeter part
double2 dPdv;
double2 dlast,dnext;
dlast.x = -vlast.x+vcur.x;
dlast.y = -vlast.y+vcur.y;
double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
dnext.x = -vcur.x+vnext.x;
dnext.y = -vcur.y+vnext.y;
double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

//first of three peri terms...it's a dyadic product outside the loop
//second of three peri terms
Matrix2x2 d2Pdvidrj; //Get in form M_{rb, psi}
d2Pdvidrj = d2Peridvdr(dvidrj,dvim1drj,dvip1drj,vlast,vcur,vnext);
tempMatrix=d2Pdvidrj*dvidri;
tempMatrix.transpose();
answer += peri*stress*dEdP*(tempMatrix);
//third of three peri terms
tempMatrix.x11 =dPdv.x*d2vidridrj[0]+dPdv.y*d2vidridrj[1];
tempMatrix.x21 =dPdv.x*d2vidridrj[2]+dPdv.y*d2vidridrj[3];
tempMatrix.x12 =dPdv.x*d2vidridrj[4]+dPdv.y*d2vidridrj[5];
tempMatrix.x22 =dPdv.x*d2vidridrj[6]+dPdv.y*d2vidridrj[7];
answer += peri*stress*dEdP*tempMatrix;

//now we compute terms related to cells gamma and beta

//cell gamma terms
double dEGdP = 2.0*KP*(h_AP.data[cellG].y - h_APpref.data[cellG].y);
double dEGdA = 2.0*KA*(h_AP.data[cellG].x  - h_APpref.data[cellG].x);
//area part
double2 dAGdv;
dAGdv.x = -0.5*(vnext.y-vother.y);
dAGdv.y = -0.5*(vother.x-vnext.x);
double2 dAGdrj = dAidrj(cellG,j);
//first term
answer += area*unstress*2.*KA*dyad(dAGdv*dvidri,dAGdrj);
//second term
d2Advidrj=d2Areadvdr(dvodrj,dvip1drj);
tempMatrix=d2Advidrj*dvidri;
tempMatrix.transpose();
answer += area*stress*dEGdA*tempMatrix;
//third term
tempMatrix.x11 =dAGdv.x*d2vidridrj[0]+dAGdv.y*d2vidridrj[1];
tempMatrix.x21 =dAGdv.x*d2vidridrj[2]+dAGdv.y*d2vidridrj[3];
tempMatrix.x12 =dAGdv.x*d2vidridrj[4]+dAGdv.y*d2vidridrj[5];
tempMatrix.x22 =dAGdv.x*d2vidridrj[6]+dAGdv.y*d2vidridrj[7];
answer += area*stress*dEGdA*tempMatrix;


//perimeter part
double2 dPGdv;
dlast.x = -vnext.x+vcur.x;
dlast.y = -vnext.y+vcur.y;
dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
dnext.x = -vcur.x+vother.x;
dnext.y = -vcur.y+vother.y;
dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
dPGdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
dPGdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

//first term
double2 dPGdrj = dPidrj(cellG,j);
answer += peri*unstress*2.*KP*dyad(dPGdv*dvidri,dPGdrj);
//second term
d2Pdvidrj = d2Peridvdr(dvidrj,dvip1drj,dvodrj,vnext,vcur,vother);
tempMatrix=d2Pdvidrj*dvidri;
tempMatrix.transpose();
answer += peri*stress*dEGdP*(tempMatrix);
//third of three peri terms
tempMatrix.x11 =dPGdv.x*d2vidridrj[0]+dPGdv.y*d2vidridrj[1];
tempMatrix.x21 =dPGdv.x*d2vidridrj[2]+dPGdv.y*d2vidridrj[3];
tempMatrix.x12 =dPGdv.x*d2vidridrj[4]+dPGdv.y*d2vidridrj[5];
tempMatrix.x22 =dPGdv.x*d2vidridrj[6]+dPGdv.y*d2vidridrj[7];
answer += peri*stress*dEGdP*tempMatrix;

//cell beta terms
double dEBdP = 2.0*KP*(h_AP.data[cellB].y - h_APpref.data[cellB].y);
double dEBdA = 2.0*KA*(h_AP.data[cellB].x - h_APpref.data[cellB].x);
//
//area terms
double2 dABdv;
dABdv.x = 0.5*(vlast.y-vother.y);
dABdv.y = 0.5*(vother.x-vlast.x);
double2 dABdrj = dAidrj(cellB,j);

//first term
answer += area*unstress*2.*KA*dyad(dABdv*dvidri,dABdrj);
//second term
d2Advidrj=d2Areadvdr(dvim1drj,dvodrj);
tempMatrix=d2Advidrj*dvidri;
tempMatrix.transpose();
answer += area*stress*dEBdA*tempMatrix;
//third term
tempMatrix.x11 =dABdv.x*d2vidridrj[0]+dABdv.y*d2vidridrj[1];
tempMatrix.x21 =dABdv.x*d2vidridrj[2]+dABdv.y*d2vidridrj[3];
tempMatrix.x12 =dABdv.x*d2vidridrj[4]+dABdv.y*d2vidridrj[5];
tempMatrix.x22 =dABdv.x*d2vidridrj[6]+dABdv.y*d2vidridrj[7];
answer += area*stress*dEBdA*tempMatrix;


//peri terms
double2 dPBdv;
dlast.x = -vother.x+vcur.x;
dlast.y = -vother.y+vcur.y;
dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
dnext.x = -vcur.x+vlast.x;
dnext.y = -vcur.y+vlast.y;
dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
dPBdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
dPBdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

//first term
double2 dPBdrj = dPidrj(cellB,j);
answer += peri*unstress*2.*KP*dyad(dPBdv*dvidri,dPBdrj);
//second term
d2Pdvidrj = d2Peridvdr(dvidrj,dvodrj,dvim1drj,vother,vcur,vlast);
tempMatrix=d2Pdvidrj*dvidri;
tempMatrix.transpose();
answer += peri*stress*dEBdP*(tempMatrix);
//third of three peri terms
tempMatrix.x11 =dPBdv.x*d2vidridrj[0]+dPBdv.y*d2vidridrj[1];
tempMatrix.x21 =dPBdv.x*d2vidridrj[2]+dPBdv.y*d2vidridrj[3];
tempMatrix.x12 =dPBdv.x*d2vidridrj[4]+dPBdv.y*d2vidridrj[5];
tempMatrix.x22 =dPBdv.x*d2vidridrj[6]+dPBdv.y*d2vidridrj[7];
answer += peri*stress*dEBdP*tempMatrix;

//update the vertices and cell indices for the next loop
vlast=vcur;
cellBm1=cellB;
cellB=cellG;
}; //that was gross

return answer;
};


/*!
\param i The index of cell i
\param nn The index of vertex of cell i
*/

double2 VoronoiQuadraticEnergy::deidHn(int i,int nn)
{
double2 answer;

//read in the needed data
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

n_idx = Index2D(neighMax,Ncells);
double P = h_AP.data[i].y; 
double A = h_AP.data[i].x;
double PA = h_APpref.data[i].x;
double PP = h_APpref.data[i].y;

int ilast = nn - 1;
if(ilast == -1){
    ilast = h_nn.data[i] - 1;
}
int inext = (nn + 1)%(h_nn.data[i]);

//From here i refers to the nth vertex
double hix = h_v.data[n_idx(nn,i)].x;
double hiy = h_v.data[n_idx(nn,i)].y;
double hiLastx = h_v.data[n_idx(ilast,i)].x;
double hiLasty = h_v.data[n_idx(ilast,i)].y;
double hiNextx = h_v.data[n_idx(inext,i)].x;
double hiNexty = h_v.data[n_idx(inext,i)].y;

answer.x = (-hiLasty + hiNexty)*KA*(A - PA) + 2*KP*(P - PP)*((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + (-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)));
answer.y = (-hiLastx + hiNextx)*KA*(-A + PA) + 2*KP*(P - PP)*((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + (-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)));

return answer;    
}


/*!
\param i The index of cell i
\param nn The index of vertex of cell i
\param j The index of vertex of cell i
We first take the derivative w/r/t vertex nn and the take the second derivative w/r/t vertex j
The goal is to return a matrix (x11,x12,x21,x22) with
x11 = d^2 ei / dh_{n,x} dh_{j,x}
x12 = d^2 ei / dh_{n,x} dh_{j,y}
x21 = d^2 ei / dh_{n,y} dh_{j,x}
x22 = d^2 ei / dh_{n,y} dh_{j,y}
*/

Matrix2x2 VoronoiQuadraticEnergy::d2eidHndHj(int i, int nn, int j)
{
Matrix2x2 answer;

//read in the needed data
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

n_idx = Index2D(neighMax,Ncells);
double P = h_AP.data[i].y; 
double A = h_AP.data[i].x;
double PA = h_APpref.data[i].x;
double PP = h_APpref.data[i].y;

//From here i in hix,hiy... refers to the nth vertex
int ilast = nn - 1;
if(ilast == -1){
    ilast = h_nn.data[i] - 1;
}
int inext = (nn + 1)%(h_nn.data[i]);
int jlast = j - 1;
if(jlast == -1){
    jlast = h_nn.data[i] - 1;
}
int jnext = (j + 1)%(h_nn.data[i]);

double hix = h_v.data[n_idx(nn,i)].x;
double hiy = h_v.data[n_idx(nn,i)].y;
double hiLastx = h_v.data[n_idx(ilast,i)].x;
double hiLasty = h_v.data[n_idx(ilast,i)].y;
double hiNextx = h_v.data[n_idx(inext,i)].x;
double hiNexty = h_v.data[n_idx(inext,i)].y;

double hjx = h_v.data[n_idx(j,i)].x;
double hjy = h_v.data[n_idx(j,i)].y;
double hjLastx = h_v.data[n_idx(jlast,i)].x;
double hjLasty = h_v.data[n_idx(jlast,i)].y;
double hjNextx = h_v.data[n_idx(jnext,i)].x;
double hjNexty = h_v.data[n_idx(jnext,i)].y;

if (j == nn) {
//when we take the derivative w/r/t the same vertice twice
answer.x11 = 2*(KA*((-0.5*hiLasty + 0.5*hiNexty)*(-0.5*hiLasty + 0.5*hiNexty)) + 
	     KP*(P - PP)*(-(((-hiLastx + hix)*(-hiLastx + hix))/(sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)))) + 
	     1/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) - 
	     ((-hiNextx + hix)*(-hiNextx + hix))/(sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))) + 
	     1/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))) + 
	     KP*(((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	     (-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	     ((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	     (-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))));

answer.x12 = 2*((0.5*hiLastx - 0.5*hiNextx)*(-0.5*hiLasty + 0.5*hiNexty)*KA + 
	    KP*(P - PP)*(-(((-hiLastx + hix)*(-hiLasty + hiy))/(sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)))) - 
	    ((-hiNextx + hix)*(-hiNexty + hiy))/(sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))) + 
	    KP*((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	    ((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))));

answer.x21 = 2*((0.5*hiLastx - 0.5*hiNextx)*(-0.5*hiLasty + 0.5*hiNexty)*KA + 
	    KP*(P - PP)*(-(((-hiLastx + hix)*(-hiLasty + hiy))/(sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)))) - 
	    ((-hiNextx + hix)*(-hiNexty + hiy))/(sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))) + 
	    KP*((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	    ((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))));

answer.x22 = 2*(KA*((-0.5*hiLastx + 0.5*hiNextx)*(-0.5*hiLastx + 0.5*hiNextx)) + 
	    KP*(P - PP)*(-(((-hiLasty + hiy)*(-hiLasty + hiy))/(sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy))*sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)))) + 
	    1/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) - 
	    ((-hiNexty + hiy)*(-hiNexty + hiy))/(sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))*sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))) + 
	    1/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy))) + 
	    KP*(((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	    ((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))));
}
else if (j == inext) {
//j is the next vertex of n
answer.x11 = 2*((-0.5*hiy + 0.5*hjNexty)*(-0.5*hiLasty + 0.5*hjy)*KA - 
	    (KP*(P - PP)*((hiy - hjy)*(hiy - hjy)))/(sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + 
	    KP*((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (hix - hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hix + hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjNextx + hjx)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy))));

answer.x12 = 2*(0.5*hix - 0.5*hjNextx)*(-0.5*hiLasty + 0.5*hjy)*KA - KA*(-A + PA) + 
	    (2*(hix - hjx)*(hiy - hjy)*KP*(P - PP))/(sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + 
	    2*KP*((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (hix - hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hiy + hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjNexty + hjy)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy)));

answer.x21 = 2*(-0.5*hiy + 0.5*hjNexty)*(0.5*hiLastx - 0.5*hjx)*KA + KA*(-A + PA) + 
	    (2*(hix - hjx)*(hiy - hjy)*KP*(P - PP))/(sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + 
	    2*KP*((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (hiy - hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hix + hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjNextx + hjx)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy)));

answer.x22 = 2*((0.5*hix - 0.5*hjNextx)*(0.5*hiLastx - 0.5*hjx)*KA - 
	    (KP*(P - PP)*((hix - hjx)*(hix - hjx)))/(sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + 
	    KP*((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (hiy - hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hiy + hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjNexty + hjy)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy))));
}
else if (j == ilast) {
//j is the last vertex of n
answer.x11 = 2*((0.5*hiy - 0.5*hjLasty)*(0.5*hiNexty - 0.5*hjy)*KA - (KP*(P - PP)*((hiy - hjy)*(hiy - hjy)))/
	    (sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + KP*((-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)) + 
	    (hix - hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hix + hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjLastx + hjx)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy))));

answer.x12 = 2*(-0.5*hix + 0.5*hjLastx)*(0.5*hiNexty - 0.5*hjy)*KA + KA*(-A + PA) + 
	    (2*(hix - hjx)*(hiy - hjy)*KP*(P - PP))/(sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + 
	    2*KP*((-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)) + 
	    (hix - hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hiy + hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjLasty + hjy)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy)));

answer.x21 = 2*(0.5*hiy - 0.5*hjLasty)*(-0.5*hiNextx + 0.5*hjx)*KA - KA*(-A + PA) + 
	    (2*(hix - hjx)*(hiy - hjy)*KP*(P - PP))/(sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + 
	    2*KP*((-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)) + 
	    (hiy - hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hix + hjx)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjLastx + hjx)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy)));

answer.x22 = 2*((-0.5*hix + 0.5*hjLastx)*(-0.5*hiNextx + 0.5*hjx)*KA - 
	    (KP*(P - PP)*((hix - hjx)*(hix - hjx)))/(sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))*sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy))) + 
	    KP*((-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)) + 
	    (hiy - hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)))*
	    ((-hiy + hjy)/sqrt((hix - hjx)*(hix - hjx) + (hiy - hjy)*(hiy - hjy)) + 
	    (-hjLasty + hjy)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy))));
}
else {
//j is not n nor neighbor of n
answer.x11 = 2*(-0.5*hiLasty + 0.5*hiNexty)*(-0.5*hjLasty + 0.5*hjNexty)*KA + 
	    2*KP*((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	    ((-hjLastx + hjx)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy)) + 
	    (-hjNextx + hjx)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy)));

answer.x12 = 2*(-0.5*hiLasty + 0.5*hiNexty)*(0.5*hjLastx - 0.5*hjNextx)*KA + 
	    2*KP*((-hiLastx + hix)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNextx + hix)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	    ((-hjLasty + hjy)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy)) + 
	    (-hjNexty + hjy)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy)));                    

answer.x21 = 2*(0.5*hiLastx - 0.5*hiNextx)*(-0.5*hjLasty + 0.5*hjNexty)*KA + 
	    2*KP*((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	    ((-hjLastx + hjx)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy)) + 
	    (-hjNextx + hjx)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy)));

answer.x22 = 2*(0.5*hiLastx - 0.5*hiNextx)*(0.5*hjLastx - 0.5*hjNextx)*KA + 
	    2*KP*((-hiLasty + hiy)/sqrt((-hiLastx + hix)*(-hiLastx + hix) + (-hiLasty + hiy)*(-hiLasty + hiy)) + 
	    (-hiNexty + hiy)/sqrt((-hiNextx + hix)*(-hiNextx + hix) + (-hiNexty + hiy)*(-hiNexty + hiy)))*
	    ((-hjLasty + hjy)/sqrt((hjLastx - hjx)*(hjLastx - hjx) + (hjLasty - hjy)*(hjLasty - hjy)) + 
	    (-hjNexty + hjy)/sqrt((-hjNextx + hjx)*(-hjNextx + hjx) + (-hjNexty + hjy)*(-hjNexty + hjy)));
}
return answer;    
}


double VoronoiQuadraticEnergy::getSigmaXX()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
   
    answer += dot(deidHn(cell,i), dHdex(ri1,ri2,ri3));
}
};
double b1,b2,b3,b4;
Box->getBoxDims(b1,b2,b3,b4);
double area = b1*b4;

return answer/area;    
}

double VoronoiQuadraticEnergy::getSigmaYY()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
   
    answer += dot(deidHn(cell,i), dHdey(ri1,ri2,ri3));
}
};


double b1,b2,b3,b4;
Box->getBoxDims(b1,b2,b3,b4);
double area = b1*b4;

return answer/area;    
}


double VoronoiQuadraticEnergy::getd2Edgammadgamma(vector<double> &d2Eidgammadgamma)
{
double answer = 0.0;
computeGeometry();
d2Eidgammadgamma.reserve(Ncells);
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);
double d2EidgammadgammaData;

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
    d2EidgammadgammaData = 0.0;
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
    for(int j = 0; j < h_nn.data[cell]; j++)
    {

	double2 rj1,rj2,rj3; 
	// index of the 3rd cell center that is neighbour of vertex i.
	// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
	int jlast = j - 1;
	if(jlast == -1){
	    jlast = h_nn.data[cell] - 1;
	}

	rj1=h_p.data[cell];
	rj2=h_p.data[h_n.data[n_idx(j,cell)]];
	rj3=h_p.data[h_n.data[n_idx(jlast,cell)]];

	answer += dot(d2eidHndHj(cell, i, j) * dHdgamma(rj1,rj2,rj3), dHdgamma(ri1,ri2,ri3));
	d2EidgammadgammaData += dot(d2eidHndHj(cell, i, j) * dHdgamma(rj1,rj2,rj3), dHdgamma(ri1,ri2,ri3));
    }                
    answer += dot(deidHn(cell,i), d2Hdgamma2(ri1,ri2,ri3));
    d2EidgammadgammaData += dot(deidHn(cell,i), d2Hdgamma2(ri1,ri2,ri3));   
}
d2Eidgammadgamma.push_back(d2EidgammadgammaData);
};


return answer;    
}

double VoronoiQuadraticEnergy::getd2Edgammadgamma()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
    for(int j = 0; j < h_nn.data[cell]; j++)
    {

	double2 rj1,rj2,rj3; 
	// index of the 3rd cell center that is neighbour of vertex i.
	// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
	int jlast = j - 1;
	if(jlast == -1){
	    jlast = h_nn.data[cell] - 1;
	}

	rj1=h_p.data[cell];
	rj2=h_p.data[h_n.data[n_idx(j,cell)]];
	rj3=h_p.data[h_n.data[n_idx(jlast,cell)]];

	answer += dot(d2eidHndHj(cell, i, j) * dHdgamma(rj1,rj2,rj3), dHdgamma(ri1,ri2,ri3));
    }                
    answer += dot(deidHn(cell,i), d2Hdgamma2(ri1,ri2,ri3));
}
};


return answer;    
}

double VoronoiQuadraticEnergy::getd2EdepsilonXdepsilonX()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
    for(int j = 0; j < h_nn.data[cell]; j++)
    {

	double2 rj1,rj2,rj3; 
	// index of the 3rd cell center that is neighbour of vertex i.
	// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
	int jlast = j - 1;
	if(jlast == -1){
	    jlast = h_nn.data[cell] - 1;
	}

	rj1=h_p.data[cell];
	rj2=h_p.data[h_n.data[n_idx(j,cell)]];
	rj3=h_p.data[h_n.data[n_idx(jlast,cell)]];

	answer += dot(d2eidHndHj(cell, i, j) * dHdex(rj1,rj2,rj3), dHdex(ri1,ri2,ri3));
    }                
    answer += dot(deidHn(cell,i), d2Hdexdex(ri1,ri2,ri3));
}
};


return answer;    
}


double VoronoiQuadraticEnergy::getd2EdepsilonYdepsilonY()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
    for(int j = 0; j < h_nn.data[cell]; j++)
    {

	double2 rj1,rj2,rj3; 
	// index of the 3rd cell center that is neighbour of vertex i.
	// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
	int jlast = j - 1;
	if(jlast == -1){
	    jlast = h_nn.data[cell] - 1;
	}

	rj1=h_p.data[cell];
	rj2=h_p.data[h_n.data[n_idx(j,cell)]];
	rj3=h_p.data[h_n.data[n_idx(jlast,cell)]];

	answer += dot(d2eidHndHj(cell, i, j) * dHdey(rj1,rj2,rj3), dHdey(ri1,ri2,ri3));
    }                
    answer += dot(deidHn(cell,i), d2Hdeydey(ri1,ri2,ri3));
}
};


return answer;    
}



double VoronoiQuadraticEnergy::getd2Edepsilondepsilon()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
    for(int j = 0; j < h_nn.data[cell]; j++)
    {

	double2 rj1,rj2,rj3; 
	// index of the 3rd cell center that is neighbour of vertex i.
	// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
	int jlast = j - 1;
	if(jlast == -1){
	    jlast = h_nn.data[cell] - 1;
	}

	rj1=h_p.data[cell];
	rj2=h_p.data[h_n.data[n_idx(j,cell)]];
	rj3=h_p.data[h_n.data[n_idx(jlast,cell)]];

	answer += dot(d2eidHndHj(cell, i, j) * dHdep(rj1,rj2,rj3), dHdep(ri1,ri2,ri3));
    }                
    answer += dot(deidHn(cell,i), d2Hdepdep(ri1,ri2,ri3));
}
};


return answer;    
}

double VoronoiQuadraticEnergy::getdEdepsilon()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];
    // double2 circumcent;
    // Circumcenter(ri2-ri1,ri3-ri1,circumcent);
    // cout<<"cencumcenter:( "<<circumcent.x<<", "<<circumcent.y<<" and vorocur: ("<<h_v.data[n_idx(i,cell)].x<<", "<<h_v.data[n_idx(i,cell)].y<<endl;
    // break;
   
    answer += dot(deidHn(cell,i), dHdep(ri1,ri2,ri3));
}
};


return answer;    
}

void VoronoiQuadraticEnergy::getd2Edgammadr(vector<double2> &d2Edgammadr)
{
computeGeometry();
d2Edgammadr.reserve(Ncells);
for (int i = 0; i < Ncells; ++i) {
d2Edgammadr[i].x = 0.0;
d2Edgammadr[i].y = 0.0;
}
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);
Matrix2x2 tempMatrixdHdri(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixd2eidHndHj(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixd2Hdridgamma(0.0,0.0,0.0,0.0);
// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
for (int cell = 0; cell < Ncells; ++cell)
{

// First we calculate the self term for ri dEcelldgammadrcell
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];


    for(int j = 0; j < h_nn.data[cell]; j++)
    {

	double2 rj1,rj2,rj3; 
	// index of the 3rd cell center that is neighbour of vertex i.
	// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
	int jlast = j - 1;
	if(jlast == -1){
	    jlast = h_nn.data[cell] - 1;
	}

	rj1=h_p.data[cell];
	rj2=h_p.data[h_n.data[n_idx(j,cell)]];
	rj3=h_p.data[h_n.data[n_idx(jlast,cell)]];
	tempMatrixd2eidHndHj=d2eidHndHj(cell, i, j);
	tempMatrixdHdri=dHdri(rj1,rj2,rj3);
	tempMatrixd2eidHndHj.transpose();
	tempMatrixdHdri.transpose();
	d2Edgammadr[cell] = d2Edgammadr[cell] + tempMatrixdHdri * (tempMatrixd2eidHndHj * dHdgamma(ri1,ri2,ri3));
    }
    tempMatrixd2Hdridgamma=d2Hdridgamma(ri1,ri2,ri3);
    tempMatrixd2Hdridgamma.transpose();              
    d2Edgammadr[cell] = d2Edgammadr[cell] + tempMatrixd2Hdridgamma * deidHn(cell,i) ;
}
// Second we calculate the term for the neighbors of cell n dEidgammadrn
for (int i = 0; i < h_nn.data[cell]; i++)
{
    //all the cells that are neighbors of vertex i
    double2 ri1,ri2,ri3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int ilast = i - 1;
    if(ilast == -1){
	ilast = h_nn.data[cell] - 1;
    }

    ri1=h_p.data[cell];
    ri2=h_p.data[h_n.data[n_idx(i,cell)]];
    ri3=h_p.data[h_n.data[n_idx(ilast,cell)]];

    // only the verteces shared by both cell cell and cell n contribute to the derivatives
    for(int j = 0; j < h_nn.data[cell]; j++)
    {

	double2 rj1,rj2,rj3,rj4,rj5; 
	// index of the 3rd cell center that is neighbour of vertex i.
	// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
	// rj4 is the next neighbot after rj2, rj5 is the last neighbot before rj3
	int jlast = j - 1;
	if(jlast == -1){
	    jlast = h_nn.data[cell] - 1;
	}
	int jlastlast = jlast - 1;
	if(jlastlast == -1){
	    jlastlast = h_nn.data[cell] - 1;
	}
	int jnext = j + 1;
	if(jnext == h_nn.data[cell]){
	    jnext = 0;
	}

	rj1=h_p.data[cell];
	rj2=h_p.data[h_n.data[n_idx(j,cell)]];
	rj3=h_p.data[h_n.data[n_idx(jlast,cell)]];
	rj4=h_p.data[h_n.data[n_idx(jnext,cell)]];
	rj5=h_p.data[h_n.data[n_idx(jlastlast,cell)]];
	// only the verteces shared by both cell cell and cell n contribute to the derivatives
	tempMatrixd2eidHndHj=d2eidHndHj(cell, i, j);
	tempMatrixdHdri=dHdri(rj3,rj1,rj2);
	tempMatrixd2eidHndHj.transpose();
	tempMatrixdHdri.transpose();
	d2Edgammadr[h_n.data[n_idx(jlast,cell)]] = d2Edgammadr[h_n.data[n_idx(jlast,cell)]] + tempMatrixdHdri * (tempMatrixd2eidHndHj * dHdgamma(ri1,ri2,ri3));
	tempMatrixd2eidHndHj=d2eidHndHj(cell, i, j);
	tempMatrixdHdri=dHdri(rj2,rj1,rj3);
	tempMatrixd2eidHndHj.transpose();
	tempMatrixdHdri.transpose();               
	d2Edgammadr[h_n.data[n_idx(j,cell)]] = d2Edgammadr[h_n.data[n_idx(j,cell)]] + tempMatrixdHdri * (tempMatrixd2eidHndHj * dHdgamma(ri1,ri2,ri3)) ;
    }      
    tempMatrixd2Hdridgamma=d2Hdridgamma(ri3,ri1,ri2);
    tempMatrixd2Hdridgamma.transpose();             
    d2Edgammadr[h_n.data[n_idx(ilast,cell)]] = d2Edgammadr[h_n.data[n_idx(ilast,cell)]] + tempMatrixd2Hdridgamma * deidHn(cell,i) ;
    tempMatrixd2Hdridgamma=d2Hdridgamma(ri2,ri1,ri3);
    tempMatrixd2Hdridgamma.transpose();  
    d2Edgammadr[h_n.data[n_idx(i,cell)]] = d2Edgammadr[h_n.data[n_idx(i,cell)]] + tempMatrixd2Hdridgamma * deidHn(cell,i) ;
}

};


}

Matrix2x2 VoronoiQuadraticEnergy::d2Eidrjdrk(int i, int j, int k)
{
computeGeometry();
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);
Matrix2x2 answer(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixd2EidHldHm(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixProduct(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixdhldHldrj(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixdhldHmdrk(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixd2Hldrjdrkx(0.0,0.0,0.0,0.0);
Matrix2x2 tempMatrixd2Hldrjdrky(0.0,0.0,0.0,0.0);

for (int l = 0; l< h_nn.data[i]; l++)
{
//all the cells that are neighbors of vertex i
double2 rl1,rl2,rl3; 
// index of the 3rd cell center that is neighbour of vertex i.
// The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
int llast = l - 1;
if(llast == -1){
    llast = h_nn.data[i] - 1;
}

rl1=h_p.data[i];
rl2=h_p.data[h_n.data[n_idx(l,i)]];
rl3=h_p.data[h_n.data[n_idx(llast,i)]];

if(j!=i&&j!=h_n.data[n_idx(l,i)]&&j!=h_n.data[n_idx(llast,i)]) continue;
//cout<<j<<"is the neighbor of "<<"i";

double2 dEidhl;
dEidhl=deidHn(i,l);

for(int m = 0; m < h_nn.data[i]; m++)
{

    double2 rm1,rm2,rm3; 
    // index of the 3rd cell center that is neighbour of vertex i.
    // The 1st cell center is cell cell, and the second one is h_n.data[n_idx(i,cell)]
    int mlast = m - 1;
    if(mlast == -1){
	mlast = h_nn.data[i] - 1;
    }

    rm1=h_p.data[i];
    rm2=h_p.data[h_n.data[n_idx(m,i)]];
    rm3=h_p.data[h_n.data[n_idx(mlast,i)]];
    if(k!=i&&k!=h_n.data[n_idx(m,i)]&&k!=h_n.data[n_idx(mlast,i)])continue;
    //cout<<k<<"is the neighbor of "<<i<<" ";

    tempMatrixd2EidHldHm=d2eidHndHj(i,l,m);
    //cout<<"tempMatrixd2EidHldHm "<<tempMatrixd2EidHldHm.x11<<" "<<tempMatrixd2EidHldHm.x12<<" "<<tempMatrixd2EidHldHm.x21<<" "<<tempMatrixd2EidHldHm.x22<<endl;

    if(j==i){
	tempMatrixdhldHldrj=dHdri(rl1,rl2,rl3);
    }else if(j==h_n.data[n_idx(l,i)]){
	tempMatrixdhldHldrj=dHdri(rl2,rl1,rl3);
    }else if(j==h_n.data[n_idx(llast,i)]){
	tempMatrixdhldHldrj=dHdri(rl3,rl2,rl1);
    };
    //cout<<"tempMatrixdhldHldrj "<<tempMatrixdhldHldrj.x11<<" "<<tempMatrixdhldHldrj.x12<<" "<<tempMatrixdhldHldrj.x21<<" "<<tempMatrixdhldHldrj.x22<<endl;
    //tempMatrixdhldHldrj.transpose();

    if(k==i){
	tempMatrixdhldHmdrk=dHdri(rm1,rm2,rm3);
    }else if(k==h_n.data[n_idx(m,i)]){
	tempMatrixdhldHmdrk=dHdri(rm2,rm1,rm3);
    }else if(k==h_n.data[n_idx(mlast,i)]){
	tempMatrixdhldHmdrk=dHdri(rm3,rm2,rm1);
    };
    //cout<<"tempMatrixdhldHmdrk "<<tempMatrixdhldHmdrk.x11<<" "<<tempMatrixdhldHmdrk.x12<<" "<<tempMatrixdhldHmdrk.x21<<" "<<tempMatrixdhldHmdrk.x22<<endl;
    tempMatrixProduct = tempMatrixd2EidHldHm * tempMatrixdhldHmdrk;
    tempMatrixProduct.transpose();
    tempMatrixProduct = tempMatrixProduct * tempMatrixdhldHldrj;
    tempMatrixProduct.transpose();
    answer += tempMatrixProduct;
    //cout<<answer.x11<<" "<<answer.x12<<" "<<answer.x21<<" "<<answer.x22<<endl;
}

if(j==i&&k==i){
    d2Hdridrj(rl1,rl2,rl3,true,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==i&&k==h_n.data[n_idx(l,i)]){
    d2Hdridrj(rl1,rl2,rl3,false,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==i&&k==h_n.data[n_idx(llast,i)]){
    d2Hdridrj(rl1,rl3,rl2,false,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==h_n.data[n_idx(l,i)]&&k==i){
    d2Hdridrj(rl2,rl1,rl3,false,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==h_n.data[n_idx(l,i)]&&k==h_n.data[n_idx(l,i)]){
    d2Hdridrj(rl2,rl1,rl3,true,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==h_n.data[n_idx(l,i)]&&k==h_n.data[n_idx(llast,i)]){
    d2Hdridrj(rl2,rl3,rl1,false,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==h_n.data[n_idx(llast,i)]&&k==i){
    d2Hdridrj(rl3,rl1,rl2,false,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==h_n.data[n_idx(llast,i)]&&k==h_n.data[n_idx(l,i)]){
    d2Hdridrj(rl3,rl2,rl1,false,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}else if(j==h_n.data[n_idx(llast,i)]&&k==h_n.data[n_idx(llast,i)]){
    d2Hdridrj(rl3,rl2,rl1,true,tempMatrixd2Hldrjdrkx,tempMatrixd2Hldrjdrky);
    answer += dEidhl.x * tempMatrixd2Hldrjdrkx + dEidhl.y * tempMatrixd2Hldrjdrky;
}
//cout<<"tempMatrixd2Hldrjdrkx "<<tempMatrixd2Hldrjdrkx.x11<<" "<<tempMatrixd2Hldrjdrkx.x12<<" "<<tempMatrixd2Hldrjdrkx.x21<<" "<<tempMatrixd2Hldrjdrkx.x22<<endl;
}
return answer;
}

/*double VoronoiQuadraticEnergy::getd2EdgammadgammaOldPaper()
{
computeGeometry();
double answer = 0.0;
//read in the needed data
ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

// for (int cell = 0; cell < Ncells; ++cell)
//     {
//         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
//         for (int i = 0; i < h_cvn.data[cell]; i++)
//         {
//             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
//             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
//         }
//     }

//Loop over all the cells
double b1,b2,b3,b4;
Box->getBoxDims(b1,b2,b3,b4);
for (int cell = 0; cell < Ncells; ++cell)
{
for (int i = 0; i < h_nn.data[cell]; i++)
{
    int2 periodicitycelli;
    double2 rcell, ri;
    rcell = h_p.data[cell];
    ri = h_p.data[h_n.data[n_idx(i,cell)]];
    Box->periodicity(ri,rcell,periodicitycelli);
    for(int j = 0; j < h_nn.data[cell]; j++)
    {
	int2 periodicitycellj;
	double2 rj;
	rj = h_p.data[h_n.data[n_idx(j,cell)]];
	Box->periodicity(rj,rcell,periodicitycellj);
                if(periodicitycellj.y!=0&&periodicitycelli.y!=0){
                    Matrix2x2 d2Ecelldridrj;
                    d2Ecelldridrj = d2Eidrjdrk(cell, h_n.data[n_idx(i,cell)] ,h_n.data[n_idx(j,cell)]);
                    answer += b4*b4 * periodicitycellj.y *periodicitycelli.y * d2Ecelldridrj.x11;
                }
            }                
        }
        };


    return answer;    
    }

void VoronoiQuadraticEnergy::getd2EdgammadrOldPaper(vector<double2> &d2Edgammadr)
    {
    computeGeometry();
    double answer = 0.0;
    d2Edgammadr.reserve(Ncells);
    for (int i = 0; i < Ncells; ++i) {
        d2Edgammadr[i].x = 0.0;
        d2Edgammadr[i].y = 0.0;
    }
    //read in the needed data
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

    // for (int cell = 0; cell < Ncells; ++cell)
    //     {
    //         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
    //         for (int i = 0; i < h_cvn.data[cell]; i++)
    //         {
    //             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
    //             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
    //         }
    //     }

    //Loop over all the cells
    double b1,b2,b3,b4;
    Box->getBoxDims(b1,b2,b3,b4);
    for (int cellj = 0; cellj < Ncells; ++cellj){
        //first term
        for (int cell = 0; cell < Ncells; ++cell)
            {
            for (int i = 0; i < h_nn.data[cell]; i++)
            {
                int2 periodicitycelli;
                double2 rcell, ri;
                rcell = h_p.data[cell];
                ri = h_p.data[h_n.data[n_idx(i,cell)]];
                Box->periodicity(ri,rcell,periodicitycelli);
                if(periodicitycelli.y!=0){
                    Matrix2x2 d2Ecelldridrcellj;
                    d2Ecelldridrcellj=d2Eidrjdrk(cell,h_n.data[n_idx(i,cell)],cellj);
                    d2Edgammadr[cellj].x += b4 * periodicitycelli.y * d2Ecelldridrcellj.x11;
                    d2Edgammadr[cellj].y += b4 * periodicitycelli.y * d2Ecelldridrcellj.x12;
                }
            }
            }

    }
    


    }

void VoronoiQuadraticEnergy::getd2EdgammadrOldPaperWrong(vector<double2> &d2Edgammadr)
    {
    computeGeometry();
    double answer = 0.0;
    d2Edgammadr.reserve(Ncells);
    for (int i = 0; i < Ncells; ++i) {
        d2Edgammadr[i].x = 0.0;
        d2Edgammadr[i].y = 0.0;
    }
    //read in the needed data
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(neighborNum,access_location::host, access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::read);

    // for (int cell = 0; cell < Ncells; ++cell)
    //     {
    //         cout<<"Cell "<<cell<<" number of vertices "<<h_cvn.data[cell]<<endl;
    //         for (int i = 0; i < h_cvn.data[cell]; i++)
    //         {
    //             cout<<"n_idx(i,cell): "<<n_idx(i,cell)<<endl;
    //             cout<<"Cell "<<cell<<" Vetex"<<i<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+1]<<" "<<h_vcn.data[3*h_cv.data[n_idx(i,cell)]+2]<<endl;
    //         }
    //     }

    //Loop over all the cells
    double b1,b2,b3,b4;
    Box->getBoxDims(b1,b2,b3,b4);
    for (int cellj = 0; cellj < Ncells; ++cellj){
        //first term
        for (int cell = 0; cell < Ncells; ++cell)
            {
            for (int i = 0; i < h_nn.data[cell]; i++)
            {
                int2 periodicitycelli;
                double2 rcell, ri;
                rcell = h_p.data[cell];
                ri = h_p.data[h_n.data[n_idx(i,cell)]];
                Box->periodicity(ri,rcell,periodicitycelli);
                if(periodicitycelli.y!=0){
                    Matrix2x2 d2Ecelldridrcellj;
                    d2Ecelldridrcellj=d2Eidrjdrk(cell,h_n.data[n_idx(i,cell)],cellj);
                    d2Edgammadr[cellj].x += b4 * periodicitycelli.y * d2Ecelldridrcellj.x11;
                    d2Edgammadr[cellj].y += b4 * periodicitycelli.y * d2Ecelldridrcellj.x12;
                }
            }
            }

        // 2nd term
        for (int k = 0; k < h_nn.data[cellj]; k++)
        {
            int2 periodicitycellk;
            double2 rcellj, rk;
            rcellj = h_p.data[cellj];
            rk = h_p.data[h_n.data[n_idx(k,cellj)]];
            Box->periodicity(rk,rcellj,periodicitycellk);
            for(int l = 0; l < h_nn.data[cellj]; l++)
            {
                int2 periodicitycelll;
                double2 rl;
                rl = h_p.data[h_n.data[n_idx(l,cellj)]];
                Box->periodicity(rl,rcellj,periodicitycelll);
                if(periodicitycellk.y!=0){
                    Matrix2x2 d2Ecelljdrldrk;
                    d2Ecelljdrldrk=d2Eidrjdrk(cellj,h_n.data[n_idx(k,cellj)],h_n.data[n_idx(l,cellj)]);
                    d2Edgammadr[cellj].x -= b4 * periodicitycellk.y * d2Ecelljdrldrk.x11;
                    d2Edgammadr[cellj].y -= b4 * periodicitycellk.y * d2Ecelljdrldrk.x12;
                }
            }                

            };
    }
    

    }

*/

/*!
get the total force
*/
void VoronoiQuadraticEnergy::reportTotalForce()
    {
        double2 answer;
        answer.x = 0; answer.y = 0;
        ArrayHandle<double2> h_f(cellForces,access_location::host,access_mode::readwrite);
        for (int ii = 0; ii < Ncells; ++ii){
            answer = h_f.data[ii] + answer;
        }
        cout<<"The total force is ( "<<answer.x<<", "<<answer.y<<" )."<<endl;
    };
