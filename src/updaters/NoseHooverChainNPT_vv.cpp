#include "voronoiQuadraticEnergy.h"
#include "Simple2DCell.h"
#include "NoseHooverChainNPT_vv.h"
#include "voronoiQuadraticEnergy.h"
#include "utilities.cuh"
#include <ostream>

/*!
Initialize everything, by default setting the target temperature to unity.
Note that in the current set up the thermostate masses are automatically set by the target temperature, assuming \tau = 1
*/
NoseHooverChainNPT::NoseHooverChainNPT(int N, int M, double P, double T)
    {
    //Initialise barostat variables
    epsilon = 0.0;
    p_epsilon = 0.0;
    P_target = P;
    P_inst = 0.0;
    d = 2; //dimensionality
    Timestep = 0;
    deltaT=0.01;
    V = N;
    Nf = (d * N) - d;
    Lx = sqrt(V);
    Ly = sqrt(V);
    GPUcompute=false;
    if(!GPUcompute)
        {
        kineticEnergyScaleFactor.neverGPU = true;
        BathVariables.neverGPU = true;
        keArray.neverGPU = true;
        keIntermediateReduction.neverGPU = true;
        }
    Points = N;
    displacements.resize(Points);
    keArray.resize(Points);
    keIntermediateReduction.resize(Points);
    Nchain = M;
    BathVariables.resize(Nchain+1);
    ArrayHandle<double4> h_bv(BathVariables);
    //set the initial position and velocity of the thermostats to zero
    for (int ii = 0; ii < Nchain+1; ++ii)
        {
        h_bv.data[ii].x = 0.0;
        h_bv.data[ii].y = 0.0;
        h_bv.data[ii].z = 0.0;
        };
    kineticEnergyScaleFactor.resize(2);
    setT(1.0);
    W = (( d * Nf) + d) * T * 10;
    };

//Set pointer to refer to voronoiModel

void NoseHooverChainNPT::set2DModel(shared_ptr<Simple2DModel> model)
            {
            State = model;
            voronoi = std::dynamic_pointer_cast<VoronoiQuadraticEnergy>(model);
            V = voronoi->getArea();
            Lx = voronoi->getLx();
            Ly = voronoi->getLy();
        }

/*!
Set the target temperature to the specified value.
A careful reading of the "Non-Hamiltonian molecular dynamics: Generalizing Hamiltonian phase
space principles to non-Hamiltonian systems" paper (jcp 2001) suggests the correct setting of the first thermostat chain mass to guarantee conservation of energy in the context of a total momentum=0 setting
*/

void NoseHooverChainNPT::setT(double T)
    {
    Temperature = T;
    ArrayHandle<double4> h_bv(BathVariables);
    h_bv.data[0].w = 2.0 * (Points-1)*Temperature;
    for (int ii = 1; ii < Nchain+1; ++ii)
        {
        h_bv.data[ii].w = Temperature;
        };
    ArrayHandle<double> kes(kineticEnergyScaleFactor,access_location::host,access_mode::overwrite);
    kes.data[0] = h_bv.data[0].w;
    kes.data[1] = 1.0;
    };

/*!
Advance by one time step. Of note, for computational efficiency the topology is only updated on the
half-time steps (i.e., right before the instantaneous forces will to be computed). This means that
after each call to the simulation to "performTimestep()" there is no guarantee that the topology will
actually be up-to-date. Probably best to call enforceTopology just before saving or evaluating shape
outside of the normal timestep loops.
*/
void NoseHooverChainNPT::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (voronoi->getNumberOfDegreesOfFreedom() != Points)
        {
        Points = voronoi->getNumberOfDegreesOfFreedom();
        displacements.resize(Points);
        setT(Temperature); //the bath mass depends on the number of degrees of freedom
        };
    integrateEquationsOfMotionCPU();
    /* if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
        */
    };

/*!
print out the current state of the bath: (pos, vel, accel, mass) for each element of the chain
*/
void NoseHooverChainNPT::reportBathData()
    {
    ArrayHandle<double4> bath(BathVariables);
    printf("position\tvelocity\tacceleration\tmass\n");
    for (int i = 0; i < BathVariables.getNumElements(); ++i)
        printf("%f\t%f\t%f\t%f\n",bath.data[i].x,bath.data[i].y,bath.data[i].z,bath.data[i].w);
    };

/*!
The implementation here closely follows algorithms 30 - 32 in Frenkel & Smit, generalized to the
case where the chain length is not necessarily always 2
*/
//Barostat-related functions

void NoseHooverChainNPT::integrateEquationsOfMotionCPU()
    {
    //Ensure valid forces exist before first half
    if (Timestep == 1)
        voronoi->computeForces();

    double K = 0.0;

    {
    ArrayHandle<double2> h_v(voronoi->returnVelocities(),
                             access_location::host,
                             access_mode::read);

    ArrayHandle<double> h_m(voronoi->returnMasses(),
                            access_location::host,
                            access_mode::read);

    for (int ii = 0; ii < Points; ++ii)
        K += 0.5 * h_m.data[ii] * dot(h_v.data[ii], h_v.data[ii]);
    }

    {
    ArrayHandle<double> h_kes(kineticEnergyScaleFactor,
                              access_location::host,
                              access_mode::readwrite);

    h_kes.data[0] = K;
    }

    phaseA();
    phaseB();
    phaseC();
    phaseB();
    phaseA();

    }

void NoseHooverChainNPT::phaseA()
    {
    ArrayHandle<double> h_kes(kineticEnergyScaleFactor);

    double dt8 = 0.125 * deltaT;
    double dt4 = 0.25  * deltaT;
    double dt2 = 0.5   * deltaT;

    ArrayHandle<double4> Bath(BathVariables);

    //Step 1
    {
    int ii = Nchain - 1;

    if (ii > 0)
        Bath.data[ii].z =
            (Bath.data[ii-1].w * Bath.data[ii-1].y * Bath.data[ii-1].y
             - Temperature)
            / Bath.data[ii].w;
    else
        Bath.data[ii].z =
            (2.0 * (h_kes.data[0] + barostatKineticEnergy())
             - (2.0 * Points - 1.0) * Temperature)
            / Bath.data[ii].w;

    Bath.data[ii].y += Bath.data[ii].z * dt4;
    }

    //Steps 2-4 working down the chain
    for (int ii = Nchain - 2; ii >= 0; --ii)
        {
        Bath.data[ii].y *= exp(-Bath.data[ii+1].y * dt8);

        if (ii == 0)
            {
            double total_ke =
                h_kes.data[0] + barostatKineticEnergy();

            Bath.data[0].z =
                (2.0 * total_ke
                 - (2.0 * Points - 1.0) * Temperature)
                / Bath.data[0].w;
            }
        else
            {
            Bath.data[ii].z =
                (Bath.data[ii-1].w
                 * Bath.data[ii-1].y
                 * Bath.data[ii-1].y
                 - Temperature)
                / Bath.data[ii].w;
            }

        Bath.data[ii].y += Bath.data[ii].z * dt4;
        }

    {
    double ef1 = exp(-Bath.data[0].y * dt8);

    //Step 5
    barostatVelocityScale(ef1);

    //Step 6
    updateBarostatVelocity(dt4);

    //Step 7
    barostatVelocityScale(ef1);

    //Step 8
   // {
   // double veps     = getBarostatVelocity();
   // double vxi1     = Bath.data[0].y;
   // double combined = vxi1 + veps;

   // double scale = exp(-combined * dt2);

   // ArrayHandle<double2> h_v(voronoi->returnVelocities(),
     //                        access_location::host,
     //                        access_mode::readwrite);

   // for (int ii = 0; ii < Points; ++ii)
     //   {
     //   h_v.data[ii].x *= scale;
     //   h_v.data[ii].y *= scale;
     //   }

    //keep kinetic energy synchronised with scaled velocities
    //h_kes.data[0] *= scale * scale;

    //}

    // Step 9
    for (int ii = 0; ii < Nchain; ++ii)
        Bath.data[ii].x += dt2 * Bath.data[ii].y;

    //Step 10
    {
    double total_ke =
        h_kes.data[0] + barostatKineticEnergy();

    Bath.data[0].z =
        (2.0 * total_ke
         - (2.0 * Points - 1.0) * Temperature)
        / Bath.data[0].w;

    Bath.data[0].y += Bath.data[0].z * dt4;
    }

    //Steps 11-12
    for (int ii = 1; ii <= Nchain - 2; ++ii)
        {
        Bath.data[ii].y *= exp(-Bath.data[ii-1].y * dt8);

        Bath.data[ii].z =
            (Bath.data[ii-1].w
             * Bath.data[ii-1].y
             * Bath.data[ii-1].y
             - Temperature)
            / Bath.data[ii].w;

        Bath.data[ii].y += Bath.data[ii].z * dt4;
        }

    //Step 13
    {
    int ii = Nchain - 1;

    if (ii > 0)
        Bath.data[ii].y *= exp(-Bath.data[ii-1].y * dt8);

    if (ii > 0)
        Bath.data[ii].z =
            (Bath.data[ii-1].w
             * Bath.data[ii-1].y
             * Bath.data[ii-1].y
             - Temperature)
            / Bath.data[ii].w;
    else
        {
        double total_ke =
            h_kes.data[0] + barostatKineticEnergy();

        Bath.data[ii].z =
            (2.0 * total_ke
             - (2.0 * Points - 1.0) * Temperature)
            / Bath.data[ii].w;
        }

    Bath.data[ii].y += Bath.data[ii].z * dt4;
    }
    }
    }

double NoseHooverChainNPT::getBarostatVelocity()
    {
    return p_epsilon / W;
    }

void NoseHooverChainNPT::barostatVelocityScale(double scale)
    {
    //Rescale barostat momentum by scale factor: v_epsilon <- v_epsilon * scale  =>  p_epsilon <- p_epsilon * scale
    p_epsilon *= scale;
    }

void NoseHooverChainNPT::updateBarostatVelocity(double dt4)
    {
    computeInstantaneousPressure();

    ArrayHandle<double> h_kes(kineticEnergyScaleFactor,
                              access_location::host,
                              access_mode::read);

    // kinetic contribution:
    // (d/Nf) * sum_i m_i v_i^2
    // = (2d/Nf) * K
    double kinetic_term =
        (2.0 / double(Nf)) * h_kes.data[0];

    double G_epsilon =
        double(d) * V * (P_inst - P_target)
        + kinetic_term;

    p_epsilon += dt4 * G_epsilon;
    }

double NoseHooverChainNPT::barostatKineticEnergy()
    {
    return 0.5 * ((p_epsilon * p_epsilon) / W);
    }

void NoseHooverChainNPT::computeInstantaneousPressure()
    {
    //compute kinetic energy
    ArrayHandle<double2> h_v(voronoi->returnVelocities(),access_location::host,access_mode::read);
    ArrayHandle<double>   h_m(voronoi->returnMasses(),access_location::host,access_mode::read);

    double K = 0.0;
    for (int ii = 0; ii < Points; ++ii)
        {
        double v2 = dot(h_v.data[ii], h_v.data[ii]);
        double mi = h_m.data[ii];
        K += 0.5 * mi * v2;
        }

    double SigmaXX = voronoi->getSigmaXX();
    double SigmaYY = voronoi->getSigmaYY();
    double virial2D = -0.5 * (SigmaXX + SigmaYY);

    P_inst = virial2D + (K / V);
    }

void NoseHooverChainNPT::reportBarostatData(std::ostream& out_stream)
{
    computeInstantaneousPressure();

    //Get thermostat energy out of array
    double E_thermostat = 0.0;
    ArrayHandle<double4> h_bv(BathVariables, access_location::host, access_mode::read);
    
    for (int ii = 0; ii < Nchain; ++ii)
        {
        double xi  = h_bv.data[ii].x; //thermostat position
        double vxi = h_bv.data[ii].y; //thermostat velocity
        double Q   = h_bv.data[ii].w; //thermostat mass

        E_thermostat += 0.5 * Q * vxi * vxi; 
        //Potential
        if (ii == 0)
            E_thermostat += (double(Nf) + 1.0) * Temperature * xi;
        else
            E_thermostat += Temperature * xi;
        }

    double E_pot = voronoi->computeEnergy();
    double E_kin = 0.0;
    {
    ArrayHandle<double2> h_v(voronoi->returnVelocities(),
                         access_location::host, access_mode::read);
    ArrayHandle<double>  h_m(voronoi->returnMasses(),
                         access_location::host, access_mode::read);
    for (int ii = 0; ii < Points; ++ii)
        E_kin += 0.5 * h_m.data[ii] * dot(h_v.data[ii], h_v.data[ii]);
    }
    double E_baro_kin = barostatKineticEnergy();
    double E_baro_pot = P_target * V;
    double H_ext = E_pot + E_kin + E_baro_pot + E_baro_kin + E_thermostat;
    double enthalpy = E_pot + E_kin + (P_target * V);

    out_stream << epsilon << "\t"
               << p_epsilon << "\t"
               << W << "\t"
               << P_target << "\t"
               << P_inst << "\t"
               << V << "\t"
               << E_pot << "\t"
               << E_kin << "\t"
               << E_baro_kin << "\t"
               << enthalpy << "\t"
               << H_ext
               << std::endl;
}

/*!
The simple part of the algorithm actually updates the positions and velocities of the partices.
This is the step in which a force calculation is required.
*/

 void NoseHooverChainNPT::phaseB()
    {
    double K_local = 0.0;
    ArrayHandle<double2> h_f(voronoi->returnForces(), access_location::host, access_mode::read);
    ArrayHandle<double2> h_v(voronoi->returnVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<double> h_m(voronoi->returnMasses(), access_location::host, access_mode::read);

    double veps = getBarostatVelocity();

    //New scaling factors from Tuckerman 2006 + GROMACS version
    double y = (1.0 + double(d)/double(Nf)) * veps * deltaT / 4.0;
    double s1 = exp(-y);
    double s2;

    //Tiny y causes problems :(
    if (fabs(y) < 1.0e-8)
        s2 = 1.0 + y*y/6.0;
    else
        s2 = sinh(y)/y;

    const double preScale  = s1 / s2;
    const double postScale = s1 * s2;

    for (int ii = 0; ii < Points; ++ii)
        {
        h_v.data[ii].x *= preScale;
        h_v.data[ii].y *= preScale;

        double scalar = deltaT2 / h_m.data[ii];

        h_v.data[ii].x += scalar * h_f.data[ii].x;
        h_v.data[ii].y += scalar * h_f.data[ii].y;

        h_v.data[ii].x *= postScale;
        h_v.data[ii].y *= postScale;

        K_local += 0.5 * h_m.data[ii] * dot(h_v.data[ii], h_v.data[ii]);
        }

    ArrayHandle<double> h_kes(kineticEnergyScaleFactor, access_location::host, access_mode::readwrite);
    h_kes.data[0] = K_local;
    } 

void NoseHooverChainNPT::updateBarostatPosition(double dt)
    {
    //Update barostat position by dt and rescale box accordingly.
    //epsilon <- epsilon + v_epsilon * dt, V(epsilon) = V_prev * exp(d * delta_epsilon)
    double delta_epsilon = getBarostatVelocity() * dt;
    epsilon += delta_epsilon;

    double factor = exp(delta_epsilon);
    Lx *= factor;
    Ly *= factor;
    voronoi->scaleRectangularUnitCell(Lx, Ly, factor);
    V = voronoi->getArea();
    }

void NoseHooverChainNPT::phaseC()
{
    double veps = getBarostatVelocity();

    //Updated scaling factors
    double x = 0.5 * veps * deltaT;
    double s1 = exp(x);
    double s2;
    if (fabs(x) < 1.0e-8)
        s2 = 1.0 + (x * x)/6.0;
    else
        s2 = sinh(x)/x;
    
    double scale_half = s1 * s2;


    //Update box dimensions, mesh, and particle positions first
    updateBarostatPosition(deltaT);

    //Now retrieve displacements and apply scaling factor -- preserves symmetric structure!
    {
        ArrayHandle<double2> h_v(voronoi->returnVelocities(),
                                 access_location::host,
                                 access_mode::read);

        ArrayHandle<double2> h_disp(displacements,
                                    access_location::host,
                                    access_mode::overwrite);

        for (int ii = 0; ii < Points; ++ii)
        { 

            //drift contribution with half scaling factor
            h_disp.data[ii].x = h_v.data[ii].x * deltaT * scale_half;
            h_disp.data[ii].y = h_v.data[ii].y * deltaT * scale_half;

        }
    }

    voronoi->moveDegreesOfFreedom(displacements);

    voronoi->enforceTopology();

    voronoi->computeForces();
}

/*!
The GPU implementation of the identical algorithm done on the CPU
*/
/* void NoseHooverChainNPT::integrateEquationsOfMotionGPU()
    {
    //The kernel calling scheme. To avoid ridiculous numbers of brackets for array handle scoping,
    //we'll define helper functions

    //for now, let's update the chain variables on the CPU... profile later
    propagateChainGPU(); // use data structure that holds [KE,s], update both.
    rescaleVelocitiesGPU(); //use the velocity vector and the [KE,s] data structure. Note that KE is already scaled by s^2 in the above step
    propagatePositionsVelocitiesGPU();
    calculateKineticEnergyGPU(); //get the kinetic energy into the [KE,s] data structure
    propagateChainGPU();
    rescaleVelocitiesGPU();
    };

/*!
Do a multi-step dance to get the positions and velocities updated on the gpu branch
*/
/* void NoseHooverChainNPT::propagatePositionsVelocitiesGPU()
    {
    double deltaT2 = 0.5*deltaT;
    //first, we move particles according to their velocities
    State->moveDegreesOfFreedom(State->returnVelocities(),deltaT2);
    State->enforceTopology();
    State->computeForces();

    //Now we execute the second half of the time step.. first we need to update the velocities according to the forces and the masses
    {//array handle scope for the second half of the time step
    ArrayHandle<double2> d_f(State->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<double2> d_v(State->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<double> d_m(State->returnMasses(),access_location::device,access_mode::read);
    gpu_NoseHooverChainNPT_update_velocities(d_v.data,d_f.data,d_m.data,deltaT,Points);
    };
    State->moveDegreesOfFreedom(State->returnVelocities(),deltaT2);
    };

/*!
This combines multiple kernel calls. First we make a vector of kinetic energies per particle, then
we perform a parallel block reduction, and then a serial reduction
*/
/* void NoseHooverChainNPT::calculateKineticEnergyGPU()
    {
    {//array handle scope for keArray preparation
    ArrayHandle<double2> d_v(State->returnVelocities(),access_location::device,access_mode::read);
    ArrayHandle<double> d_m(State->returnMasses(),access_location::device,access_mode::read);
    gpu_prepare_KE_vector(d_v.data,d_m.data,d_keArray.data,Points);
    }

    {//array handle scope for parallel reduction
    ArrayHandle<double> d_keArray(keArray,access_location::device,access_mode::read);
    ArrayHandle<double> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::readwrite);
    ArrayHandle<double> d_keIntermediate(keIntermediateReduction,access_location::device,access_mode::overwrite);

    gpu_parallel_reduction(d_keArray.data,d_keIntermediate.data,d_kes.data,0,Points);
    }
    };

/*!
Simply call the velocity rescaling function...
*/
/* void NoseHooverChainNPT::rescaleVelocitiesGPU()
    {
    ArrayHandle<double2> d_v(State->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<double> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::read);
    gpu_NoseHooverChainNPT_scale_velocities(d_v.data,d_kes.data,Points);
    };
*/ 
NoseHooverChainNPT::~NoseHooverChainNPT() = default;
