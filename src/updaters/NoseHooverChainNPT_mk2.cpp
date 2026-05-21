#include "voronoiQuadraticEnergy.h"
#include "Simple2DCell.h"
#include "NoseHooverChainNPT_vv.h"
#include "voronoiQuadraticEnergy.h"
#include "utilities.cuh"
#include <ostream>

/* extern double getSigmaXX();
extern double getSigmaYY();

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
    Ndof = N;
    displacements.resize(Ndof);
    keArray.resize(Ndof);
    keIntermediateReduction.resize(Ndof);
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
    W = (( d * N) + d) * T * 10;
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
    h_bv.data[0].w = 2.0 * (Ndof-1)*Temperature;
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
    if (voronoi->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = voronoi->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
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
    {
    double K = 0.0;
    {
    ArrayHandle<double2> h_v(voronoi->returnVelocities(), access_location::host, access_mode::read);
    ArrayHandle<double>  h_m(voronoi->returnMasses(),     access_location::host, access_mode::read);
    for (int ii = 0; ii < Ndof; ++ii)
        K += 0.5 * h_m.data[ii] * dot(h_v.data[ii], h_v.data[ii]);
    }
    ArrayHandle<double> h_kes(kineticEnergyScaleFactor, access_location::host, access_mode::readwrite);
    h_kes.data[0] = K;
    }
    phaseA();
    phaseB();
    phaseC();
    phaseB();
    phaseA();
    reportBarostatData();
    }

void NoseHooverChainNPT::phaseA()
    {
    //get rid of this later --- it doesn't need it past the first timestep in the simulation
    ArrayHandle<double> h_kes(kineticEnergyScaleFactor);
    double dt8 = 0.125 * deltaT;
    double dt4 = 0.25  * deltaT;
    double dt2 = 0.5   * deltaT;

    ArrayHandle<double4> Bath(BathVariables);
    //First half: steps 1-9
    //Step 1: Update v_{xi_M} by dt/4 (no rescale before it)
    {
    int ii = Nchain - 1;
    if (ii > 0)
        Bath.data[ii].z = (Bath.data[ii-1].w * Bath.data[ii-1].y * Bath.data[ii-1].y - Temperature)
                          / Bath.data[ii].w;
    else
        //single thermostat: G = (2*KE - (2N-1)*T) / Q
        Bath.data[ii].z = (2.0*(h_kes.data[0] + barostatKineticEnergy()) - (2.0*Ndof - 1.0)*Temperature)
                          / Bath.data[ii].w;
    Bath.data[ii].y += Bath.data[ii].z * dt4;
    }

    //Steps 2-4: Sweep down from ii = Nchain-2 to ii = 0
    //Each iteration: rescale v_{xi_ii} by v_{xi_{ii+1}}, then update v_{xi_ii}
    for (int ii = Nchain - 2; ii >= 0; --ii)
        {
        //Now we rescale
        Bath.data[ii].y *= exp(-Bath.data[ii+1].y * dt8);

        //Compute G for this level
        if (ii == 0)
            {
            //G_0 couples to both particle KE and barostat KE
            double total_ke = h_kes.data[0] + barostatKineticEnergy();
            Bath.data[0].z = (2.0 * total_ke - (2.0*Ndof - 1.0)*Temperature) / Bath.data[0].w;
            }
        else
            {
            Bath.data[ii].z = (Bath.data[ii-1].w * Bath.data[ii-1].y * Bath.data[ii-1].y - Temperature)
                              / Bath.data[ii].w;
            }

        //Update!
        Bath.data[ii].y += Bath.data[ii].z * dt4;
        }

    {
    double ef1 = exp(-Bath.data[0].y * dt8);

    //Step 5: rescale v_epsilon by ef1 (particle rescaling stored, applied later)
    barostatVelocityScale(ef1);   //scales v_epsilon in-place

    //Step 6: update v_epsilon by G_epsilon * dt/4
    updateBarostatVelocity(dt4);  //v_epsilon += G_epsilon * dt4

    //Step 7: rescale v_epsilon by ef1 again
    barostatVelocityScale(ef1);

    //Step 8: rescale particles by exp(-(v_{xi_1} + (1+d/Nf)*v_epsilon) * dt/2)
    {
    double veps     = getBarostatVelocity();
    double vxi1     = Bath.data[0].y;
    double combined = vxi1 + (1.0 + double(Ndim) / double(Ndof)) * veps;
    double scale    = exp(-combined * dt2);

    ArrayHandle<double2> h_v(voronoi->returnVelocities(), access_location::host, access_mode::readwrite);
    for (int ii = 0; ii < Ndof; ++ii)
        h_v.data[ii] *= scale;
    barostatVelocityScale(scale);
    }

    //Step 9: Update thermostat positions xi_j += v_{xi_j} * dt/2
    for (int ii = 0; ii < Nchain; ++ii)
        Bath.data[ii].x += dt2 * Bath.data[ii].y;

    //Now on to the second half :)

    //Step 10: Update v_{xi_1} by dt/4 (no rescale before it)
    {
    double total_ke = h_kes.data[0] + barostatKineticEnergy();
    Bath.data[0].z  = (2.0 * total_ke - (2.0*Ndof - 1.0)*Temperature) / Bath.data[0].w;
    Bath.data[0].y += Bath.data[0].z * dt4;
    }

    //Steps 11-12: Sweep up from ii = 1 to Nchain-2. Each iteration will rescale v_{xi_ii} by v_{xi_{ii-1}}, then update v_{xi_ii}
    for (int ii = 1; ii <= Nchain - 2; ++ii)
        {
        //Rescale
        Bath.data[ii].y *= exp(-Bath.data[ii-1].y * dt8);

        //Compute G
        Bath.data[ii].z = (Bath.data[ii-1].w * Bath.data[ii-1].y * Bath.data[ii-1].y - Temperature)
                          / Bath.data[ii].w;

        //Update
        Bath.data[ii].y += Bath.data[ii].z * dt4;
        }

    //Step 13: Rescale v_{xi_M} by v_{xi_{M-1}}, then update v_{xi_M}
    {
    int ii = Nchain - 1;
    if (ii > 0)  //what if there's only one?
        Bath.data[ii].y *= exp(-Bath.data[ii-1].y * dt8);

    //Compute G for top of chain
    if (ii > 0)
        Bath.data[ii].z = (Bath.data[ii-1].w * Bath.data[ii-1].y * Bath.data[ii-1].y - Temperature)
                          / Bath.data[ii].w;
    else
        {
        double total_ke = h_kes.data[0] + barostatKineticEnergy();
        Bath.data[ii].z = (2.0 * total_ke - (2.0*Ndof - 1.0)*Temperature) / Bath.data[ii].w;
        }

    Bath.data[ii].y += Bath.data[ii].z * dt4;
    }
    };
}

void NoseHooverChainNPT::getBarostatVelocity()
    {
    // v_epsilon = p_epsilon / W
    return p_epsilon / W;
    }

void NoseHooverChainNPT::barostatVelocityScale(double scale)
    {
    //Rescale barostat momentum by scale factor: v_epsilon <- v_epsilon * scale  =>  p_epsilon <- p_epsilon * scale
    p_epsilon *= scale;
    }

void NoseHooverChainNPT::updateBarostatVelocity(double dt4)
    {
    // p_epsilon <- p_epsilon + G_epsilon * dt/4
    // G_epsilon = V * (P_inst - P_target)
    computeInstantaneousPressure();
    p_epsilon += dt4 * V * (P_inst - P_target);
    }

double NoseHooverChainNPT::barostatKineticEnergy()
    {
    return 0.5 * ((p_epsilon * p_epsilon) / W);
    }

void NoseHooverChainNPT::computeInstantaneousPressure()
    {
    // compute kinetic energy
    ArrayHandle<double2> h_v(voronoi->returnVelocities(),access_location::host,access_mode::read);
    ArrayHandle<double>   h_m(voronoi->returnMasses(),access_location::host,access_mode::read);

    double K = 0.0;
    for (int ii = 0; ii < Ndof; ++ii)
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
    out_stream << epsilon << "\t"
               << p_epsilon << "\t"
               << W << "\t"
               << P_target << "\t"
               << P_inst << "\t"
               << V << "\t"
               << voronoi->computeEnergy() << "\t"
               << voronoi->computeKineticEnergy() << "\t"
               << barostatKineticEnergy()
               << std::endl;
}

/*!
The simple part of the algorithm actually updates the positions and velocities of the partices.
This is the step in which a force calculation is required.
*/

void NoseHooverChainNPT::phaseB()
    {
    // Phase B: update particle velocities by dt/2 using current forces (action of L_1).
    // Forces must already be computed before this is called.
    // Accumulates particle KE into h_kes.data[0] for use by the thermostat/barostat.
    double deltaT2 = 0.5 * deltaT;
    double K_local = 0.0;

    ArrayHandle<double2> h_f(voronoi->returnForces(),  access_location::host, access_mode::read);
    ArrayHandle<double2> h_v(voronoi->returnVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<double>  h_m(voronoi->returnMasses(),  access_location::host, access_mode::read);

    for (int ii = 0; ii < Ndof; ++ii)
        {
        h_v.data[ii]  += (deltaT2 / h_m.data[ii]) * h_f.data[ii];
        K_local       += 0.5 * h_m.data[ii] * dot(h_v.data[ii], h_v.data[ii]);
        }

    ArrayHandle<double> h_kes(kineticEnergyScaleFactor, access_location::host, access_mode::readwrite);
    h_kes.data[0] = K_local;
    }

void NoseHooverChainNPT::updateBarostatPosition(double dt)
    {
    // Update barostat position by dt and rescale box accordingly.
    // epsilon <- epsilon + v_epsilon * dt
    // V(epsilon) = V_prev * exp(d * delta_epsilon)
    double delta_epsilon = getBarostatVelocity() * dt;
    epsilon += delta_epsilon;

    double factor = exp(delta_epsilon);
    Lx *= factor;
    Ly *= factor;
    voronoi->setRectangularUnitCell(Lx, Ly);
    V = voronoi->getArea();
    }

void NoseHooverChainNPT::phaseC()
    {
    // Phase C: update box, rescale positions, and advance particle positions by dt
    // Every operation in coordinate space occurs here

    // Step 1: Update barostat position by full timestep. This determines the new cell volume: V(epsilon) = V_prev * exp(d*epsilon)
    double veps = getBarostatVelocity();
    updateBarostatPosition(deltaT);   //epsilon <- epsilon + v_epsilon * dt

    // Step 2 & 3: Rescale and advance particle positions. r_i <- r_i + (v_i + v_epsilon * r_i) * dt
    // The v_epsilon * r_i term accounts for the affine cell deformation, i.e. the cell-frame velocity of the particle due to box expansion/contraction.
    {
    ArrayHandle<double2> h_disp(displacements, access_location::host, access_mode::overwrite);
    ArrayHandle<double2> h_r(voronoi->returnPositions(), access_location::host, access_mode::read);
    ArrayHandle<double2> h_v(voronoi->returnVelocities(), access_location::host, access_mode::read);

    for (int ii = 0; ii < Ndof; ++ii)
        h_disp.data[ii] = (h_v.data[ii] + veps * h_r.data[ii]) * deltaT;
    }

    voronoi->moveDegreesOfFreedom(displacements);
    voronoi->enforceTopology();
    voronoi->computeForces();   // recompute forces at new positions for use in the second call of phase B
    }

void NoseHooverChainNPT::propagateChain()
    {
    ArrayHandle<double> h_kes(kineticEnergyScaleFactor);
    double dt8 = 0.125*deltaT;
    double dt4 = 0.25*deltaT;
    double dt2 = 0.5*deltaT;

    //partially update bath velocities and accelerations (quarter-timestep), from Nchain to 0
    ArrayHandle<double4> Bath(BathVariables);
    for (int ii = Nchain-1; ii > 0; --ii)
        {
        //update the acceleration: G = (Q_{i-1}*v_{i-1}^2 - T)/Q_i
        double total_ke = h_kes.data[0] + barostatKineticEnergy();
        double target_dof = 2.0 * Ndof - 1.0; //(2N-2) particle DOFs + 1 barostat DOF
        Bath.data[0].z = (2.0 * total_ke - target_dof * Temperature) / Bath.data[0].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        double ef = exp(-dt8*Bath.data[ii+1].y);
        Bath.data[ii].y *= ef;
        Bath.data[ii].y += Bath.data[ii].z*dt4;
        Bath.data[ii].y *= ef;
        };

    //Use particle KE from h_kes.data[0] and add barostat KE on-the-fly when computing G0.
    //IMPORTANT: do NOT overwrite h_kes.data[0] here; we only read it.
    Bath.data[0].z = (2.0*(h_kes.data[0] + barostatKineticEnergy())/Bath.data[0].w - 1.0);
    double ef = exp(-dt8*Bath.data[1].y);
    Bath.data[0].y *= ef;
    Bath.data[0].y += Bath.data[0].z*dt4;
    Bath.data[0].y *= ef;

    //update bath positions (half timestep)
    for (int ii = 0; ii < Nchain; ++ii)
        Bath.data[ii].x += dt2*Bath.data[ii].y;

    //get the factor that will scale particle velocities (store it but do not apply it here)
    h_kes.data[1] = exp(-dt2*Bath.data[0].y);
    //finally, do the other quarter-timestep of the velocities and accelerations, from 0 to Nchain
    Bath.data[0].z = (2.0*(h_kes.data[0] + barostatKineticEnergy())/Bath.data[0].w - 1.0);
    ef = exp(-dt8*Bath.data[1].y);
    Bath.data[0].y *= ef;
    Bath.data[0].y += Bath.data[0].z*dt4;
    Bath.data[0].y *= ef;
    for (int ii = 1; ii < Nchain; ++ii)
        {
        Bath.data[ii].z = (Bath.data[ii-1].w*Bath.data[ii-1].y*Bath.data[ii-1].y-Temperature)/Bath.data[ii].w;
        double ef2 = exp(-dt8*Bath.data[ii+1].y);
        Bath.data[ii].y *= ef2;
        Bath.data[ii].y += Bath.data[ii].z*dt4;
        Bath.data[ii].y *= ef2;
        };
    };

/* void NoseHooverChainNPT::propagateChainGPU()
    {
    ArrayHandle<double> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::readwrite);
    ArrayHandle<double4> d_Bath(BathVariables,access_location::device,access_mode::readwrite);
    gpu_NoseHooverChainNPT_propagateChain(d_kes.data,d_Bath.data,Temperature,deltaT,Nchain,Ndof);
    };

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
    gpu_NoseHooverChainNPT_update_velocities(d_v.data,d_f.data,d_m.data,deltaT,Ndof);
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
    gpu_prepare_KE_vector(d_v.data,d_m.data,d_keArray.data,Ndof);
    }

    {//array handle scope for parallel reduction
    ArrayHandle<double> d_keArray(keArray,access_location::device,access_mode::read);
    ArrayHandle<double> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::readwrite);
    ArrayHandle<double> d_keIntermediate(keIntermediateReduction,access_location::device,access_mode::overwrite);

    gpu_parallel_reduction(d_keArray.data,d_keIntermediate.data,d_kes.data,0,Ndof);
    }
    };

/*!
Simply call the velocity rescaling function...
*/
/* void NoseHooverChainNPT::rescaleVelocitiesGPU()
    {
    ArrayHandle<double2> d_v(State->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<double> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::read);
    gpu_NoseHooverChainNPT_scale_velocities(d_v.data,d_kes.data,Ndof);
    };
*/ 
NoseHooverChainNPT::~NoseHooverChainNPT() = default;
