#ifndef NoseHooverChainNPT_H
#define NoseHooverChainNPT_H

#include "simpleEquationOfMotion.h"
#include "voronoiQuadraticEnergy.h"
#include "Simple2DCell.h"

/*! \file NoseHooverChainNPT.h */
//! Implements NPT dynamics according to the Nose-Hoover equations of motion with a chain of thermostats
/*!
 *This allows one to do standard NPT simulations. A chain (whose length can be specified by the user)
 of thermostats is used to maintain the target temperature. We closely follow the Frenkel & Smit
 update scheme, which is itself based on:
 Martyna, Tuckerman, Tobias, and Klein
 Mol. Phys. 87, 1117 (1996)
*/
class NoseHooverChainNPT : public simpleEquationOfMotion
    {
    public:
        //!The base constructor asks for the number of particles and the length of the chain
        NoseHooverChainNPT(int N, int M, double P);

        //!The system that can compute forces, move degrees of freedom, etc.
        shared_ptr<Simple2DModel> State;
        //!set the internal State to the given model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model){State = _model;};
        //!Also need Simple2DCell for the unit cell rescaling
        shared_ptr<Simple2DCell> Cell;
        virtual void set2DCell(shared_ptr<Simple2DCell> _model){Cell = _model;};
        //!Also need VoronoiQuadraticEnergy for computeEnergy, getSigmaXX, and getSigmaYY
        //shared_ptr<VoronoiQuadraticEnergy> VQE;
        //virtual void setVoronoiQuadraticEnergy(shared_ptr<VoronoiQuadraticEnergy> _vqe){VQE = _vqe;};

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        //! virtual void integrateEquationsOfMotionGPU();

        //!Get temperature, T
        double getT(){return Temperature;};
        //!Set temperature, T, and also the bath masses!
        void setT(double T);

        //!Helper structure for GPU branch. A two-component GPU array that contains the total KE and the velocity scale factor
        GPUArray<double> kineticEnergyScaleFactor;
        //!the (position,velocity,acceleration,mass) of the bath degrees of freedom
        GPUArray<double4> BathVariables;

        //!Report the current status of the bath
        void reportBathData();
        void reportBarostatData();

        virtual ~NoseHooverChainNPT();

    protected:
        //!Barostat position
        double epsilon;
        //!Barostat momentum and mass
        double p_epsilon;
        double epsilon_old;
        double delta_eps;
        double W;
        //!Target pressure and instantaneous pressure
        double P_target;
        double P_inst;
        //!"Neutral" area - when pressure is at target pressure exactly??
        double V;
        double Lx;
        double Ly;
        int d;

        //Barostat helpers
        double barostatKineticEnergy();
        void setBarostatParameters(double W_in, double P_target_in, double V0_in);
        void computeInstantaneousPressure();
        void updateBarostatHalfStep(double dt);
        void rescaleBoxAndPositions(double delta_epsilon);
        void rescaleVelocitiesBarostat(double delta_epsilon);
        void rescaleThermoVelocities();

        //!The targeted temperature
        double Temperature;
        //!The lference area at epsilon = 0ength of the NH chain
        int Nchain;
        //!The number of particles in the State
        int Ndof;
        //!A helper vector for the GPU branch...can be asked to store 0.5*m[i]*v[i]^2 as an array
        GPUArray<double> keArray;
        //!A helper structure for performing parallel reduction of the keArray
        GPUArray<double> keIntermediateReduction;

        //!Propagate the chain
        void propagateChain();
        void propagateChainGPU();
        //!Propagate the position and velocity of the particles
        void propagatePositionsVelocities();

        //!Rescale velocities on the GPU
        void rescaleVelocitiesGPU();
        //! combine kernel calls for vector combination and parallel reduction to compute the KE in the helper structure
        void calculateKineticEnergyGPU();
        //!Propagate the position and velocity of the particles...on the gpu
        void propagatePositionsVelocitiesGPU();

    };
#endif
