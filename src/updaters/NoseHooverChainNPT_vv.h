#ifndef NoseHooverChainNPT_H
#define NoseHooverChainNPT_H

#include "simpleEquationOfMotion.h"
#include "voronoiQuadraticEnergy.h"
#include "Simple2DCell.h"
#include <ostream>

class NoseHooverChainNPT : public simpleEquationOfMotion
    {
    public:
        //!The base constructor asks for the number of particles and the length of the chain
        NoseHooverChainNPT(int N, int M, double P, double T);

        //!The system that can compute forces, move degrees of freedom, etc.
        shared_ptr<Simple2DModel> State;
        shared_ptr<VoronoiQuadraticEnergy> voronoi;
        //!set the internal State to the given model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model);

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();

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
        void reportBarostatData(std::ostream& out_stream);

        virtual ~NoseHooverChainNPT();

        //!Barostat variables moved to public so can be accessed by voronoi.cpp
        //!Barostat position
        double epsilon;
        //!Barostat momentum and mass
        double p_epsilon;
        double epsilon_old;
        double delta_epsilon;
        double W;
        //!Target pressure and instantaneous pressure
        double P_target;
        double P_inst;
        //!"Neutral" area - when pressure is at target pressure exactly??
        double V;
        double Lx;
        double Ly;
        double Nf;

    protected:
        int d;
        int Timestep;
        double deltaT;
        bool GPUcompute;
        GPUArray<double2> displacements;

        //Barostat and Integration helpers
        void phaseA();
        void phaseB();
        void phaseC();

        double getBarostatVelocity();
        void barostatVelocityScale(double scale);
        void updateBarostatVelocity(double dt4);
        void updateBarostatPosition(double dt);
        
        double barostatKineticEnergy();
        void computeInstantaneousPressure();

        double total_ke;
        double target_dof;

        //!The targeted temperature
        double Temperature;
        //!The length of the NH chain
        int Nchain;
        //!The number of particles in the State
        int Points;
        
        //!A helper vector for the GPU branch...can be asked to store 0.5*m[i]*v[i]^2 as an array
        GPUArray<double> keArray;
        //!A helper structure for performing parallel reduction of the keArray
        GPUArray<double> keIntermediateReduction;

    };
#endif
