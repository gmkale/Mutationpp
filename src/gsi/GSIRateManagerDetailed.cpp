/**
 * @file GSIRateManagerPhenomenological.cpp
 *
 * @brief Class which computes the chemical production rate for each species
 *        based on detailed chemistry models for catalysis and ablation.
 */

/*
 * Copyright 2018 von Karman Institute for Fluid Dynamics (VKI)
 *
 * This file is part of MUlticomponent Thermodynamic And Transport
 * properties for IONized gases in C++ (Mutation++) software package.
 *
 * Mutation++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Mutation++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Mutation++.  If not, see
 * <http://www.gnu.org/licenses/>.
 */


#include "NewtonSolver.h"
#include "Thermodynamics.h"
#include "Transport.h"

#include "GSIReaction.h"
#include "GSIRateLaw.h"
#include "GSIRateManager.h"
#include "GSIStoichiometryManager.h"
#include "SurfaceProperties.h"
#include "SurfaceState.h"

using namespace Eigen;
using namespace std;

using namespace Mutation::Numerics;
using namespace Mutation::Utilities::Config;
using namespace Mutation::Thermodynamics;

namespace Mutation {
    namespace GasSurfaceInteraction {

//=============================================================================

class GSIRateManagerDetailed :
    public GSIRateManager,
    public NewtonSolver<VectorXd, GSIRateManagerDetailed>
{
public:
    GSIRateManagerDetailed(DataGSIRateManager args)
        : GSIRateManager(args),
          m_surf_props(args.s_surf_state.getSurfaceProperties()),
		  m_ns(args.s_thermo.nSpecies()),
		  m_nr(args.s_reactions.size()),
          mv_kf(m_nr),
          mv_rate(m_nr),
          mn_site_sp(m_surf_props.nSiteSpecies()),
          mn_site_cat(m_surf_props.nSiteCategories()),
          mv_sp_in_site(mn_site_cat), // Define
          mv_sigma(mn_site_cat), // Define, site density, specified in the input file
          mv_rhoi(m_ns),
          mv_nd(m_ns+mn_site_sp),
          is_surf_steady_state(true),
          m_tol(1.e-15),
          m_pert(1.e-1),
          mv_X(mn_site_sp),
          mv_dX(mn_site_sp),
          mv_f_unpert(m_ns+mn_site_sp),
          mv_f(m_ns+mn_site_sp),
          m_jac(mn_site_sp,mn_site_sp),
          mv_mass(m_thermo.speciesMw()/NA)
    {
        for (int i_reac = 0; i_reac < m_nr; ++i_reac) {
            m_reactants.addReaction(
                i_reac, args.s_reactions[i_reac]->getReactants());
            m_irr_products.addReaction(
                i_reac, args.s_reactions[i_reac]->getProducts());
        }

        for (int i = 0; i < mn_site_cat; i++) { // All these better be done in SurfaceProperties
            // below gets the specified surface site density in the gsi input file
            mv_sigma(i) = m_surf_props.nSiteDensityInCategory(i);        // @TODO Remove this function
            // below get the number of species in a given site category, including empty sites
            mv_sp_in_site[i] = m_surf_props.nSpeciesInSiteCategory(i);   // @TODO Remove this function
        }

        //std::cout << "Surf Cov Init (mv_X init) " << mv_X << std::endl;

        //std::cout << mv_sp_in_site[0] << std::endl;
        //std::cout << "--------------------------" << std::endl;

        // Setup NewtonSolver
        setMaxIterations(1);
        setWriteConvergenceHistory(false);
        setEpsilon(m_tol);
    }

//=============================================================================

    ~GSIRateManagerDetailed(){}

//=============================================================================

    Eigen::VectorXd computeRates()
    {
        // IT DOESNT APPEAR AS IF THIS IS ACTUALLY USED UNTIL POST SYSTEM SOLUTION?

        // Note mv_nd is density he
        mv_rhoi = m_surf_state.getSurfaceRhoi();

        // Get reaction rate constant
        for (int i_r = 0; i_r < m_nr; ++i_r)
            mv_kf(i_r) =
                v_reactions[i_r]->getRateLaw()->forwardReactionRateCoefficient(
                    mv_rhoi, m_surf_state.getSurfaceT());

        // Getting all number densities
        m_thermo.convert<RHO_TO_CONC>(mv_rhoi.data(), mv_nd.data());
        // the above converts the gas phase species to mol/m3 and the surface species to #/m3?

        //std::cout << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;

        // this then sets gas phase species to #/m3 and surface species to fraction of total coverage
        mv_nd.head(m_ns) *= NA;
        mv_nd.tail(mn_site_sp) = m_surf_props.getSurfaceSiteCoverageFrac();

        //std::cout << "--------------------------" << std::endl;
        //std::cout << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;

        //std::cout << mn_site_cat << std::endl;
        //std::cout << mv_sp_in_site[0] << std::endl;
        //std::cout << "--------------------------" << std::endl;

        int k = 0;
        for (int i = 0; i < mn_site_cat; i++) //loop over number of site categories, (s), (c), etc.
            for (int j = 0; j < mv_sp_in_site[i]; j++, k++) //loop over number of species in site category, N-s, O-s, etc. includes empty sites as species

                // overwriting the surface species number density, to coverage times site density specified in input
                mv_nd(m_ns+k) *= mv_sigma(i);  // All these better be done in SurfaceProperties
                //std::cout << "--------------------------" << std::endl;
                //std::cout << "  ND    " << mv_nd <<  std::endl;
                //std::cout << "--------------------------" << std::endl;
        

        // In the case of surface at steady state is considered,
        // it updates the mv_nd.tail(mn_site_sp)
        bool is_surf_cov_steady_state = m_surf_props.isSurfaceCoverageSteady();
        if (is_surf_cov_steady_state)
            computeSurfaceSteadyStateCoverage();

        // Constant rate times densities of species
        mv_rate = mv_kf;



        //std::cout << " 0     " << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_kf " << mv_rate << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_nd " << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;

        m_reactants.multReactions(mv_nd, mv_rate);

        //std::cout << " 1     " << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_kf " << mv_rate << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_nd " << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;

        mv_nd.setZero();
        m_reactants.incrSpecies(mv_rate, mv_nd);

        //std::cout << " 2     " << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_kf " << mv_rate << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_nd " << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;


        m_irr_products.decrSpecies(mv_rate, mv_nd);

        //std::cout << " 3     " << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_kf " << mv_rate << std::endl;
        //std::cout << "--------------------------" << std::endl;
        //std::cout << "mv_nd " << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;

        //std::cout << "Surf Cov CompRates (mv_X init) " << mv_X << std::endl;

        // Multiply by molar mass
        return (mv_nd.cwiseProduct(mv_mass)).head(m_ns);
    }

//=============================================================================

    Eigen::VectorXd computeRatesPerReaction()
    {

        //std::cout << " Compute Rates per Reaction " << std::endl;


        // Note mv_nd is density here
        mv_rhoi = m_surf_state.getSurfaceRhoi();

    	// Getting the kfs with the initial conditions
        for (int i_r = 0; i_r < m_nr; ++i_r) {
            mv_kf(i_r) =
                v_reactions[i_r]->getRateLaw()->forwardReactionRateCoefficient(
            		mv_rhoi, m_surf_state.getSurfaceT());
        }

        //std::cout << "--------------------------" << std::endl;
        //std::cout << "Initial kf " << std::endl;
        //std::cout << mv_kf << std::endl;
        //std::cout << "--------------------------" << std::endl;

        m_thermo.convert<RHO_TO_CONC>(mv_rhoi.data(), mv_nd.data());
        mv_nd.head(m_ns) *= NA;
        mv_nd.tail(mn_site_sp) = m_surf_props.getSurfaceSiteCoverageFrac();

        // here the surface coverages are as expected from the initialization, in ref to the Minn FRC implementation
        //std::cout << "------------0--------------" << std::endl;
        //std::cout << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;

        //std::cout << "Surf Cov 2 (mv_X init) " << mv_X << std::endl;

        int k = 0;
        for (int i = 0; i < mn_site_cat; i++)
            for (int j = 0; j < mv_sp_in_site[i]; j++, k++)
                mv_nd(m_ns+k) *= mv_sigma(i);  // All these better be done in SurfaceProperties

        //std::cout << "-------------1-------------" << std::endl;
        //std::cout << mv_nd << std::endl;
        //std::cout << "--------------------------" << std::endl;

        // In the case of surface at steady state,
        // it updates the mv_nd.tail(mn_site_sp)
        bool is_surf_cov_steady_state = m_surf_props.isSurfaceCoverageSteady();
        if (is_surf_cov_steady_state)
            computeSurfaceSteadyStateCoverage();

        //std::cout << "Surf Cov 3 (mv_X init) " << mv_X << std::endl;

        // double B = mv_sigma(0);
        // cout << scientific << setprecision(100);
        // std::cout << "mpp kf1 = " << mv_kf(0)*B << " kf2 = " << mv_kf(1)*B << std::endl;

        mv_rate = mv_kf;




        m_reactants.multReactions(mv_nd, mv_rate);

        //std::cout << "--------------------------" << std::endl;
        //std::cout << " CPRR " << mv_rate << std::endl;
        //std::cout << "--------------------------" << std::endl;

        //std::cout << "Surf Cov 4 (mv_X init) " << mv_X << std::endl;

        return mv_rate;
    }


//=============================================================================

    int nSurfaceReactions(){ return m_nr; }

//=============================================================================

    void updateFunction(VectorXd& v_X)
    {

        //std::cout << " updateFunction " << std::endl;

         //std::cout << "v_X = " << v_X << std::endl;
         //std::cout << "mv_nd = " << mv_nd << std::endl;
        
         //mv_f.head(m_ns) = mv_nd.head(m_ns);
        mv_nd.tail(mn_site_sp) = v_X;

         //std::cout << "mn site sp = " << mn_site_sp << std::endl;
        //std::cout << "kf " << mv_kf << std::endl;

        mv_rate = mv_kf;
        m_reactants.multReactions(mv_nd, mv_rate);


        //std::cout << "ND check = " << mv_nd << std::endl;
        //std::cout << "mv_rate = " << mv_rate << std::endl;
        // std::cout << "mv_f = " << mv_f << std::endl;

        mv_f.setZero();

        //std::cout << "F1 = " << mv_f << std::endl;
        //std::cout << "mv_rate = " << mv_rate << std::endl;

    	m_reactants.incrSpecies(mv_rate, mv_f);

        //std::cout << "F2 = " << mv_f << std::endl;
        //std::cout << "mv_rate = " << mv_rate << std::endl;

    	m_irr_products.decrSpecies(mv_rate, mv_f);

        //std::cout << "F3 = " << mv_f << std::endl;
        //std::cout << "mv_rate = " << mv_rate << std::endl;

        //std::cout << "F = \n" << mv_f << std::endl;
    }
//=============================================================================

    void updateJacobian(VectorXd& v_X)
    {

        //std::cout << " Update Jacobian " << std::endl;

    	mv_f_unpert = mv_f;
        // Make pert

        //std::cout << "F = \n" << mv_f << std::endl;

        for (int i = 0; i < mn_site_sp; ++i) {
            
            double m_X_unpert = v_X(i);
            double sign_hold;
            if (v_X[i] < 1)
                sign_hold = -1;
            else
                sign_hold = 1;

            double pert = 1.e-7 * max(abs(v_X[i]), 1.e-20) * sign_hold;


            //double pert = v_X(i) * m_pert;

           //std::cout << "v_X UpJac " << v_X(i) << std::endl;
           // std::cout << "pert" << pert << std::endl;

           

            //std::cout << "s_check = \n" << s_check << std::endl;

    		v_X(i) += pert;

           //std::cout << "v_X" << v_X << std::endl;
           //std::cout << "mv_f" << mv_f << std::endl;

            // Update Jacobian column
            updateFunction(v_X);

            //std::cout << "v_X" << v_X << std::endl;
            //std::cout << "mv_f @ i" << i << mv_f << std::endl;
            

            m_jac.col(i) =
                (mv_f.tail(mn_site_sp)-mv_f_unpert.tail(mn_site_sp)) / pert;
            
           // std::cout << "jac1 = \n" << m_jac.col(i) << std::endl;
            //std::cout << "mf_mod" << mv_f.tail(mn_site_sp) << std::endl;
            //std::cout << "mf_unpert" << mv_f_unpert.tail(mn_site_sp) << std::endl;
            //std::cout << " pert " << pert << std::endl;

            v_X(i) = m_X_unpert;

            //std::cout << "jac col = \n" << m_jac.col(i) << std::endl;
        }

        //std::cout << "jac = \n" << m_jac << std::endl;
    }

//=============================================================================

    VectorXd& systemSolution()
    {
        //std::cout << " System Solution " << std::endl;
        //std::cout << "-------------------------" << std::endl;
         //std::cout << "X = \n" << mv_X << std::endl;

         //std::cout << "jac = \n" << m_jac << std::endl;

        double a = (m_jac.diagonal()).cwiseAbs().maxCoeff();

        //std::cout << "jac a " << a << std::endl;

        // std::cout << "jac = \n" << m_jac + a*MatrixXd::Ones(mn_site_sp,mn_site_sp) << std::endl;
        // std::cout << "f = \n" << mv_f_unpert.tail(mn_site_sp) << std::endl;

        mv_dX = (m_jac + a*MatrixXd::Ones(mn_site_sp,mn_site_sp)).
            fullPivLu().solve(mv_f_unpert.tail(mn_site_sp));
        // m_jac(0,0) = 1;
        // m_jac(0,1) = 1;
        // mv_f_unpert(0) = mv_sigma(0);
        // mv_dX = (m_jac).
        //     fullPivLu().solve(mv_f_unpert.tail(mn_site_sp));
         
         //std::cout << "-------------------------" << std::endl;
         //std::cout << "DX = \n" << mv_dX << std::endl;
         //std::cout << "-------------------------" << std::endl;
          
         
        // double in; std::cin >> in;


        return mv_dX;
    }
//=============================================================================

    double norm()
    {
        //std::cout << " Norm " << std::endl;
        //std::cout << "Norm Call" << std::endl;
        //double NormCheck = mv_f_unpert.lpNorm<Eigen::Infinity>();
        //std::cout << "Norm Check " << NormCheck << std::endl;
        return mv_f_unpert.tail(mn_site_sp).lpNorm<Eigen::Infinity>();
    }
//=============================================================================
private:
    void computeSurfaceSteadyStateCoverage(){

        //std::cout << " computeSurfaceSteadyStateCoverage " << std::endl;
        // mv_nd.head(m_ns) = 1.;
        // mv_X = mv_nd.tail(mn_site_sp);
        //mv_X.setConstant(5.e17);
        //mv_X.setConstant(3.011e18);

        //mv_X.setZero(); @TODO
        //for (int i = 0; i < mv_sigma.size(); ++i) {
        //    mv_X(0) +=  mv_sigma(i) / mv_sigma.size();
        //    mv_X(element_of_site(i)) +=  mv_sigma(i) / mv_sigma.size();
        //    }

        //std::cout << "Surf Cov CSSSC " << mv_X << std::endl;

        //std::cout << "mv_simga(0) " << mv_sigma(0) << std::endl;
        //std::cout << "m site sp" << mn_site_sp << std::endl;

        // mv_X.setConstant(mv_sigma(0) / (mv_sigma.size() + 1)); // not sure why its initialized like that?
        mv_X.setConstant(mv_sigma(0) / (mn_site_sp)); // initial coverage just the total site density divided by total site species (this current setup probably won't work for pyrolyzing runs, where you have more than 1 site type)

        //std::cout << "Surf Cov CSSSC " << mv_X << std::endl;

        //std::cout << "Before X = \n" << mv_X << std::endl;
        mv_X = solve(mv_X);

        //std::cout << "PreTol Sol'n = \n" << mv_X << std::endl;
        

        applyTolerance(mv_X);
        //std::cout << "-----------------------" << std::endl;
        //std::cout << "Final soln = \n" << mv_X << std::endl;
        //std::cout << "-----------------------" << std::endl;

        // Setting up the SurfaceSiteCoverage.
        // This is not essential for efficiency.
        mv_nd.tail(mn_site_sp) = mv_X;
        m_surf_props.setSurfaceSiteCoverageFrac(mv_X/mv_sigma(0));
    }

//=============================================================================
    inline void applyTolerance(Eigen::VectorXd& v_x) const {

        //std::cout << " Apply Tolerance " << std::endl;

        for (int i = 0; i < v_x.size(); i++)
            if ((abs(v_x(i)) / mv_sigma(0)) < m_tol) 
                v_x(i) = 0.; // changed back to abs for now, but possible some values more negative than the tolerance were getting picked up
    }

//=============================================================================
private:
    SurfaceProperties& m_surf_props;

    const size_t m_ns;
    const size_t m_nr;

    bool is_surf_steady_state;

    VectorXd mv_kf;
    VectorXd mv_rate;

    // Surface properties
    const size_t mn_site_sp;
    const size_t mn_site_cat;
    vector<int> mv_sp_in_site;
    VectorXd mv_sigma;

    VectorXd mv_mass;
    VectorXd mv_rhoi;
    VectorXd mv_nd;

    // For the steady state coverage solver.
    const double m_tol;
    double m_pert;
    VectorXd mv_X;
    VectorXd mv_dX;
    VectorXd mv_work;
    VectorXd mv_f;
    VectorXd mv_f_unpert;
    Eigen::MatrixXd m_jac;

    GSIStoichiometryManager m_reactants;
    GSIStoichiometryManager m_irr_products;
};

ObjectProvider<GSIRateManagerDetailed, GSIRateManager>
    gsi_rate_manager_detailed_mass("detailed_mass");

ObjectProvider<GSIRateManagerDetailed, GSIRateManager>
    gsi_rate_manager_detailed_mass_energy("detailed_mass_energy");

    } // namespace GasSurfaceInteraction
} // namespace Mutation
