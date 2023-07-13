/**
 * @file GSIRateLawGammaT.cpp
 *
 * @brief Class which computes the reaction rate constant for a surface Eley-Rideal reaction for use in the ZA surface mech. 
 * Note I forwent including the sum of the stoich coeffs since it's 1 for all reactions in this set.
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


#include "Thermodynamics.h"
#include "Transport.h"

#include "AutoRegistration.h"
#include "Utilities.h"

#include "GSIRateLaw.h"
#include "SurfaceProperties.h"

using namespace Mutation::Utilities::Config;

namespace Mutation {
    namespace GasSurfaceInteraction {

class GSIRateLawZAERArrhenius : public GSIRateLaw
{
public:
    GSIRateLawZAERArrhenius(ARGS args)
        : GSIRateLaw(args),
          m_surf_props(args.s_surf_props),
          mv_react(args.s_reactants),
          pos_T_trans(0),
          pos_gas_r(0),
          pos_site_r(1)
    {
        assert(args.s_node_rate_law.tag() == "ZAER_arrhenius");

        args.s_node_rate_law.getAttribute( "pre_exp", m_pre_exp,
            "The sticking coeffcient probability for the reaction "
            "should be provided.");
        args.s_node_rate_law.getAttribute( "beta", m_beta, 
	    "Temperature exponent should be provided.");
	args.s_node_rate_law.getAttribute( "ER_eff", m_ER_eff, 
	    "ER reaction efficiency should be provided.");
        args.s_node_rate_law.getAttribute( "T", m_T_act,
            "The activation temperature for the reaction "
            "should be provided for an adsorption reaction.");

        // For the gas in the reactants
        m_idx_gas = mv_react[pos_gas_r];
        // Error if m_idx_gas > ns

        // For the sites
        int idx_site = mv_react[pos_site_r];
        // Error if idx_site > ns

        m_site_categ = m_surf_props.siteSpeciesToSiteCategoryIndex(idx_site);
        m_n_sites = m_surf_props.nSiteDensityInCategory(m_site_categ);
    }

//==============================================================================

    ~GSIRateLawZAERArrhenius( ){ }

//==============================================================================

    double forwardReactionRateCoefficient(
        const Eigen::VectorXd& v_rhoi, const Eigen::VectorXd& v_Tsurf) const
    {
    	const double Tsurf = v_Tsurf(pos_T_trans);

    	const int set_state_with_rhoi_T = 1;
        m_thermo.setState(v_rhoi.data(), v_Tsurf.data(), set_state_with_rhoi_T);
        const double thermal_speed =
            m_transport.speciesThermalSpeed(m_idx_gas);

        return m_pre_exp * thermal_speed /
            (4. * m_n_sites) * m_ER_eff *  exp(m_beta * log(Tsurf)) * exp(-m_T_act / Tsurf);
    }

private:
    const size_t pos_T_trans;
    const size_t pos_gas_r;
    const size_t pos_site_r;

    int m_idx_gas;
    int m_site_categ;
    double m_n_sites;

    double m_pre_exp;
    double m_T_act;
    double m_beta;
    double m_ER_eff;

    const std::vector<int>& mv_react;
    const SurfaceProperties& m_surf_props;
};

ObjectProvider<
    GSIRateLawZAERArrhenius, GSIRateLaw>
    gsi_rate_law_zaer_arrhenius("ZAER_arrhenius");

    } // namespace GasSurfaceInteraction
} // namespace Mutation
