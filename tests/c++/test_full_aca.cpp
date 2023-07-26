/*
 * Copyright 2014-2018 von Karman Institute for Fluid Dynamics (VKI)
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

// To ensure M++ matches with Prata need use the MB beam temp in all calculations related to the mean
// thermal speed, other temperatures correspond to the surface temperature

#include "mutation++.h"
#include "Configuration.h"
#include "TestMacros.h"
#include <catch2/catch.hpp>
#include <Eigen/Dense>

#include "SurfaceProperties.h"

using namespace Mutation;
using namespace Catch;
using namespace Eigen;

TEST_CASE("Full ACA tests.","[gsi]")
{
    const double tol = 100. * std::numeric_limits<double>::epsilon();
    const double tol_det = 1.e2 * std::numeric_limits<double>::epsilon();

    Mutation::GlobalOptions::workingDirectory(TEST_DATA_FOLDER);

    SECTION("Surface Species and Coverage.")
    {
        // Setting up M++
        MixtureOptions opts("Full_ACA_NASA9_ChemNonEq1T");
        Mixture mix(opts);

        CHECK(mix.nSpecies() == 9);

        // Check global options
        CHECK(mix.nSurfaceReactions() == 0);
        CHECK(mix.getSurfaceProperties().nSurfaceSpecies() == 5);
        CHECK(mix.getSurfaceProperties().nSiteSpecies() == 4);

        // Check Species
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("N-s") == 10);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("O-s") == 11);
	CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("O*-s") == 12);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("C-b") == 13);

        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("s") == 9);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("b") == -1);

        // Check surface species association with gaseous species, these indices
	// are relative to the species defined in the mixture file
        CHECK( mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("N-s")) == 0);
        CHECK( mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O-s")) == 1);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O*-s")) == 1);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("C-b")) == 5);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("s")) == -2);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("b")) == -1);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(100) == -1);

        // Check site species map correctly to the site category
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("N-s")) == 0);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O*-s")) == 0);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O-s")) == 0);

        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("C-b")) == -1);

        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("s")) == 0);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("b")) == -1);

        CHECK(mix.getSurfaceProperties().nSiteDensityInCategory(0) == 6.022e18);
        CHECK(mix.getSurfaceProperties().nSiteDensityInCategory(1) == -1);

    }

    SECTION("ACA Model.")
    {
	//Set up M++
        MixtureOptions optsACA("Full_FRC_ACA_NASA9_ChemNonEq1T");
	Mixture mixACA(optsACA);

	size_t ns = 9;
	size_t nss = 4;
        size_t nr = 20;
	size_t nsf = 5; // includes bulk carbon

	CHECK(mixACA.nSpecies() == ns);
	CHECK(mixACA.nSurfaceReactions() == nr);
	CHECK(mixACA.getSurfaceProperties().nSurfaceSpecies() == nsf);
	CHECK(mixACA.getSurfaceProperties().nSiteSpecies() == nss);

        const size_t iN = 0;
        const size_t iO = 1;
	const size_t iN2 = 3;
	const size_t iO2 = 4;
	const size_t iCO = 6;
        const size_t iCN = 7;
	const size_t iCO2 = 8;
	
	const int set_state_rhoi_T = 1;

        const double tol = 10e-4;

	ArrayXd v_rhoi(ns);
	ArrayXd num_den_i(ns);
        ArrayXd wdot(ns); ArrayXd wdotmpp(ns);
        ArrayXd rates(nr); ArrayXd ratesmpp(nr);
	ArrayXd kfmpp(nr); 
        wdot.setZero(); wdotmpp.setZero();

	ArrayXd mm = mixACA.speciesMw(); 

	//std::cout << "T = " << T << " P = " << P << std::endl;

	CHECK(mixACA.getSurfaceProperties().isSurfaceCoverageSteady() == true);

	ArrayXd v_surf_cov_mpp_frac(mixACA.getSurfaceProperties().nSurfaceSpecies());
        ArrayXd v_surf_cov_frac(mixACA.getSurfaceProperties().nSurfaceSpecies()); 
	
	// Equilibrium Surface
	double P;
	double T;	
	
        //double P = 1.e-5;
        //double dP = 10.;
        //double T; // K
        //double dT = 1000.; // K

        for (int i = 0; i < 2; i++) {
		
		if (i == 0) {
			P = 1600;
			T = 1000;
		}
		if (i == 1) { 
			P = 10000;
			T = 2000;
		}
		
		// EQ at loop T and P
                mixACA.equilibrate(T, P);
                mixACA.densities(v_rhoi.data());

		for (int q = 0; q < ns; q++) {
			num_den_i[q] = mixACA.X()[q] * mixACA.numberDensity();	
		}

		//-------------------------------------------------------
		//Atomic nitrogen reactions first, Prata claims the behavior is abstracted for atomic O
		// and atomic N arriving at the surface
		//-------------------------------------------------------

		// Compute surface state based on EQ mass dens
                mixACA.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
                double nN = mixACA.X()[iN] * mixACA.numberDensity();
		double n_conc = nN/NA;

                mixACA.surfaceReactionRatesPerReaction(ratesmpp.data());
                mixACA.surfaceReactionRates(wdotmpp.data());
		mixACA.forwardRateCoefficients(kfmpp.data());
		
		// Compute 'analytical' rate coefficients for atomic nitrogen reactions
                const double B = 1e-5;
                double F_N = 0.25 * sqrt(8 * KB * T / (PI * (mm(iN)/NA)));
		
		// Adsorption and desorption rates
                double kN1 = (F_N/B)*exp(-2500./T); //kN1
                double kN2 = (2 * PI * mm(iN) * KB * KB * T * T) / (NA* HP * HP * HP * B);
		kN2 *= exp(-73971.6/T); // kN2 
		
		// Eley-Rideal rates (gas phase + surface phase)
                double kN4 = (F_N/B) * 0.5 * exp(-2000./T); // kN4
                double kN3 = (F_N/B) * 1.5 * exp(-7000./T);  // kN3

		// Langmuir-Hishelwood rates (surface phase only)
		double F2_N = sqrt( PI * KB * T / (2*(mm(iN)/NA)));
		double kN5 = sqrt(NA/B) * F2_N * 0.1 * exp(-21000. / T); // kN5
		double kN6 = 1e+8 * exp(-20676. / T); // kN6

		// Compute consolidated rates as based on analytical expression in Prata et al
		double A_3 = 0.;
		double B_3 = kN1 * n_conc;
		double C_3 = 2.*kN5;
		double D_3 = kN2 + kN6 + (kN3 + kN4) * n_conc;
		
		//----------------------------------------------------------
		//Reactions involving atomic O, low energy
		//----------------------------------------------------------
		
		double nO = mixACA.X()[iO] * mixACA.numberDensity();
		double o_conc = nO/NA;
		double F_O = 0.25 * sqrt(8. * KB * T / (PI * (mm(iO)/NA)));		

		//Adsorption and desorption rates
		double kO1 = (F_O / B) * 0.3; //kO1
		double kO2 = (2. * PI * (mm(iO)/NA) * KB * KB * T * T) / (NA* HP * HP * HP * B); 
		kO2 *= exp(-44277. / T); //kO2

		// Eley-Rideal
		double kO3 = (F_O / B) * 100. *  exp(-4000. / T); //kO3
		double kO4 = (F_O / B) *  exp(-500. / T); //kO4
		
		// Langmuir-Hinshelwood
		double F2_O = sqrt( PI * KB * T / (2*(mm(iO)/NA)));
		double kO9 = sqrt(NA / B) * F2_O * 5e-5 * exp(-15000. / T); //kO9
		
		//----------------------------------------------------------
		//Reactions involving atomic O*, high energy
		//----------------------------------------------------------
		
		//Adsorption and desorption rates
		double kO5 = (F_O / B) * 0.7; //kO5
		double kO6 =(2 * PI * (mm(iO)/NA) * KB * KB * T * T) / (NA* HP * HP * HP * B); 
		kO6 *= exp(-96500. / T); //kO6

		// Eley-Rideal
		double kO7 = (F_O / B) * 1000. *  exp(-4000. / T); //kO7

		// Langmuir-Hinshelwood
		double kO8 = sqrt(NA / B) * F2_O * 1e-3 * exp(-15000. / T); //kO8
		
		//----------------------------------------------------------
		//Reactions involving atomic O2
		//----------------------------------------------------------
		
		//Adsorption and desorption rates
		double nO2 = mixACA.X()[iO2] * mixACA.numberDensity();
		double o2_conc = nO2/NA;
		double F_O2 = 0.25 * sqrt(8. * KB * T / (PI * (mm(iO2)/NA)));
		double kOx1 = (F_O2 / (B * B)) * exp(-8000. / T); //kOx1
		double kOx4 = (F_O2 / (B * B)) * exp(-8000. / T); //kOx4
		
		// Eley-Rideal
		double kOx2 = (F_O2 / B) * 100. *  exp(-4000. / T); //kOx2
		double kOx3 = (F_O2 / B) *  exp(-500. / T); //kOx3
		double kOx5 = (F_O2 / B) * 1000. *  exp(-4000. / T); //kOx5

		// Analytical expressions for the consolidated rates for O, O*, etc
		double A_1 = 2. * kOx1 * o2_conc;
		double B_1 = kO1 * o_conc;
		double C_1 = 2. * kO9;
		double D_1 = kO2 + (kO3 + kO4)*o_conc + (kOx2 + kOx3)*o2_conc;
		double A_2 = 2. * kOx4 * o2_conc;
		double B_2 = kO5 * o_conc;
		double C_2 = 2. * kO8;
		double D_2 = kO6 + kO7*o_conc + kOx5*o2_conc;

		//---------------------------------------------------------
		// Hard coded results for surface coverage from an external script for the Minn FRC model
		//---------------------------------------------------------
		
		ArrayXd v_surf_cov_ext(nss);
		ArrayXd v_surf_cov_ext_frac(nss);
		
		if (i == 0) // for P = 1600 Pa, T = 1000 K, units [mol/m3]
			v_surf_cov_ext(0) = 9.992398692668828e-06; // s
			v_surf_cov_ext(1) = 4.198322580234444e-09; // N-s
			v_surf_cov_ext(2) = 2.968536294455605e-09; // O-s
			v_surf_cov_ext(3) = 4.344484564819589e-10; // O*-s

		if (i == 1) // for P = 10,000 Pa, T = 2000 K, units [mol/m3]
			v_surf_cov_ext(0) = 9.971063640498129e-06; // s
			v_surf_cov_ext(1) = 4.153719292961034e-10; // N-s
			v_surf_cov_ext(2) = 2.567725173514249e-08; // O-s
			v_surf_cov_ext(3) = 2.843735837433133e-09; // O*-s

		for (int j = 0; j < nss; j++) {
			v_surf_cov_ext_frac[j] = v_surf_cov_ext[j]/B;
		}
		
		v_surf_cov_mpp_frac = mixACA.getSurfaceProperties().getSurfaceSiteCoverageFrac();
		
		std::cout << "T " << T << " P " << P << std::endl;
		std::cout << "-----------------------------------" << std::endl;
		std::cout << "External Surf Frac " << std::endl;
		std::cout << v_surf_cov_ext_frac << std::endl;
		std::cout << "MPP Surf Frac " << std::endl;
		std::cout << v_surf_cov_mpp_frac << std::endl;
		

        }

    }

}
