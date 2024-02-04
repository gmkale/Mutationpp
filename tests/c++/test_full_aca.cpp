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


TEST_CASE("Full ACA tests.", "[gsi]")
{
	const double tol = 100. * std::numeric_limits<double>::epsilon();
	const double tol_det = 1.e2 * std::numeric_limits<double>::epsilon();

	Mutation::GlobalOptions::workingDirectory(TEST_DATA_FOLDER);

	SECTION("Surface Species and Coverage.")
	{
		// Setting up M++
		MixtureOptions opts("Full_FRC_ACA_NASA9_ChemNonEq1T");
		Mixture mix(opts);

		CHECK(mix.nSpecies() == 9);

		// Check global options
		CHECK(mix.nSurfaceReactions() == 20);
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
		CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
			mix.getSurfaceProperties().surfaceSpeciesIndex("N-s")) == 0);
		CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
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
	SECTION("Check rates.") // Verify the coverage solution here compares to known solution
	{
		//Set up M++
		MixtureOptions opts_ACA("Full_FRC_ACA_NASA9_ChemNonEq1T");
		Mixture mix(opts_ACA);

		const double B = 6.022e18; //site density as specified by Prata

		const size_t iN = 0;
		const size_t iO = 1;
		const size_t iN2 = 3;
		const size_t iO2 = 4;
		const size_t iCO = 6;
		const size_t iCN = 7;
		const size_t iCO2 = 8;

		ArrayXd v_rhoi(mix.nSpecies());
		ArrayXd ratesmpp(mix.nSurfaceReactions());
		ArrayXd wdotmpp(mix.nSpecies());
		ArrayXd mm = mix.speciesMw();

		const int set_state_rhoi_T = 1;

		CHECK(mix.getSurfaceProperties().isSurfaceCoverageSteady() == true);

		ArrayXd v_surf_cov_mpp_frac(mix.getSurfaceProperties().nSurfaceSpecies()); // M++ coverage fraction
		ArrayXd v_surf_cov_frac(mix.getSurfaceProperties().nSurfaceSpecies()); // Reference coverage fraction

		//std::cout << "--------------------------" << std::endl;
		//std::cout << "MPP Coverage " << v_surf_cov_mpp_frac << std::endl;
		//std::cout << "--------------------------" << std::endl;

		// check initilization of coverage fraction
		//v_surf_cov_mpp_frac = mixO.getSurfaceProperties().getSurfaceSiteCoverageFrac();
		//std::cout << "--------------------------" << std::endl;
		//std::cout << "MPP Coverage " << v_surf_cov_mpp_frac << std::endl;
		//std::cout << "--------------------------" << std::endl;

		//double P = 101325.; // simulation pressure (this can be a bit questionable depending on the conditions used)
		double T_start = 800.; // starting temp for loop
		double dT = 100.;
		double T;

		size_t n_temps = 17;
		ArrayXd T_surf(n_temps);
		ArrayXd ref_surf_cov_s(n_temps);
		ArrayXd ref_surf_cov_O(n_temps);
		ArrayXd ref_surf_cov_Ost(n_temps);
		ArrayXd ref_surf_cov_N(n_temps);

		// fill temperatures
		for (int k = 0; k < n_temps; k++) {
			T_surf[k] = T_start + (k * dT);
		}
		
		// Testing computed surface coverage at UT ICP conditions (1 atm)
		// for T = 800:100:2400 all at P = 1 atm, these values were externally computed
		// empty site coverage
		ref_surf_cov_s[0] = 0.999911440005248;
		ref_surf_cov_s[1] = 0.999818793946886;
		ref_surf_cov_s[2] = 0.999688378685768;
		ref_surf_cov_s[3] = 0.999523100868279;
		ref_surf_cov_s[4] = 0.999327354373843;
		ref_surf_cov_s[5] = 0.999105968424609;
		ref_surf_cov_s[6] = 0.998863636663497;
		ref_surf_cov_s[7] = 0.998604570090356;
		ref_surf_cov_s[8] = 0.998332272180933;
		ref_surf_cov_s[9] = 0.998049561760558;
		ref_surf_cov_s[10] = 0.997759578154337;
		ref_surf_cov_s[11] = 0.997470024076281;
		ref_surf_cov_s[12] = 0.997205477483897;
		ref_surf_cov_s[13] = 0.997030972664088;
		ref_surf_cov_s[14] = 0.997060491823863;
		ref_surf_cov_s[15] = 0.997366525572047;
		ref_surf_cov_s[16] = 0.997828719925199;

		ref_surf_cov_O[0] = 0.000075086487472;
		ref_surf_cov_O[1] = 0.000157727306505;
		ref_surf_cov_O[2] = 0.000275012844468;
		ref_surf_cov_O[3] = 0.000424253212647;
		ref_surf_cov_O[4] = 0.000601392030850;
		ref_surf_cov_O[5] = 0.000801985776677;
		ref_surf_cov_O[6] = 0.001021718949551;
		ref_surf_cov_O[7] = 0.001256696044732;
		ref_surf_cov_O[8] = 0.001503579692349;
		ref_surf_cov_O[9] = 0.001759389538973;
		ref_surf_cov_O[10] = 0.002020133755595;
		ref_surf_cov_O[11] = 0.002275950719234;
		ref_surf_cov_O[12] = 0.002498177630726;
		ref_surf_cov_O[13] = 0.002616071469851;
		ref_surf_cov_O[14] = 0.002510044938805;
		ref_surf_cov_O[15] = 0.002103723595239;
		ref_surf_cov_O[16] = 0.001516092395212;

		ref_surf_cov_Ost[0] = 1.34735072792377e-05;
		ref_surf_cov_Ost[1] = 2.34787466090319e-05;
		ref_surf_cov_Ost[2] = 3.66084696874001e-05;
		ref_surf_cov_Ost[3] = 5.26459164849449e-05;
		ref_surf_cov_Ost[4] = 7.12535469677119e-05;
		ref_surf_cov_Ost[5] = 9.20452235555951e-05;
		ref_surf_cov_Ost[6] = 0.000114639584605162;
		ref_surf_cov_Ost[7] = 0.000138703666676863;
		ref_surf_cov_Ost[8] = 0.000163997379099120;
		ref_surf_cov_Ost[9] = 0.000190427410108913;
		ref_surf_cov_Ost[10] = 0.000218115125669221;
		ref_surf_cov_Ost[11] = 0.000247478468988474;
		ref_surf_cov_Ost[12] = 0.000279324633819169;
		ref_surf_cov_Ost[13] = 0.000314945689905738;
		ref_surf_cov_Ost[14] = 0.000356183950358963;
		ref_surf_cov_Ost[15] = 0.000405372509786366;
		ref_surf_cov_Ost[16] = 0.000465086598378484;

		ref_surf_cov_N[0] = 4.93638169582648e-18;
		ref_surf_cov_N[1] = 1.05532412411257e-15;
		ref_surf_cov_N[2] = 7.72121416413486e-14;
		ref_surf_cov_N[3] = 2.58897349525821e-12;
		ref_surf_cov_N[4] = 4.83390237928231e-11;
		ref_surf_cov_N[5] = 5.75158772454514e-10;
		ref_surf_cov_N[6] = 4.80234711266628e-09;
		ref_surf_cov_N[7] = 3.01982349602674e-08;
		ref_surf_cov_N[8] = 1.50747619640428e-07;
		ref_surf_cov_N[9] = 6.21290359286901e-07;
		ref_surf_cov_N[10] = 2.17296439922690e-06;
		ref_surf_cov_N[11] = 6.54673549610173e-06;
		ref_surf_cov_N[12] = 1.70202515584389e-05;
		ref_surf_cov_N[13] = 3.80101761554031e-05;
		ref_surf_cov_N[14] = 7.32792869721738e-05;
		ref_surf_cov_N[15] = 0.000124378322928087;
		ref_surf_cov_N[16] = 0.000190101081210507;

		// initialize comparison arrays
		ArrayXd s_log(n_temps); s_log.setZero();
		ArrayXd N_log(n_temps); N_log.setZero();
		ArrayXd O_log(n_temps); O_log.setZero();
		ArrayXd Ost_log(n_temps); Ost_log.setZero();
		ArrayXd rates(mix.nSurfaceReactions()); rates.setZero();

		//check this gives the same flux to surface as expected?
		// int count_press = 7;
		// ArrayXd P(7);
		// P[0] = 1; P[1] = 10; P[2] = 100; P[3] = 1000; P[4] = 10000; P[5] = 50000; P[6] = 100000;
		double P = 101325.;
		
		for (int j = 0; j < 1; j++) {

			//std::cout << P[j] << std::endl;

			for (int i = 0; i < n_temps; i++) {

				//T_surf = 1500;
				//P = 1600;

				// initialize surface state based on EQ number densities, for this there's no distinction between 
				mix.equilibrate(T_surf[i], P);
				mix.densities(v_rhoi.data());

				// Print mixture densities for comparison to the call from Matlab
				//std::cout << "--------------------------" << std::endl;
				//std::cout << " T " << T_surf[i] << " P " << P << std::endl;
				//std::cout << "EQ Species Densities \n " << v_rhoi << std::endl;
				//std::cout << "--------------------------" << std::endl;

				// check initial surface coverage
				//v_surf_cov_mpp_frac = mix.getSurfaceProperties().getSurfaceSiteCoverageFrac();
				//std::cout << "--------------------------" << std::endl;
				//std::cout << "MPP Coverage: Initial \n" << v_surf_cov_mpp_frac << std::endl;
				//std::cout << "--------------------------" << std::endl;

				// set the surface state
				mix.setSurfaceState(v_rhoi.data(), &T_surf[i], set_state_rhoi_T);
				// surface production rates per reaction rather than per species, I ASSUME units of kg/m2-s
				mix.surfaceReactionRatesPerReaction(ratesmpp.data());
				// surface production rates, units of kg/m2-s
				mix.surfaceReactionRates(wdotmpp.data());

				//std::cout << "s" << std::endl;
				//std::cout << "N" << std::endl;
				//std::cout << "O" << std::endl;
				//std::cout << "O*" << std::endl;

				// get new surf coverage based on updated surf state
				v_surf_cov_mpp_frac = mix.getSurfaceProperties().getSurfaceSiteCoverageFrac();
				//std::cout << "--------------------------" << std::endl;
				//std::cout << "MPP Coverage: Updated \n" << v_surf_cov_mpp_frac << std::endl;
				//std::cout << "--------------------------" << std::endl;

				// COMPARE COVERAGE SOLUTION (EXTERNAL SOL'N TO M++)

				// COMPARE PRODUCTION RATES FOR O RELATED SURFACE REACTIONS (against the analytical Jacobian constructed external to M++)
				//----------------------------------------------------------
				//Reactions involving atomic O, low energy, and reactions involving atomic O*, high energy
				//----------------------------------------------------------
				double nO = v_rhoi[iO] * NA;
				double F_O = 0.25 * sqrt((8. * KB * T_surf[i]) / (PI * (mm(iO) / NA)));
				double F2_O = sqrt(PI * KB * T_surf[i] / (2. * (mm(iO) / NA)));

				//Adsorption and desorption rates
				double kO1 = (F_O / B) * 0.3; //kO1
				double kO2 = (2. * PI * (mm(iO) / NA) * KB * KB * T_surf[i] * T_surf[i]) / (NA * HP * HP * HP * B) * exp(-44277. / T_surf[i]); //kO2
				// Eley-Rideal
				double kO3 = (F_O / B) * 100. * exp(-4000. / T_surf[i]); //kO3
				double kO4 = (F_O / B) * exp(-500. / T_surf[i]); //kO4
				//Adsorption and desorption rates
				double kO5 = (F_O / B) * 0.7; //kO5
				double kO6 = (2. * PI * (mm(iO) / NA) * KB * KB * T_surf[i] * T_surf[i]) / (NA * HP * HP * HP * B) * exp(-96500. / T_surf[i]); //kO6
				// Eley-Rideal
				double kO7 = (F_O / B) * 1000. * exp(-4000. / T_surf[i]); //kO7
				// Langmuir-Hinshelwood
				double kO8 = sqrt(1.0 / B) * F2_O * 1.e-3 * exp(-15000. / T_surf[i]); //kO8
				double kO9 = sqrt(1.0 / B) * F2_O * 5.e-5 * exp(-15000. / T_surf[i]); //kO9

				////----------------------------------------------------------
				//// Rates for O atom reactions in Minn FRC model
				////----------------------------------------------------------
				rates(6) = kO1 * nO * (ref_surf_cov_s(i) * B);
				rates(7) = kO2 * (ref_surf_cov_O(i) * B);
				rates(8) = kO3 * nO * (ref_surf_cov_O(i) * B);
				rates(9) = kO4 * nO * (ref_surf_cov_O(i) * B);
				rates(10) = kO5 * nO * (ref_surf_cov_s(i) * B);
				rates(11) = kO6 * (ref_surf_cov_Ost(i) * B);
				rates(12) = kO7 * nO * (ref_surf_cov_Ost(i) * B);
				rates(13) = kO8 * (ref_surf_cov_Ost(i) * B) * (ref_surf_cov_Ost(i) * B);
				rates(14) = kO9 * (ref_surf_cov_O(i) * B) * (ref_surf_cov_O(i) * B);

				////----------------------------------------------------------
				//// Rate comparison for O atom reactions in Minn FRC model
				////----------------------------------------------------------
				//std::cout << "--------------------------" << std::endl;
				//std::cout << "Rates for O atom reactions \n" << std::endl;
				//std::cout << rates(6) << " " << ratesmpp(6) << std::endl;
				//std::cout << rates(7) << " " << ratesmpp(7) << std::endl;
				//std::cout << rates(8) << " " << ratesmpp(8) << std::endl;
				//std::cout << rates(9) << " " << ratesmpp(9) << std::endl;
				//std::cout << rates(10) << " " << ratesmpp(10) << std::endl;
				//std::cout << rates(11) << " " << ratesmpp(11) << std::endl;
				//std::cout << rates(12) << " " << ratesmpp(12) << std::endl;
				//std::cout << rates(13) << " " << ratesmpp(13) << std::endl;
				//std::cout << rates(14) << " " << ratesmpp(14) << std::endl;
				//std::cout << "--------------------------" << std::endl;

				//// @TODO add check loop for the above rate expressions

				//// COMPARE PRODUCTION RATES FOR N RELATED SURFACE REACTIONS (against results from analytical Jacobian constructed external to M++)
				////----------------------------------------------------------
				////Reactions involving atomic N
				////----------------------------------------------------------
				double nN = v_rhoi[iN] * NA;
				double F_N = 0.25 * sqrt(8. * KB * T_surf[i] / (PI * (mm(iN) / NA)));
				double F2_N = sqrt((PI * KB * T_surf[i]) / (2. * (mm(iN) / NA)));

				// Adsorption and desorption rates
				double kN1 = (F_N / B) * exp(-2500. / T_surf[i]); //kN1
				double kN2 = (2. * PI * mm(iN) * KB * KB * T_surf[i] * T_surf[i]) / (NA * HP * HP * HP * B);
				kN2 *= exp(-73971.6 / T_surf[i]); // kN2 
				// Eley-Rideal rates (gas phase + surface phase)
				double kN4 = (F_N / B) * 0.5 * exp(-2000. / T_surf[i]); // kN4
				double kN3 = (F_N / B) * 1.5 * exp(-7000. / T_surf[i]);  // kN3
				// Langmuir-Hishelwood rates (surface/adsorbed phase)
				double kN5 = sqrt(1.0 / B) * F2_N * 0.1 * exp(-21000. / T); // kN5
				double kN6 = 1.e+8 * exp(-20676. / T); // kN6

				////----------------------------------------------------------
				//// Rates for N atom reactions in Minn FRC model
				////----------------------------------------------------------
				//rates(9) = kN1 * nN * (ref_surf_cov_s(i) * B);
				//rates(10) = kN2 * (ref_surf_cov_N(i) * B);
				//rates(11) = kN3 * nN * (ref_surf_cov_s(i) * B);
				//rates(12) = kN4 * nN * (ref_surf_cov_s(i) * B);
				//rates(13) = kN5 * (ref_surf_cov_N(i) * B) * (ref_surf_cov_N(i) * B);
				//rates(14) = kN6 * (ref_surf_cov_N(i) * B);

				////----------------------------------------------------------
				//// Rate comparison for N atom reactions in Minn FRC model
				////----------------------------------------------------------
				//std::cout << "--------------------------" << std::endl;
				//std::cout << "Rates for N atom reactions \n" << std::endl;
				//std::cout << rates(9) << " " << ratesmpp(0) << std::endl;
				//std::cout << rates(10) << " " << ratesmpp(1) << std::endl;
				//std::cout << rates(11) << " " << ratesmpp(2) << std::endl;
				//std::cout << rates(12) << " " << ratesmpp(3) << std::endl;
				//std::cout << rates(13) << " " << ratesmpp(4) << std::endl;
				//std::cout << rates(14) << " " << ratesmpp(5) << std::endl;
				//std::cout << "--------------------------" << std::endl;

				//// @TODO add check loop for the above rate expressions

				////----------------------------------------------------------
				////Reactions involving O2
				////----------------------------------------------------------
				////Adsorption and desorption rates
				double nO2 = mix.X()[iO2] * mix.numberDensity();
				double F_O2 = 0.25 * sqrt((8. * KB * T_surf[i]) / (PI * (mm(iO2) / NA)));

				double kOx1 = (F_O2 / (B * B)) * exp(-8000. / T_surf[i]); //kOx1
				double kOx4 = (F_O2 / (B * B)) * exp(-8000. / T_surf[i]); //kOx4
				// Eley-Rideal
				double kOx2 = (F_O2 / B) * 100. * exp(-4000. / T_surf[i]); //kOx2
				double kOx3 = (F_O2 / B) * exp(-500. / T_surf[i]); //kOx3
				double kOx5 = (F_O2 / B) * 1000. * exp(-4000. / T_surf[i]); //kOx5

				////----------------------------------------------------------
				//// Rates for O2 reactions in Minn FRC model
				////----------------------------------------------------------
				//rates(15) = kOx1 * nO2 * (ref_surf_cov_s(i) * B) * (ref_surf_cov_s(i) * B);
				//rates(16) = kOx2 * nO2 * (ref_surf_cov_O(i) * B);
				//rates(17) = kOx3 * nO2 * (ref_surf_cov_O(i) * B);
				//rates(18) = kOx4 * nO2 * (ref_surf_cov_s(i) * B) * (ref_surf_cov_s(i) * B);
				//rates(19) = kOx5 * nO2 * (ref_surf_cov_Ost(i) * B);

				////----------------------------------------------------------
				//// Rate comparison for O2 reactions in Minn FRC model
				////----------------------------------------------------------
				//std::cout << "--------------------------" << std::endl;
				//std::cout << "Rates for O2 reactions \n" << std::endl;
				//std::cout << rates(15) << " " << ratesmpp(15) << std::endl;
				//std::cout << rates(16) << " " << ratesmpp(16) << std::endl;
				//std::cout << rates(17) << " " << ratesmpp(17) << std::endl;
				//std::cout << rates(18) << " " << ratesmpp(18) << std::endl;
				//std::cout << rates(19) << " " << ratesmpp(19) << std::endl;
				//std::cout << "--------------------------" << std::endl;

				//// @TODO add check loop for the above rate expressions

				//// log the surface coverage 
				s_log[i] = v_surf_cov_mpp_frac[0];
				N_log[i] = v_surf_cov_mpp_frac[1];
				O_log[i] = v_surf_cov_mpp_frac[2];
				Ost_log[i] = v_surf_cov_mpp_frac[3];

				if (v_surf_cov_mpp_frac(0) / B >= 1.0e-14) {
					std::cout << " ---------- (s) ----------------" << std::endl;
					CHECK((v_surf_cov_frac(0) - v_surf_cov_mpp_frac(0)) / v_surf_cov_mpp_frac(0) == Catch::Detail::Approx(0.0).epsilon(tol));
					std::cout << (v_surf_cov_frac(0) - v_surf_cov_mpp_frac(0)) / v_surf_cov_mpp_frac(0) << std::endl;
					std::cout << " ---------- N-(s) ----------------" << std::endl;
					CHECK((v_surf_cov_frac(1) - v_surf_cov_mpp_frac(1)) / v_surf_cov_mpp_frac(1) == Catch::Detail::Approx(0.0).epsilon(tol));
					std::cout << (v_surf_cov_frac(1) - v_surf_cov_mpp_frac(1)) / v_surf_cov_mpp_frac(1) << std::endl;
					std::cout << " ---------- O-(s) ----------------" << std::endl;
					CHECK((v_surf_cov_frac(2) - v_surf_cov_mpp_frac(2)) / v_surf_cov_mpp_frac(2) == Catch::Detail::Approx(0.0).epsilon(tol));
					std::cout << (v_surf_cov_frac(2) - v_surf_cov_mpp_frac(2)) / v_surf_cov_mpp_frac(2) << std::endl;
					std::cout << " ---------- O*-(s) ----------------" << std::endl;
					CHECK((v_surf_cov_frac(3) - v_surf_cov_mpp_frac(3)) / v_surf_cov_mpp_frac(3) == Catch::Detail::Approx(0.0).epsilon(tol));
					std::cout << (v_surf_cov_frac(3) - v_surf_cov_mpp_frac(3)) / v_surf_cov_mpp_frac(3) << std::endl;
				}
				else {
					CHECK(v_surf_cov_frac(0) / B <= 1.0e-14);
				}

				std::cout << "--------------------------" << std::endl;
				std::cout << " P " << tol << " T " << T_surf[i] << std::endl;
				std::cout << " (s) " << v_surf_cov_mpp_frac[0]  << std::endl;
				std::cout << " N(s) " << v_surf_cov_mpp_frac[1]  << std::endl;
				std::cout << " O(s) " << v_surf_cov_mpp_frac[2]  << std::endl;
				std::cout << " O*(s) " << v_surf_cov_mpp_frac[3]  << std::endl;
				std::cout << " Sum " << v_surf_cov_mpp_frac.sum() << std::endl;
				std::cout << "--------------------------" << std::endl;

			}
		}

		std::cout << "--------------------------" << std::endl;
		std::cout << "T \n" << T_surf << std::endl;
		std::cout << "--------------------------" << std::endl;

		std::cout << "--------------------------" << std::endl;
		std::cout <<  "(s) Cov \n" << s_log << std::endl;
		std::cout << "--------------------------" << std::endl;

		std::cout << "--------------------------" << std::endl;
		std::cout << "[O-(s)] Cov \n" << O_log << std::endl;
		std::cout << "--------------------------" << std::endl;

		std::cout << "--------------------------" << std::endl;
		std::cout << "[O*-(s)] Cov \n" << Ost_log << std::endl;
		std::cout << "--------------------------" << std::endl;

		std::cout << "--------------------------" << std::endl;
		std::cout << "[N-(s)] Cov \n" << N_log << std::endl;
		std::cout << "--------------------------" << std::endl;

	}

}

	  // SECTION("ACA Model.")
	  // {
	   ////Set up M++
	//   MixtureOptions optsACA("Full_FRC_ACA_NASA9_ChemNonEq1T");
	   //Mixture mixACA(optsACA);

	   //const double B = 6.022e18;

	   //size_t ns = 9;
	   //size_t nss = 4;
	//   size_t nr = 20;
	   //size_t nsf = 5; // includes bulk carbon

	   //CHECK(mixACA.nSpecies() == ns);
	   //CHECK(mixACA.nSurfaceReactions() == nr);
	   //CHECK(mixACA.getSurfaceProperties().nSurfaceSpecies() == nsf);
	   //CHECK(mixACA.getSurfaceProperties().nSiteSpecies() == nss);

	//   const size_t iN = 0;
	//   const size_t iO = 1;
	   //const size_t iN2 = 3;
	   //const size_t iO2 = 4;
	   //const size_t iCO = 6;
	//   const size_t iCN = 7;
	   //const size_t iCO2 = 8;
	   //
	   //const int set_state_rhoi_T = 1;

	//   const double tol = 10e-4;

	   //ArrayXd v_rhoi(ns);
	   //ArrayXd num_den_i(ns);
	//   ArrayXd wdot(ns); ArrayXd wdotmpp(ns);
	//   ArrayXd rates(nr); ArrayXd ratesmpp(nr);
	   //ArrayXd kfmpp(nr); 
	//   wdot.setZero(); 
	   //wdotmpp.setZero();
	   //ArrayXd mm = mixACA.speciesMw(); 

	   ////std::cout << "T = " << T << " P = " << P << std::endl;

	   //CHECK(mixACA.getSurfaceProperties().isSurfaceCoverageSteady() == true);

	   //ArrayXd v_surf_cov_mpp_frac(mixACA.getSurfaceProperties().nSurfaceSpecies());
	//   ArrayXd v_surf_cov_frac(mixACA.getSurfaceProperties().nSurfaceSpecies());
	//
	   ////mixACA.getSurfaceProperties().forwardReactionCoefficients(ratesmpp(nr));	

	   //// Equilibrium Surface
	   //double P;
	   //double T;	
	   //
	//       //double P = 1.e-5;
	//       //double dP = 10.;
	//       //double T; // K
	//       //double dT = 1000.; // K

	//       for (int i = 0; i < 2; i++) {
	   //	
	   //	if (i == 0) {
	   //		P = 1600;
	   //		T = 1000;
	   //	}
	   //	if (i == 1) { 
	   //		P = 10000;
	   //		T = 2000;
	   //	}

	   //	double p_ev = 0.9;
	   //	double p_sv = 0.1;
	   //	
	   //	// EQ at loop T and P
	//               mixACA.equilibrate(T, P);
	//               mixACA.densities(v_rhoi.data());
	   //			
	   //			//ArrayXd pm = mixACA.phaseMoles();

	   //			//std::cout << "-----------------------------------" << std::endl;
	   //			//std::cout << " Multiphase Testing " << std::endl;
	   //			//std::cout << pm << std::endl;
	   //			//std::cout << "-----------------------------------" << std::endl;

	   //			

	   //	for (int q = 0; q < ns; q++) {
	   //		num_den_i[q] = mixACA.X()[q] * mixACA.numberDensity();	
	   //	}

	   //	//-------------------------------------------------------
	   //	//Atomic nitrogen reactions first, Prata claims the behavior is abstracted for atomic O
	   //	// and atomic N arriving at the surface
	   //	//-------------------------------------------------------

	   //	// Compute surface state based on EQ mass dens
	//               mixACA.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
	//               double nN = mixACA.X()[iN] * mixACA.numberDensity();
	   //	double n_conc = nN/NA;
	   //	
	   //	//std::cout << "----------------1---------------" << std::endl;
	   //	//std::cout << "MPP Rates" << ratesmpp << std::endl;

	//               mixACA.surfaceReactionRatesPerReaction(ratesmpp.data());
	//               mixACA.surfaceReactionRates(wdotmpp.data());
	   //	mixACA.forwardRateCoefficients(kfmpp.data());

	   //	//std::cout << "----------------2---------------" << std::endl;
	   //	//std::cout << "MPP Rates" << ratesmpp << std::endl;
	   //	
	   //	// Compute 'analytical' rate coefficients for atomic nitrogen reactions
	//               double F_N = 0.25 * sqrt(8 * KB * T / (PI * (mm(iN)/NA)));
	   //	
	   //	// Adsorption and desorption rates
	//               double kN1 = (F_N/B)*exp(-2500./T); //kN1
	//               double kN2 = (2 * PI * mm(iN) * KB * KB * T * T) / (NA* HP * HP * HP * B);
	   //	kN2 *= exp(-73971.6/T); // kN2 
	   //	
	   //	// Eley-Rideal rates (gas phase + surface phase)
	//               double kN4 = (F_N/B) * 0.5 * exp(-2000./T); // kN4
	//               double kN3 = (F_N/B) * 1.5 * exp(-7000./T);  // kN3

	   //	// Langmuir-Hishelwood rates (surface phase only)
	   //	double F2_N = sqrt( PI * KB * T / (2*(mm(iN)/NA)));
	   //	double kN5 = sqrt(NA/B) * F2_N * 0.1 * exp(-21000. / T); // kN5
	   //	double kN6 = 1e+8 * exp(-20676. / T); // kN6

	   //	// Compute consolidated rates as based on analytical expression in Prata et al
	   //	double A_3 = 0.;
	   //	double B_3 = kN1 * n_conc;
	   //	double C_3 = 2.*kN5;
	   //	double D_3 = kN2 + kN6 + (kN3 + kN4) * n_conc;
	   //	
	   //	//----------------------------------------------------------
	   //	//Reactions involving atomic O, low energy
	   //	//----------------------------------------------------------
	   //	
	   //	double nO = mixACA.X()[iO] * mixACA.numberDensity();
	   //	double o_conc = nO/NA;
	   //	double F_O = 0.25 * sqrt(8. * KB * T / (PI * (mm(iO)/NA)));		

	   //	//Adsorption and desorption rates
	   //	double kO1 = (F_O / B) * 0.3; //kO1
	   //	double kO2 = (2. * PI * (mm(iO)/NA) * KB * KB * T * T) / (NA* HP * HP * HP * B); 
	   //	kO2 *= exp(-44277. / T); //kO2

	   //	// Eley-Rideal
	   //	double kO3 = (F_O / B) * 100. *  exp(-4000. / T); //kO3
	   //	double kO4 = (F_O / B) *  exp(-500. / T); //kO4
	   //	
	   //	// Langmuir-Hinshelwood
	   //	double F2_O = sqrt( PI * KB * T / (2*(mm(iO)/NA)));
	   //	double kO9 = sqrt(NA / B) * F2_O * 5e-5 * exp(-15000. / T); //kO9
	   //	
	   //	//----------------------------------------------------------
	   //	//Reactions involving atomic O*, high energy
	   //	//----------------------------------------------------------
	   //	
	   //	//Adsorption and desorption rates
	   //	double kO5 = (F_O / B) * 0.7; //kO5
	   //	double kO6 =(2 * PI * (mm(iO)/NA) * KB * KB * T * T) / (NA* HP * HP * HP * B); 
	   //	kO6 *= exp(-96500. / T); //kO6

	   //	// Eley-Rideal
	   //	double kO7 = (F_O / B) * 1000. *  exp(-4000. / T); //kO7

	   //	// Langmuir-Hinshelwood
	   //	double kO8 = sqrt(NA / B) * F2_O * 1e-3 * exp(-15000. / T); //kO8
	   //	
	   //	

	   //	// Analytical expressions for the consolidated rates for O, O*, etc
	   //	double A_1 = 2. * kOx1 * o2_conc;
	   //	double B_1 = kO1 * o_conc;
	   //	double C_1 = 2. * kO9;
	   //	double D_1 = kO2 + (kO3 + kO4)*o_conc + (kOx2 + kOx3)*o2_conc;
	   //	double A_2 = 2. * kOx4 * o2_conc;
	   //	double B_2 = kO5 * o_conc;
	   //	double C_2 = 2. * kO8;
	   //	double D_2 = kO6 + kO7*o_conc + kOx5*o2_conc;

	   //	//---------------------------------------------------------
	   //	// Hard coded results for surface coverage from an external script for the Minn FRC model
	   //	//---------------------------------------------------------
	   //	
	   //	ArrayXd v_surf_cov_ext(nss);
	   //	ArrayXd v_surf_cov_ext_frac(nss);
	   //	
	   //	if (i == 0) // for P = 1600 Pa, T = 1000 K, units [#/m3]
	   //		v_surf_cov_ext(0) = 6.021976431367426e+18; // s
	   //		v_surf_cov_ext(1) = 3.281261193023990e+03; // N-s
	   //		v_surf_cov_ext(2) = 1.927352372805957e+13; // O-s
	   //		v_surf_cov_ext(3) = 4.295108842860487e+12; // O*-s

	   //	if (i == 1) // for P = 10,000 Pa, T = 2000 K, units [#/m3]
	   //		v_surf_cov_ext(0) = 6.021997116269985e+18; // s
	   //		v_surf_cov_ext(1) = 5.138700132518334e-10; // N-s
	   //		v_surf_cov_ext(2) = 2.356781643725696e+12; // O-s
	   //		v_surf_cov_ext(3) = 5.269483715967111e+11; // O*-s

	   //	for (int j = 0; j < nss; j++) {
	   //		v_surf_cov_ext_frac[j] = v_surf_cov_ext[j]/B;
	   //	}
	   //	
	   //	v_surf_cov_mpp_frac = mixACA.getSurfaceProperties().getSurfaceSiteCoverageFrac();
	   //	
	   //	std::cout << "-----------------------------------" << std::endl;
	   //	std::cout << "T " << T << " P " << P << std::endl;
	   //	std::cout << "-----------------------------------" << std::endl;
	   //	std::cout << "External Surf Frac " << std::endl;
	   //	std::cout << v_surf_cov_ext_frac << std::endl;
	   //	std::cout << "MPP Surf Frac " << std::endl;
	   //	std::cout << v_surf_cov_mpp_frac << std::endl;
	   //	


	   //	//const double thermal_speed = m_transport.speciesThermalSpeed(mv_react[idx_react]);
	   //	//m_surf_props(args.s_surf_props);
	   //	//m_site_categ = m_surf_props.siteSpeciesToSiteCategoryIndex(mv_react[idx_site]);
	//       	//m_n_sites = pow(m_surf_props.nSiteDensityInCategory(m_site_categ), m_stick_coef_power);

	   //	//std::cout << "MW " << mm[iN] << std::endl;
	   //	//std::cout << "RU " << RU << std::endl;

	   //	//mv_kf(i_r) = v_reactions[i_r]->getRateLaw()->forwardReactionRateCoefficient(mv_rhoi, m_surf_state.getSurfaceT());



      // }

   // }


