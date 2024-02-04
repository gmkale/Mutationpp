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

#include "mutation++.h"
#include "Configuration.h"
#include "TestMacros.h"
#include <catch2/catch.hpp>
#include <Eigen/Dense>

#include "SurfaceProperties.h"

using namespace Mutation;
using namespace Catch;
using namespace Eigen;
using namespace Catch::Matchers;

TEST_CASE("Detailed surface chemistry tests.","[gsi]")
{
    const double tol = 100. * std::numeric_limits<double>::epsilon();
    const double tol_det = 1.e2 * std::numeric_limits<double>::epsilon();

    Mutation::GlobalOptions::workingDirectory(TEST_DATA_FOLDER);

    SECTION("Surface Species and Coverage.")
    {
        // Setting up M++
        MixtureOptions opts("smb_detailed_coverage_NASA9_ChemNonEq1T");
        Mixture mix(opts);

        CHECK(mix.nSpecies() == 7);

        // Check global options
        CHECK(mix.nSurfaceReactions() == 0);
        CHECK(mix.getSurfaceProperties().nSurfaceSpecies() == 10);
        CHECK(mix.getSurfaceProperties().nSiteSpecies() == 9);

        // Check Species
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("N-s") == 8);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("O-s") == 9);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("N2-s") == 10);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("NO-c") == 12);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("C-b") == 16);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("O-c") == 13);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("C-p") == 15);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("A") == -1);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("s") == 7);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("c") == 11);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("p") == 14);
        CHECK(mix.getSurfaceProperties().surfaceSpeciesIndex("b") == -1);

        // Check surface species association with gaseous species
        CHECK( mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("N-s")) == 0);
        CHECK( mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O-s")) == 1);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("N2-s")) == 3);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("NO-c")) == 2);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("C-b")) == 5);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O-c")) == 1);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("C-p")) == 5);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("s")) == -2);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("c")) == -2);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("p")) == -2);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("b")) == -1);
        CHECK(mix.getSurfaceProperties().surfaceToGasIndex(100) == -1);

        // Check site species map correctly to the site category
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("N-s")) == 0);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O-c")) == 1);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("O-s")) == 0);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("N2-s")) == 0);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("NO-c")) == 1);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("C-b")) == -1);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("C-p")) == 2);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("s")) == 0);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("c")) == 1);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("p")) == 2);
        CHECK(mix.getSurfaceProperties().siteSpeciesToSiteCategoryIndex(
            mix.getSurfaceProperties().surfaceSpeciesIndex("b")) == -1);

        CHECK(mix.getSurfaceProperties().nSiteDensityInCategory(0) == 3.e19);
        CHECK(mix.getSurfaceProperties().nSiteDensityInCategory(1) == 7.e19);
        CHECK(mix.getSurfaceProperties().nSiteDensityInCategory(2) == 1.e20);
        CHECK(mix.getSurfaceProperties().nSiteDensityInCategory(3) == -1);

    }

/*    SECTION("Adsorption-Desorption Equilibrium.")
    {
        // Setting up M++
        MixtureOptions opts("smb_ads_des_eq_NASA9_ChemNonEq1T");
        Mixture mix(opts);

        size_t ns = 4;
        size_t nr = 2;
        CHECK(mix.nSpecies() == ns);
        CHECK(mix.nSurfaceReactions() == nr);

        const size_t iO = 0;
        const int set_state_rhoi_T = 1;

        ArrayXd v_rhoi(ns);
        ArrayXd wdot(ns); ArrayXd wdotmpp(ns);
        ArrayXd rates(nr); ArrayXd ratesmpp(nr);
        wdot.setZero(); wdotmpp.setZero();

        ArrayXd mm = mix.speciesMw();

        CHECK(mix.getSurfaceProperties().isSurfaceCoverageSteady() == false);
        ArrayXd v_surf_cov_frac(mix.getSurfaceProperties().nSurfaceSpecies());
        ArrayXd v_surf_cov_ss_frac(mix.getSurfaceProperties().nSurfaceSpecies());

        // Equilibrium Surface
        double P = 1000.;
        double T; // K
        double dT = 500.; // K
        for (int i = 25; i < 30; i++) { // 30
            T = (i+1) * dT; // += i*dT;

            mix.equilibrate(T, P);
            mix.densities(v_rhoi.data());

            mix.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
            double nO = mix.X()[iO] * mix.numberDensity();

            const double B = 1.e20;
            double kf1 = sqrt(RU * T / (2 * PI * mm(iO)));
            double kf2 = 2 * PI * mm(iO) / NA * KB * KB * T * T / (HP * HP * HP);
            kf2 *= exp(-100000./T);

            // Changing the surface coverage
            mix.getSurfaceProperties().setIsSurfaceCoverageSteady(false);
            v_surf_cov_frac(0) = 1.;
            const double a = 1.e-1;

            for (int j = 0; j < 0; j++){
                v_surf_cov_frac(0) *= a;                      // s
                v_surf_cov_frac(1) = 1. - v_surf_cov_frac(0); // Os

                mix.getSurfaceProperties().setSurfaceSiteCoverageFrac(v_surf_cov_frac);
                mix.surfaceReactionRatesPerReaction(ratesmpp.data());

                rates(0) = kf1 * nO * v_surf_cov_frac(0);
                rates(1) = kf2 * v_surf_cov_frac(1);
                CHECK(rates(0) == Catch::Detail::Approx(ratesmpp(0)).epsilon(tol));
                CHECK(rates(1) == Catch::Detail::Approx(ratesmpp(1)).epsilon(tol));

                wdot(iO) = mm(iO) / NA * (rates(0)-rates(1)); // From the surface point of view
                mix.surfaceReactionRates(wdotmpp.data());

                CHECK(wdot(0) == Catch::Detail::Approx(wdotmpp(0)).epsilon(tol));
                CHECK(wdot(1) == Catch::Detail::Approx(wdotmpp(1)).epsilon(tol));
                CHECK(wdot(2) == Catch::Detail::Approx(wdotmpp(2)).epsilon(tol));
                CHECK(wdot(3) == Catch::Detail::Approx(wdotmpp(3)).epsilon(tol));
            }

            // Equilibrium surface coverage
            mix.getSurfaceProperties().setIsSurfaceCoverageSteady(true);
            mix.surfaceReactionRatesPerReaction(ratesmpp.data());
            mix.surfaceReactionRates(wdotmpp.data());

            v_surf_cov_ss_frac = mix.getSurfaceProperties().getSurfaceSiteCoverageFrac();

            // THIS IS WRONG!
            v_surf_cov_frac.setZero();
            v_surf_cov_frac(0) = kf2 / (nO * kf1 + kf2);
            // double asdf = 1.e0 - v_surf_cov_ss_frac(0);
            v_surf_cov_frac(1) = 1.e0 - v_surf_cov_frac(0);

            // CHECK(v_surf_cov_ss_frac(0) == Catch::Detail::Approx(v_surf_cov_frac(0)).epsilon(tol));
            // CHECK(v_surf_cov_ss_frac(1) == Catch::Detail::Approx(v_surf_cov_frac(1)).epsilon(tol));
            std::cout << std::setprecision(100); // "i = " << i << std::endl;
            // std::cout << "    kf1 = " << kf1 << " kf2 = " << kf2 << std::endl;
            std::cout << "HERE 1 = " << v_surf_cov_frac(0) << " HERE 2 = " << v_surf_cov_frac(1) << std::endl;
            std::cout << "Sum = " << v_surf_cov_frac(0) + v_surf_cov_frac(1) << std::endl;
            std::cout << "MPP  1 = " << v_surf_cov_ss_frac(0) << " MPP  2 = " << v_surf_cov_ss_frac(1) << std::endl;
            std::cout << "Sum = " << v_surf_cov_ss_frac(0) + v_surf_cov_ss_frac(1) << std::endl;

            rates.setZero();
            rates(0) = kf1 * nO * v_surf_cov_frac(0);
            rates(1) = kf2 * v_surf_cov_frac(1);
            // CHECK(rates(0) == Catch::Detail::Approx(ratesmpp(0)).epsilon(tol_det));
            // CHECK(rates(1) == Catch::Detail::Approx(ratesmpp(1)).epsilon(tol_det));

            std::cout << "is zero here= " << rates(0) << " " << rates(1) << std::endl;
            std::cout << "is zero mpp = " << ratesmpp(0) << " " << ratesmpp(1) << std::endl;

            wdot(iO) = mm(iO) / NA * (rates(0)-rates(1)); // O
            //CHECK(wdot(0) == Catch::Detail::Approx(wdotmpp(0)).epsilon(tol));
            //CHECK(wdot(1) == Catch::Detail::Approx(wdotmpp(1)).epsilon(tol));
            //CHECK(wdot(2) == Catch::Detail::Approx(wdotmpp(2)).epsilon(tol));
            //CHECK(wdot(3) == Catch::Detail::Approx(wdotmpp(3)).epsilon(tol));

            std::cout << "HERE = " << wdot(0) << " MPP = " << wdotmpp(0) << std::endl;
            std::cout << "HERE = " << wdot(1) << " MPP = " << wdotmpp(1) << std::endl;
            // std::cout << "MPP = " << wdot(2) << " " << wdotmpp(2) << std::endl;
            // std::cout << "MPP = " << wdot(3) << " " << wdotmpp(3) << std::endl;
            double in; std::cin >> in;

        }

    } */

//    SECTION("PSMM Model.")
//    {
//        // Setting up M++
//        MixtureOptions optspsmm("smb_psmm_NASA9_ChemNonEq1T");
//        Mixture mixpsmm(optspsmm);
////        MixtureOptions optspark("smb_oxidation_NASA9_ChemNonEq1T");
////        Mixture mixpark(optspark);
//
//        const double tol = 10e-4;
//
//        size_t ns = 5;
//        size_t nr = 5;
//        CHECK(mixpsmm.nSpecies() == ns);
//        CHECK(mixpsmm.nSurfaceReactions() == nr);
//
//        const size_t iO = 0;
//        const size_t iCO = 3;
//        const size_t iCO2 = 4;
//
//        const int set_state_rhoi_T = 1;
//
//        ArrayXd v_rhoi(ns);
//        ArrayXd wdot(ns); ArrayXd wdotmpp(ns);
//        ArrayXd rates(nr); ArrayXd ratesmpp(nr);
//        wdot.setZero(); wdotmpp.setZero();
//
//        ArrayXd mm = mixpsmm.speciesMw();
//
//        CHECK(mixpsmm.getSurfaceProperties().isSurfaceCoverageSteady() == true);
//        ArrayXd v_surf_cov_mpp_frac(mixpsmm.getSurfaceProperties().nSurfaceSpecies());
//        ArrayXd v_surf_cov_frac(mixpsmm.getSurfaceProperties().nSurfaceSpecies());
//
//        // Equilibrium Surface
//        double P = 1.e-5;
//        double dP = 10.;
//        double T; // K
//        double dT = 200.; // K
//        for (int i = 0; i < 10; i++) {
//            for (int j = 0; j < 16; j++) {
//                T = (j+1) * dT;
//
//                mixpsmm.equilibrate(T, P);
//                mixpsmm.densities(v_rhoi.data());
//
//                mixpsmm.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
//                double nO = mixpsmm.X()[iO] * mixpsmm.numberDensity();
//
//                mixpsmm.surfaceReactionRatesPerReaction(ratesmpp.data());
//                mixpsmm.surfaceReactionRates(wdotmpp.data());
//
//                const double B = 1.e20;
//                double F = 1./B * sqrt(RU * T / (2 * PI * mm(iO)));
//                double kfads = F;
//                double kfdes = 2 * PI * mm(iO) / NA * KB * KB * T * T / (HP * HP * HP) / B;
//                kfdes *= exp(-44277./T);
//
//                double kfer1 = F * 5.737e+1 * exp(-4667./T);
//                double kfer2 = F * 8.529e-6 * exp( 6958./T);
//                double kfer3 = F * 1.203e-1 * exp( 2287./T);
//
//                v_surf_cov_frac(0) = (kfdes + kfer2*nO)/(kfads*nO + kfdes + kfer2*nO)*B;
//                v_surf_cov_frac(1) = B - v_surf_cov_frac(0);
//
//                v_surf_cov_mpp_frac = mixpsmm.getSurfaceProperties().getSurfaceSiteCoverageFrac();
//                v_surf_cov_mpp_frac *= B;
//                //std::cout << "Coverage MPP  = " << v_surf_cov_mpp_frac(0) << " " << v_surf_cov_mpp_frac(1) << std::endl;
//                //std::cout << "Coverage HERE  = " << v_surf_cov_frac(0) << " " << v_surf_cov_frac(1) << std::endl;
//
//                CHECK(v_surf_cov_mpp_frac(0) >= 0.0);
//                CHECK(v_surf_cov_mpp_frac(1) >= 0.0);
//                CHECK((B - v_surf_cov_mpp_frac.sum())/B == Catch::Detail::Approx(0.0).epsilon(tol));
//                if (v_surf_cov_mpp_frac(0)/B >= 1.0e-14)
//                    CHECK((v_surf_cov_frac(0) - v_surf_cov_mpp_frac(0))/v_surf_cov_mpp_frac(0) == Catch::Detail::Approx(0.0).epsilon(tol));
//                else
//                    CHECK(v_surf_cov_frac(0)/B <= 1.0e-14);
//
//                rates(0) = kfads * nO * v_surf_cov_frac(0);
//                rates(1) = kfdes * v_surf_cov_frac(1);
//                rates(2) = kfer1 * nO * v_surf_cov_frac(1);
//                rates(3) = kfer2 * nO *v_surf_cov_frac(1);
//                rates(4) = kfer3 * nO * v_surf_cov_frac(0);
//
//                //std::cout << "T = " << T << " P = " << P << std::endl;
//                //std::cout << "MPP  Rates = " << ratesmpp(0) << " "
//                //                             << ratesmpp(1) << " "
//                //                             << ratesmpp(2) << " "
//                //                             << ratesmpp(3) << " "
//                //                             << ratesmpp(4) << std::endl;
//                //std::cout << "HERE Rates = " << rates(0) <<  " "
//                //                             << rates(1) <<  " "
//                //                             << rates(2) <<  " "
//                //                             << rates(3) <<  " "
//                //                             << rates(4) << std::endl;
//
//                for (int ii = 0; ii < 5; ++ii ) {
//                    if (abs(ratesmpp(ii)) >= 1.0e-14)
//                        CHECK((rates(ii) - ratesmpp(ii))/ratesmpp(ii) == Catch::Detail::Approx(0.0).epsilon(tol));
//                    else
//                        CHECK(abs(rates(ii)) <= 1.0e-14);
//                }
//
//                wdot(0) = - mm(iO) / NA * (-rates(0) + rates(1) - rates(2) - rates(3) - rates(4));
//                wdot(1) = 0.;
//                wdot(2) = 0.;
//                wdot(3) = - mm(iCO) / NA * (+ rates(2) + rates(4));
//                wdot(4) = - mm(iCO2) / NA * (+ rates(3));
//
//                //std::cout << "T = " << T << " P = " << P << std::endl;
//                //std::cout << "MPP  Rates = " << wdotmpp(0) << " "
//                //                             << wdotmpp(1) << " "
//                //                             << wdotmpp(2) << " "
//                //                             << wdotmpp(3) << " "
//                //                             << wdotmpp(4) << std::endl;
//                //std::cout << "HERE Rates = " << wdot(0) <<  " "
//                //                             << wdot(1) <<  " "
//                //                             << wdot(2) <<  " "
//                //                             << wdot(3) <<  " "
//                //                             << wdot(4) << std::endl;
//
//                for (int ii = 0; ii < 4; ++ii ) {
//                    if (abs(wdotmpp(ii)) >= 1e-14)
//                        CHECK((wdot(ii) - wdotmpp(ii))/wdot(ii) == Catch::Detail::Approx(0.0).epsilon(tol));
//                     else
//                        CHECK(abs(wdot(ii)) <= 1.0e-14);
//                }
//
//                // Park
////                mixpark.equilibrate(T, P);
////               mixpark.densities(v_rhoi.data());
//
////               mixpark.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
////               nO = mixpark.X()[iO] * mixpark.numberDensity();
//
//                //std::cout << "End" << std::endl;
//                //double in; std::cin >> in;
//
//            }
//
//            P *= dP;
//        }
//
//    }

    //SECTION("Nitridation Model.")
    //{
    //    // Setting up M++
    //    MixtureOptions optsFRC("smb_FRC_nitridation_NASA9_ChemNonEq1T");
    //    Mixture mixFRC(optsFRC);

    //    size_t ns = 4;
    //    size_t nr = 4;

    //    CHECK(mixFRC.nSpecies() == ns);
    //    CHECK(mixFRC.nSurfaceReactions() == nr);

    //    const size_t iN = 0;
    //    const size_t iN2 = 1;
    //    const size_t iCN = 3;

    //    const int set_state_rhoi_T = 1;

    //    const double tol = 10e-4;

    //    ArrayXd v_rhoi(ns);
    //    ArrayXd wdot(ns); ArrayXd wdotmpp(ns);
    //    ArrayXd rates(nr); ArrayXd ratesmpp(nr);
    //    wdot.setZero(); wdotmpp.setZero();

    //    ArrayXd mm = mixFRC.speciesMw();

    //    CHECK(mixFRC.getSurfaceProperties().isSurfaceCoverageSteady() == true);
    //    ArrayXd v_surf_cov_mpp_frac(mixFRC.getSurfaceProperties().nSurfaceSpecies());
    //    ArrayXd v_surf_cov_frac(mixFRC.getSurfaceProperties().nSurfaceSpecies());

    //    // Equilibrium Surface
    //    double P = 1.e-5;
    //    double dP = 10.;
    //    double T; // K
    //    double dT = 200.; // K
    //    for (int i = 0; i < 15; i++) {
    //        for (int j = 0; j < 16; j++) {
    //            T = (j+1) * dT;

    //            mixFRC.equilibrate(T, P);
    //            mixFRC.densities(v_rhoi.data());

    //            mixFRC.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
    //            double nN = mixFRC.X()[iN] * mixFRC.numberDensity();

    //            mixFRC.surfaceReactionRatesPerReaction(ratesmpp.data());
    //            mixFRC.surfaceReactionRates(wdotmpp.data());

    //            const double B = 6.022e18;
    //            double F = 1./B * sqrt(RU * T / (2 * PI * mm(iN)));
    //            double kfads = F*exp(-7500./T);
    //            double kfdes = 2 * PI * mm(iN) / NA * KB * KB * T / (HP * HP * HP) / B;
    //            kfdes *= exp(-73971.6/T);

    //            double kfer1 = F * 9.0e+5 * exp(-20676./T);
    //            double kfer2 = F * 1.1e+6 * exp(-18000./T);

    //            v_surf_cov_frac(0) = (kfdes + nN*(kfer1+kfer2))/(nN*(kfads + kfer1 + kfer2) + kfdes)*B;
    //            v_surf_cov_frac(1) = B - v_surf_cov_frac(0);

    //            v_surf_cov_mpp_frac = mixFRC.getSurfaceProperties().getSurfaceSiteCoverageFrac();
    //            v_surf_cov_mpp_frac *= B;

    //            //Check total number of sites is respected and free spot is the same of analytical solution
    //            CHECK(v_surf_cov_mpp_frac(0) >= 0.0);
    //            CHECK(v_surf_cov_mpp_frac(1) >= 0.0);
    //            CHECK((B - v_surf_cov_mpp_frac.sum())/B == Catch::Detail::Approx(0.0).epsilon(tol));
    //            if (v_surf_cov_mpp_frac(0)/B >= 1.0e-14)
    //                CHECK((v_surf_cov_frac(0) - v_surf_cov_mpp_frac(0))/v_surf_cov_mpp_frac(0) == Catch::Detail::Approx(0.0).epsilon(tol));
    //            else
    //                CHECK(v_surf_cov_frac(0)/B <= 1.0e-14);

    //            //Check rates
    //            rates(0) = kfads * nN * v_surf_cov_frac(0);
    //            rates(1) = kfdes * v_surf_cov_frac(1);
    //            rates(2) = kfer1 * nN * v_surf_cov_frac(1);
    //            rates(3) = kfer2 * nN * v_surf_cov_frac(1);

    //            for (int ii = 0; ii < 4; ++ii ) {
    //                if (abs(ratesmpp(ii)) >= 1.0e-14)
    //                    CHECK((rates(ii) - ratesmpp(ii))/ratesmpp(ii) == Catch::Detail::Approx(0.0).epsilon(tol));
    //                else
    //                    CHECK(abs(rates(ii)) <= 1.0e-14);
    //            }

    //            //Check chemical production
    //            wdot(0) = - mm(iN) / NA * (-rates(0) + rates(1) - rates(3));
    //            wdot(1) = - mm(iN2) / NA * (rates(3));
    //            wdot(2) = 0.;
    //            wdot(3) = -mm(iCN) / NA * rates(2);

    //            for (int ii = 0; ii < 4; ++ii ) {
    //                if (abs(wdotmpp(ii)) >= 1e-14)
    //                    CHECK((wdot(ii) - wdotmpp(ii))/wdot(ii) == Catch::Detail::Approx(0.0).epsilon(tol));
    //                 else
    //                    CHECK(abs(wdot(ii)) <= 1.0e-14);
    //            }

    //        }

    //        P *= dP;
    //    }

    //}


    //SECTION("Nitridation Model 2.")
    //{
    //    // Setting up M++
    //    MixtureOptions optsFRC("smb_FRC2_nitridation_NASA9_ChemNonEq1T");
    //    Mixture mixFRC(optsFRC);

    //    size_t ns = 4;
    //    size_t nr = 6;

    //    CHECK(mixFRC.nSpecies() == ns);
    //    CHECK(mixFRC.nSurfaceReactions() == nr);

    //    const size_t iN = 0;
    //    const size_t iN2 = 1;
    //    const size_t iCN = 3;

    //    const int set_state_rhoi_T = 1;

    //    const double tol = 10e-4;

    //    ArrayXd v_rhoi(ns);
    //    ArrayXd wdot(ns); ArrayXd wdotmpp(ns);
    //    ArrayXd rates(nr); ArrayXd ratesmpp(nr);
    //    ArrayXd N_log(17); N_log.setZero();
    //    wdot.setZero(); wdotmpp.setZero();

    //    ArrayXd mm = mixFRC.speciesMw() / NA;

    //    CHECK(mixFRC.getSurfaceProperties().isSurfaceCoverageSteady() == true);
    //    ArrayXd v_surf_cov_mpp_frac(mixFRC.getSurfaceProperties().nSurfaceSpecies());
    //    ArrayXd v_surf_cov_frac(mixFRC.getSurfaceProperties().nSurfaceSpecies());

    //    // Equilibrium Surface

    //    double P = 1600;
    //    double dP = 10.;
    //    double T_start = 800; // K
    //    double dT = 100.; // K
    //    double T;

    //    for (int i = 0; i < 1; i++) {
    //        for (int j = 0; j < 17; j++) {
    //            T = (j * dT) + T_start;
    //            //std::cout << T <<  "   " << P << std::endl; 
    // 
    //            mixFRC.equilibrate(T, P);
    //            mixFRC.densities(v_rhoi.data());

    //            mixFRC.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
    //            double nN = mixFRC.X()[iN] * mixFRC.numberDensity();

    //            mixFRC.surfaceReactionRatesPerReaction(ratesmpp.data());
    //            mixFRC.surfaceReactionRates(wdotmpp.data());

    //            const double B = 6.022e18;
    //            double F = 0.25 * sqrt((8. * KB * T) / (PI * mm(iN)));

    //            double kN1= (F/B) * exp(-2500./T);
    //            double kN2 = ((2.0 * PI * mm(iN) * KB * KB * T * T) / (HP * HP * HP * B )) * exp(-73971.6 / T);

    //            double kN3 = (F/B) * 1.5 * exp(-7000./T);
    //            double kN4 = (F/B) * 0.5 * exp(-2000./T);

		  //      double kN5 = sqrt(NA/(B)) * sqrt((PI*KB*T)/ (2.0 * mm(iN)))* 0.1 * exp(-21000.0/ T);
		  //      double kN6 = 1.e8 * exp(-20676.0 / T);

    //            std::cout << " mN " << mm[iN] << std::endl;
		  //      std::cout << "kN1 is " << kN1 << std::endl;
    //            std::cout << "kN2 is " << kN2 << std::endl;
    //            std::cout << "kN3 is " << kN3 << std::endl;
    //            std::cout << "kN4 is " << kN4 << std::endl;
    //            std::cout << "kN5 is " << kN5 << std::endl;
    //            std::cout << "kN6 is " << kN6 << std::endl;

		  //      double A = 2.0*kN5;
		  //      double BB = (kN1 + kN3 + kN4)*nN + kN2 + kN6;
		  //      double C = kN1 * B * nN;

    //            v_surf_cov_frac(1) = (sqrt((BB*BB) + 4.*A*C) - BB) / (2.0 * A); //N-s
    //            v_surf_cov_frac(0) = B - v_surf_cov_frac(1); //s

    //            v_surf_cov_frac *= (1./B);

    //            //v_surf_cov_frac(0) = (kfdes + nN*(kfer1+kfer2))/(nN*(kfads + kfer1 + kfer2) + kfdes)*B;
    //            //v_surf_cov_frac(1) = B - v_surf_cov_frac(0);

    //            v_surf_cov_mpp_frac = mixFRC.getSurfaceProperties().getSurfaceSiteCoverageFrac();
    //            N_log[j] = v_surf_cov_mpp_frac[1];

    //            std::cout << "-------------------------" << std::endl;
    //            std::cout << "T " << T << " P " << P << std::endl;
    //            std::cout << "Analytical Coverage" << std::endl;
		  //      std::cout << v_surf_cov_frac(0) << "   " << v_surf_cov_frac(1)<< std::endl;
    //            //std::cout << "Test 1 " << std::endl;
    //            //std::cout << B - test1 << test1 << std::endl;
    //            std::cout << "Mpp Coverage" << std::endl;
		  //      std::cout << v_surf_cov_mpp_frac(0) << "   " << v_surf_cov_mpp_frac(1)<< std::endl;
    //            std::cout << "-------------------------" << std::endl;

    //            v_surf_cov_mpp_frac *= B;


    //            //std::cout << "-------------------------" << std::endl;
    //            //std::cout << "Mpp Coverage" << std::endl;
    //            //std::cout << v_surf_cov_mpp_frac << std::endl;
    //            //std::cout << "-------------------------" << std::endl;

    //            ////Check total number of sites is respected and free spot is the same of analytical solution
    //            //CHECK(v_surf_cov_mpp_frac(0) >= 0.0);
    //            //CHECK(v_surf_cov_mpp_frac(1) >= 0.0);
    //            //CHECK((B - v_surf_cov_mpp_frac.sum())/B == Catch::Detail::Approx(0.0).epsilon(tol));
    //            //if (v_surf_cov_mpp_frac(0)/B >= 1.0e-14)
    //            //    CHECK((v_surf_cov_frac(0) - v_surf_cov_mpp_frac(0))/v_surf_cov_mpp_frac(0) == Catch::Detail::Approx(0.0).epsilon(tol));
    //            //else
    //            //    CHECK(v_surf_cov_frac(0)/B <= 1.0e-14);

    //            //Check rates
    //            //rates(0) = kfads * nN * v_surf_cov_frac(0);
    //            //rates(1) = kfdes * v_surf_cov_frac(1);
    //            //rates(2) = kfer1 * nN * v_surf_cov_frac(1);
    //            //rates(3) = kfer2 * nN * v_surf_cov_frac(1);
    //            //rates(4) = kflh1 * v_surf_cov_frac(1)* v_surf_cov_frac(1);
    //            //rates(5) = kflh2 * v_surf_cov_frac(1);

    //    	    /*std::cout << "HERE" << std::endl;
		  //      std::cout << rates(0) << " " << ratesmpp(0) << std::endl;
		  //      std::cout << rates(1) << " " << ratesmpp(1) << std::endl;
		  //      std::cout << rates(2) << " " << ratesmpp(2) << std::endl;
		  //      std::cout << rates(3) << " " << ratesmpp(3) << std::endl;
		  //      std::cout << rates(4) << " " << ratesmpp(4) << std::endl;
		  //      std::cout << rates(5) << " " << ratesmpp(5) << std::endl;*/

    //            //for (int ii = 0; ii < 6; ++ii ) {
    //            //    if (abs(ratesmpp(ii)) >= 1.0e-14)
    //            //        CHECK((rates(ii) - ratesmpp(ii))/ratesmpp(ii) == Catch::Detail::Approx(0.0).epsilon(tol));
    //            //    else
    //            //        CHECK(abs(rates(ii)) <= 1.0e-14);
    //            //}

    //            ////Check chemical production
    //            //wdot(0) = - mm(iN) / NA * (-rates(0) + rates(1) - rates(3));
    //            //wdot(1) = - mm(iN2) / NA * (rates(3) + rates(4));
    //            //wdot(2) = 0.;
    //            //wdot(3) = -mm(iCN) / NA * (rates(2) + rates(5));

    //            //for (int ii = 0; ii < 4; ++ii ) {
    //            //    if (abs(wdotmpp(ii)) >= 1e-14)
    //            //        CHECK((wdot(ii) - wdotmpp(ii))/wdot(ii) == Catch::Detail::Approx(0.0).epsilon(tol));
    //            //     else
    //            //        CHECK(abs(wdot(ii)) <= 1.0e-14);
    //            //}

    //        }
    //        std::cout << "--------------------------" << std::endl;
    //        std::cout << "[N-(s)] Cov \n" << N_log << std::endl;
    //        std::cout << "--------------------------" << std::endl;


    //        P *= dP;
    //    }

    //}
    //SECTION("Call Solver.")
    //{
    //    // Setting up M++
    //    MixtureOptions opts("EQ_NASA9_ChemNonEq1T");
    //    Mixture mix(opts);

    //    double T = 2000.;
    //    double P = 101325.;
    //    mix.equilibrate(T, P);

    //    std::cout << mix.nGas() << " (gas) " << mix.nCondensed() << " (condensed)\n";

    //    for (int i = 0; i < mix.nSpecies(); ++i)
    //        std::cout << mix.X()[i] << '\n';

    //    //ArrayXd mole_fracs(mix.nSpecies());
    //    
    //    //mix.equilibrate(300.0, 101325., 2., 1.);
    //    //mole_fracs = mix.X();
    //    //std::cout << mix.X() << std::endl;
    //}

    SECTION("Nitridation Model 2.")
    {
        // Setting up M++
        MixtureOptions optsFRC("smb_FRC2_nitridation_NASA9_ChemNonEq1T");
        Mixture mixFRC(optsFRC);

        size_t ns = 4;
        size_t nr = 6;

        CHECK(mixFRC.nSpecies() == ns);
        CHECK(mixFRC.nSurfaceReactions() == nr);

        const size_t iN = 0;
        const size_t iN2 = 1;
        const size_t iCN = 3;

        const int set_state_rhoi_T = 1;

        const double tol = 10e-4;

        ArrayXd v_rhoi(ns);
        ArrayXd wdot(ns); ArrayXd wdotmpp(ns);
        ArrayXd rates(nr); ArrayXd ratesmpp(nr);
        wdot.setZero(); wdotmpp.setZero();

        ArrayXd mm = mixFRC.speciesMw();

        CHECK(mixFRC.getSurfaceProperties().isSurfaceCoverageSteady() == true);
        ArrayXd v_surf_cov_mpp_frac(mixFRC.getSurfaceProperties().nSurfaceSpecies());
        ArrayXd v_surf_cov_frac(mixFRC.getSurfaceProperties().nSurfaceSpecies());

        // Equilibrium Surface, problema a P = 1.e-5 e T = 1000.0 @TODO

        int count_press = 1;
        ArrayXd P(7);
        P[0] = 1600; P[1] = 10; P[2] = 100; P[3] = 1000; P[4] = 10000; P[5] = 50000; P[6] = 100000;
       
        //double P = 1.e-4;
        //double dP = 10.;
        //double P;
        double T; // K
        double dT = 200.; // K

        for (int i = 0; i < count_press; i++) { // i = 14 prior to mod

            //std::cout << P[i] << std::endl;

            for (int j = 0; j < 1; j++) { // j = 12 prior to mod
                T = (j + 6) * dT;
                
                //std::cout << tol <<  "   " << P << "  " << T <<  std::endl;
                //T = 2407.;
                //P = 1500.;

                //T = 1200.;
                //P = 1600.;

                mixFRC.equilibrate(T, P[i]);
                mixFRC.densities(v_rhoi.data());

                mixFRC.setSurfaceState(v_rhoi.data(), &T, set_state_rhoi_T);
                double nN = mixFRC.X()[iN] * mixFRC.numberDensity();

                mixFRC.surfaceReactionRatesPerReaction(ratesmpp.data());
                mixFRC.surfaceReactionRates(wdotmpp.data());

                const double B = 6.022e18;
                double F = 0.25 * sqrt(8. * KB * T / (PI * (mm(iN)/NA)));
                //std::cout << "F is " << F << std::endl;
                F *= (1.0 / B);
                //std::cout << "F is " << F << std::endl;

                double kfads = F * exp(-2500. / T);
                double kfdes = (2.0 * PI * (mm(iN) / NA) * KB * KB * T * T) / (HP * HP * HP * B);
                kfdes *= exp(-73971.6 / T);

                double kfer1 = F * 1.5 * exp(-7000. / T);
                double kfer2 = F * 0.5 * exp(-2000. / T);
                //double kflh1 = 4.96155568504e-12; //sqrt(1.0/(B)) * sqrt(PI*KB*T/ (8.0 * mm(iN)))* 0.1 * exp(-21000.0/ T);
                //double kflh1 = sqrt(NA/(B)) * sqrt(PI*KB*T/ (8.0 * mm(iN)))* 0.1 * exp(-21000.0/ T);
                double kflh1 = sqrt(1. / (B)) * sqrt(PI * KB * T / (2.0 * (mm(iN)/NA))) * 0.1 * exp(-21000.0 / T);
                double kflh2 = 1.e8 * exp(-20676.0 / T);

                //std::cout << "m_N is " << (mm(iN)/NA) << std::endl;
                //
                //std::cout << "kN1 is " << kfads << std::endl;
                //std::cout << "kN2 is " << kfdes << std::endl;
                //std::cout << "kN3 is " << kfer1 << std::endl;
                //std::cout << "kN4 is " << kfer2 << std::endl;
                //std::cout << "kN5 is " << kflh1 << std::endl;
                //std::cout << "kN6 is " << kflh2 << std::endl;

                double A = -2.0 * kflh1;
                double BB = (-kfads - kfer1 - kfer2) * nN - kfdes - kflh2;
                double C = kfads * B * nN;

                v_surf_cov_frac(1) = (-BB - sqrt(BB * BB - 4. * A * C)) / (2.0 * A); // N-(s)
                v_surf_cov_frac(0) = B - v_surf_cov_frac(1); // (s)

                //double Ap = 2.0 * kflh1;
                //double BBp = (kfads + kfer1 + kfer2) * nN + kfdes + kflh2;
                //double Cp = kfads * B * nN;

                //ArrayXd v_surf_cov_frac_p(mixFRC.getSurfaceProperties().nSurfaceSpecies());
                //v_surf_cov_frac_p(1) = (sqrt(BBp * BBp + 4. * Ap * Cp) - BBp) / (2.0 * Ap);
                //v_surf_cov_frac_p(0) = B - v_surf_cov_frac_p(1);

                //std::cout << "----------------------------------------------" << std::endl;
                //std::cout << " Michele " << std::endl;
                //std::cout << v_surf_cov_frac_p(0) / B << "   " << v_surf_cov_frac_p(1) / B << std::endl;
                //std::cout << " Greyson " << std::endl;
                //std::cout << v_surf_cov_frac_p(0) / B << "   " << v_surf_cov_frac_p(1) / B << std::endl;
                //std::cout << "----------------------------------------------" << std::endl;

                //v_surf_cov_frac(0) = (kfdes + nN*(kfer1+kfer2))/(nN*(kfads + kfer1 + kfer2) + kfdes)*B;
                //v_surf_cov_frac(1) = B - v_surf_cov_frac(0);

                v_surf_cov_mpp_frac = mixFRC.getSurfaceProperties().getSurfaceSiteCoverageFrac();
                v_surf_cov_mpp_frac *= B;

                //std::cout << "----------------------------------------------" << std::endl;
                //std::cout << "Analytical" << std::endl;
                //std::cout << "(s)" << "--------------" << "N-(s)" << std::endl;
                std::cout << v_surf_cov_frac(0)/B << "   " << v_surf_cov_frac(1) /B<< std::endl;
                //std::cout << "M++" << std::endl;
                //std::cout << "(s)" << "--------------" <<  "N-(s)" << std::endl;
                std::cout << v_surf_cov_mpp_frac(0)/B << "   " << v_surf_cov_mpp_frac(1)/B << std::endl;
                //std::cout << "----------------------------------------------" << std::endl;

                //Check total number of sites is respected and free spot is the same of analytical solution
                CHECK(v_surf_cov_mpp_frac(0) >= 0.0); // site solution is positive
                CHECK(v_surf_cov_mpp_frac(1) >= 0.0); // site solution is positive
                CHECK((B - v_surf_cov_mpp_frac.sum()) / B == Catch::Detail::Approx(0.0).epsilon(tol));

                if (v_surf_cov_mpp_frac(0) / B >= 1.0e-14) {
                    CHECK((v_surf_cov_frac(0) - v_surf_cov_mpp_frac(0)) / v_surf_cov_mpp_frac(0) == Catch::Detail::Approx(0.0).epsilon(tol));
                    std::cout << (v_surf_cov_frac(0) - v_surf_cov_mpp_frac(0)) / v_surf_cov_mpp_frac(0) << std::endl;
                }
                else {
                    CHECK(v_surf_cov_frac(0) / B <= 1.0e-14);
                }

                //Compute analytical rates
                rates(0) = kfads * nN * v_surf_cov_frac(0);
                rates(1) = kfdes * v_surf_cov_frac(1);
                rates(2) = kfer1 * nN * v_surf_cov_frac(1);
                rates(3) = kfer2 * nN * v_surf_cov_frac(1);
                rates(4) = kflh1 * v_surf_cov_frac(1) * v_surf_cov_frac(1);
                rates(5) = kflh2 * v_surf_cov_frac(1);

                // Check steady state converage
                double SS_check;
                SS_check = rates(0) - rates(1) - rates(2) - rates(3) - 2 * rates(4) - rates(5);
                        std::cout << "--------------------------" << std::endl;
                        std::cout << " P " << P[i] << " T " << T << std::endl;
                        std::cout << " Steady State \n" << SS_check << std::endl;
                        std::cout << "--------------------------" << std::endl;
            

                /*std::cout << "HERE" << std::endl;
            std::cout << rates(0) << " " << ratesmpp(0) << std::endl;
            std::cout << rates(1) << " " << ratesmpp(1) << std::endl;
            std::cout << rates(2) << " " << ratesmpp(2) << std::endl;
            std::cout << rates(3) << " " << ratesmpp(3) << std::endl;
            std::cout << rates(4) << " " << ratesmpp(4) << std::endl;
            std::cout << rates(5) << " " << ratesmpp(5) << std::endl;*/

                //for (int ii = 0; ii < 6; ++ii) {
                //    if (abs(ratesmpp(ii)) >= 1.0e-14)
                //        //CHECK((rates(ii) - ratesmpp(ii)) / ratesmpp(ii) == Approx(0.0).epsilon(tol));
                //        std::cout << (rates(ii) - ratesmpp(ii)) / ratesmpp(ii) << std::endl;
                //    else
                //        CHECK(abs(rates(ii)) <= 1.0e-14);
                //}

                //Check chemical production
                wdot(0) = -mm(iN) / NA * (-rates(0) + rates(1) - rates(3));
                wdot(1) = -mm(iN2) / NA * (rates(3) + rates(4));
                wdot(2) = 0.;
                wdot(3) = -mm(iCN) / NA * (rates(2) + rates(5));

                //for (int ii = 0; ii < 4; ++ii) {
                //    if (abs(wdotmpp(ii)) >= 1e-14)
                //        CHECK((wdot(ii) - wdotmpp(ii)) / wdot(ii) == Approx(0.0).epsilon(tol));
                //    else
                //        CHECK(abs(wdot(ii)) <= 1.0e-14);
                //}

            }

            //P *= dP;
        }
    }

}
