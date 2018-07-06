#ifndef GSI_RATE_LAW_H
#define GSI_RATE_LAW_H

#include <eigen3/Eigen/Dense>

namespace Mutation { namespace Thermodynamics { class Thermodynamics; }}
namespace Mutation { namespace Transport { class Transport; }}
namespace Mutation { namespace Utilities { namespace IO { class XmlElement; }}}

namespace Mutation {
    namespace GasSurfaceInteraction {

class SurfaceProperties;

//==============================================================================

/**
 * Structure for passing the data in the self registration
 * for GSIRateLaw class
 */
struct DataGSIRateLaw
{
    Mutation::Thermodynamics::Thermodynamics& s_thermo;
    const Mutation::Transport::Transport& s_transport;
    const Mutation::Utilities::IO::XmlElement& s_node_rate_law;
    const SurfaceProperties& s_surf_props;
    const std::vector<int>& s_reactants;
    const std::vector<int>& s_products;
};

//==============================================================================

/**
 * Abstract base class that computes the forward reaction rate coefficient
 * for all categories of surface interaction processes.
 */
class GSIRateLaw
{
public:
	/**
	 * Required for self registering different rate laws.
	 */
    typedef const DataGSIRateLaw& ARGS;

	/// Returns name of this type.
	static std::string typeName() { return "GSIRateLaw"; }

    /**
     * Constructor of the abstract base class, taking as input a structure
     * of type DataGSIRateLaw.
     */
    GSIRateLaw(ARGS args)
        : m_thermo(args.s_thermo),
          m_transport(args.s_transport)
    {}

    /**
     * Destructor
     */
    virtual ~GSIRateLaw(){}

    /**
     * Purely virtual function which returns the forward reaction rate
     * coefficients in kg/s. Its units depend on the specific process and
     * gas-surface interaction model.
     *
     * @param rhoi   species densities at the wall in kg/s
     * @param Twall  wall temperature at the wall according to the
     *               state model in K
     */
    virtual double forwardReactionRateCoefficient(
        const Eigen::VectorXd& v_rhoi,
        const Eigen::VectorXd& v_Twall) const = 0;

protected:
    Mutation::Thermodynamics::Thermodynamics& m_thermo;
    const Mutation::Transport::Transport& m_transport;

};

    } // namespace GasSurfaceInteraction
} // namespace Mutation

#endif // GSI_RATE_LAW_H