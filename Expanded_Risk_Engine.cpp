#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

static double normal_cdf(double x)
{
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

struct BlackScholesCall
{
    double strike; 
    double rate;
    double volatility;

    // We remove the static maturity variable from the struct configuration 
    // so we can pass it dynamically through our evaluation matrix execution loop.
    double operator()(double spot, double maturity) const
    {
        if (maturity <= 0.0) {
            // At expiration, an option's value is simply its intrinsic payout
            return std::max(0.0, spot - strike);
        }

        const double sqrtT = std::sqrt(maturity);
        const double d1 =
            (std::log(spot / strike)
            + (rate + 0.5 * volatility * volatility) * maturity)
            / (volatility * sqrtT);

        const double d2 = d1 - volatility * sqrtT;

        return spot * normal_cdf(d1)
                - strike * std::exp(-rate * maturity) * normal_cdf(d2);
    }
};

// Numerical Delta: First-order derivative via Central Finite Difference
template<typename PriceModel>
double calculate_delta(const PriceModel& model, double spot, double maturity, double h)
{
    return (model(spot + h, maturity) - model(spot - h, maturity)) / (2.0 * h);
}

// Numerical Gamma: Second-order derivative tracking Delta acceleration
template<typename PriceModel>
double calculate_gamma(const PriceModel& model, double spot, double maturity, double h)
{
    return (model(spot + h, maturity) - 2.0 * model(spot, maturity) + model(spot - h, maturity)) / (h * h);
}

int main()
{
    // Configure standard contract parameters (Strike: $100, 20% Vol, 1% Risk-Free Rate)
    BlackScholesCall call{100.0, 0.01, 0.20};
    
    double bump = 0.1; // Finite difference step size (h)

    // Define our matrix boundary vectors
    std::vector<double> spot_prices = {90.0, 95.0, 100.0, 105.0, 110.0};
    std::vector<double> maturities  = {1.0, 0.5, 0.25, 0.083}; // 1 Year, 6 Months, 3 Months, 1 Month

    // Print Header Layout
    std::cout << std::left << std::setw(12) << "Spot ($)" 
              << std::setw(14) << "Maturity (Y)" 
              << std::setw(14) << "Call Price" 
              << std::setw(12) << "Delta (Δ)" 
              << std::setw(12) << "Gamma (Γ)" << "\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";

    std::cout << std::fixed << std::setprecision(5);

    // Execute the Multi-Asset Surface Grid Sweep
    for (double spot : spot_prices)
    {
        for (double tte : maturities)
        {
            double price = call(spot, tte);
            double delta = calculate_delta(call, spot, tte, bump);
            double gamma = calculate_gamma(call, spot, tte, bump);

            std::cout << std::left << std::setw(12) << spot 
                      << std::setw(14) << tte 
                      << std::setw(14) << price 
                      << std::setw(12) << delta 
                      << std::setw(12) << gamma << "\n";
        }
        std::cout << "─────────────────────────────────────────────────────────────\n";
    }

    return 0;
}