#include <cmath>
#include <iostream>
#include <iomanip>

static double normal_cdf(double x)
{
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

struct BlackScholesCall
{
    double strike; 
    double rate;
    double volatility;
    double time;
    double maturity;

    double operator()(double spot) const
    {
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

template<typename Price>
double delta(Price&& price, double spot, double h)
{
    return (price(spot + h) - price(spot - h)) / (2.0 * h);
}

BlackScholesCall call{100.0, 0.01, 0.20, 1.0, 1.0};

double spot = 100.0;
double bump = 0.5;

int main()
{
    double price = call(spot);
    double d = delta(call, spot, bump);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Price: " << price << "\n";
    std::cout << "Delta: " << d << "\n";
    return 0;
}
