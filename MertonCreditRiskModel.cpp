#include <iostream>
#define _USE_MATH_DEFINES // Must be at the very top!
#include <cmath>
#include <iomanip>


// Standard Normal CDF: N(x)
double normal_cdf(double x) {
    // std::sqrt(0.5) is mathematically identical to 1 / sqrt(2)
    return 0.5 * std::erfc(-x * M_SQRT1_2);

}

struct FirmValuation {
    double equity;
    double debt_market;
    double distance_to_default; // d2
};

FirmValuation compute_merton(double V, double D, double r, double T, double sigma) {
    double d1 = (std::log(V /D) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    double N_d1 = normal_cdf(d1);
    double N_d2 = normal_cdf(d2);

    double equity = V * N_d1 - D * std::exp(-r * T) * N_d2;
    double debt_market = V - equity; // Merton Identity: V = E + D_market

    return {equity, debt_market, d2};
}

//  ^ that's the end of compute_merton

int main() {
    // STEP 4: Call Site - This is where execution RESUMES after return

    double V = 100, D = 80, r = 0.05, T = 1.0, sigma = 0.30;

    // Call: Jump INTO compute_merton, then come back here with the struct
    FirmValuation result = compute_merton(V, D, r, T, sigma);

    // STEP 5: Unpack the struct you returned
    std::cout << "Equity: " << result.equity << "\n";           // 26.46
    std::cout << "Debt Mkt: " << result.debt_market << "\n";    // 73.54
    std::cout << "d2 (Distance to Default): " << result.distance_to_default << "\n"; // 0.76
    std::cout << "Prob Default = N(-d2): " << normal_cdf(-result.distance_to_default)*100 << "%\n";

    // C++17 modern professor version:
    // auto [equity, debt_mkt, d2] = compute_merton(V,D,r,T,sigma);

    return 0;
}