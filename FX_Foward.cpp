// pfe_fx_forward.cpp
// C++17: Monte Carlo EE(t) and PFE(t, alpha) for an FX forward under GBM.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// --------- Utilities ---------

// Quantile (0<alpha<1) of a vector (non-const because we sort).
double percentile(std::vector<double>& xs, double alpha) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    // "linear interpolation between closest ranks" can be used;
    // here we do a simple nearest-rank with floor to be conservative for PFE.
    const double pos = alpha * (xs.size() - 1);
    const size_t idx = static_cast<size_t>(std::floor(pos));
    const double frac = pos - idx;
    if (idx + 1 < xs.size())
        return xs[idx] * (1.0 - frac) + xs[idx + 1] * frac;
    return xs.back();
}

// Discount factor assuming flat domestic rate r_d.
inline double discount(double r_d, double t) {
    return std::exp(-r_d * t);
}

// --------- Model & Pricer ---------

// Evolve FX under GBM: dS = S * ( (r_d - r_f) dt + sigma dW )
void simulate_paths_gbm(
    std::vector<std::vector<double>>& S, // [nPaths][nSteps+1]
    double S0, double r_d, double r_f, double sigma,
    double T, int nSteps, std::mt19937_64& rng)
{
    const double dt = T / nSteps;
    std::normal_distribution<double> Z(0.0, 1.0);

    for (size_t p = 0; p < S.size(); ++p) {
        S[p][0] = S0;
        for (int j = 1; j <= nSteps; ++j) {
            const double z = Z(rng);
            const double drift = (r_d - r_f - 0.5 * sigma * sigma) * dt;
            const double diff  = sigma * std::sqrt(dt) * z;
            S[p][j] = S[p][j - 1] * std::exp(drift + diff);
        }
    }
}

// Simple FX forward MtM from bank's perspective at time t:
// V(t) = N * ( S(t) - K ) * DF_d(t)
// (Domestic-discounted payoff; sign assumes receiving S, paying K at T.)
// If you want precise forward maturing at T, you can scale by DF(T)/DF(t)
// and/or set value only at maturity; here we keep a running MtM proxy.
inline double forward_mtm(double notional, double S_t, double K, double r_d, double t) {
    return notional * (S_t - K) * discount(r_d, t);
}

// Exposure is positive part of MtM.
inline double exposure(double Vt) { return std::max(Vt, 0.0); }

// --------- Main EE/PFE Engine ---------

struct Results {
    std::vector<double> times;     // size nSteps+1
    std::vector<double> EE;        // Expected Exposure at each time
    std::vector<double> PFE;       // Potential Future Exposure at each time
};

Results compute_EE_PFE_FXForward(
    int nPaths, int nSteps, double T,
    double S0, double r_d, double r_f, double sigma,
    double notional, double strikeK,
    double alpha, uint64_t seed = 42ULL)
{
    // 1) Simulate FX paths
    std::mt19937_64 rng(seed);
    std::vector<std::vector<double>> S(nPaths, std::vector<double>(nSteps + 1));
    simulate_paths_gbm(S, S0, r_d, r_f, sigma, T, nSteps, rng);

    // 2) Time grid
    std::vector<double> times(nSteps + 1);
    for (int j = 0; j <= nSteps; ++j) times[j] = (T * j) / nSteps;

    // 3) For each time, compute exposures across paths, then EE and PFE
    std::vector<double> EE(nSteps + 1, 0.0);
    std::vector<double> PFE(nSteps + 1, 0.0);
    std::vector<double> bucket(nPaths);

    for (int j = 0; j <= nSteps; ++j) {
        const double t = times[j];

        // Build exposure samples at time t across paths
        for (int p = 0; p < nPaths; ++p) {
            const double Vt = forward_mtm(notional, S[p][j], strikeK, r_d, t);
            bucket[p] = exposure(Vt);
        }

        // EE(t) = mean of positive exposures
        const double sum = std::accumulate(bucket.begin(), bucket.end(), 0.0);
        EE[j] = sum / static_cast<double>(nPaths);

        // PFE(t, alpha) = alpha-quantile of exposures
        // (we make a working copy because percentile sorts in-place)
        std::vector<double> tmp = bucket;
        PFE[j] = percentile(tmp, alpha);
    }

    return { std::move(times), std::move(EE), std::move(PFE) };
}

// --------- Demo / CLI ---------

int main(int argc, char** argv) {
    // Default parameters (override via argv if desired).
    int    nPaths  = 20000;
    int    nSteps  = 20;         // e.g., quarterly over 5 years => set T=5.0 and nSteps=20
    double T       = 2.0;        // years
    double S0      = 1.10;       // spot FX (e.g., USD per EUR)
    double r_d     = 0.035;      // domestic rate
    double r_f     = 0.015;      // foreign rate
    double sigma   = 0.12;       // FX vol
    double N       = 10'000'000; // notional
    double K       = 1.12;       // forward strike
    double alpha   = 0.95;       // PFE quantile
    uint64_t seed  = 42ULL;

    // (Optional) basic CLI parsing for quick tweaks
    if (argc > 1) nPaths = std::stoi(argv[1]);
    if (argc > 2) nSteps = std::stoi(argv[2]);
    if (argc > 3) T      = std::stod(argv[3]);

    auto res = compute_EE_PFE_FXForward(
        nPaths, nSteps, T, S0, r_d, r_f, sigma, N, K, alpha, seed
    );

    // Pretty print
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "t,EE,PFE\n";
    for (size_t j = 0; j < res.times.size(); ++j) {
        std::cout << res.times[j] << "," << res.EE[j] << "," << res.PFE[j] << "\n";
    }

    // A quick sanity summary at T/2 and T
    auto halfway = res.times.size() / 2;
    std::cout << "\nSummary\n";
    std::cout << "EE(T/2)  = " << res.EE[halfway] << "\n";
    std::cout << "PFE(T/2) = " << res.PFE[halfway] << "\n";
    std::cout << "EE(T)    = " << res.EE.back() << "\n";
    std::cout << "PFE(T)   = " << res.PFE.back() << "\n";

    return 0;
}
