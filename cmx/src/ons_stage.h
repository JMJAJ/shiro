// xzip research stage (build define XZ_ONS, off = the tree as released): a terminal second-order logistic combiner.
//
// Every learner of the cmix lineage (mixers, SSE, APMs, the BITLSTM's online parts) is a first-order method: a step
// along the gradient with a hand-tuned rate schedule.  This stage is an Online Newton Step (Hazan, Agarwal & Kale
// 2007) on the log loss, which is exp-concave, so ONS has O(d log T) regret against the best fixed weight vector,
// with no learning-rate schedule.  It keeps A = lambda * A + g g^T (a running Gauss-Newton estimate of the loss
// curvature) and steps w -= eta * A^-1 g.  A^-1 is kept directly with Sherman-Morrison (O(d^2) per bit, d = 6), and
// the forgetting factor lambda < 1 keeps the step from freezing, so the stage tracks a non-stationary stream (an RLS
// filter in the logistic domain).
//
// Input: the probability the coder would use (behind the BITLSTM, the floor stage and the semantic-cache end stages),
// plus four logits from upstream of it: the predictor before and after the BITLSTM, the end-mixer logit and the
// adaptive-depth gate logit, plus a bias.  It runs one weight set per (bit position, adaptive-depth skip).  Weights
// start at (1, 0, ...), so a fresh set codes exactly as without this stage.  Bits with |logit| >= kZCap pass through
// untouched and do not train: the floor stage codes those with 32-bit counts, which a float stage would only blur.
// Stability (batch 9: with lambda 1 - 2^-16 the first version coded 30 MB to 24 MB): the inputs are nearly collinear,
// so A is close to singular and forgetting inflates A^-1 along the flat directions.  Three guards: A^-1 is capped at
// its start eps^-1 I by trace (a ridge), each weight moves at most DMAX per bit, and every set keeps a decayed running
// difference of its own code length against its input's; while ONS is behind, the set codes its input unchanged (and
// keeps learning), so the stage cannot fall far behind the stage it follows.
// The tree builds with -ffp-model=fast (finite math assumed: isnan / isfinite fold away), so non-finite values are
// caught by their bit pattern: a non-finite or wild input is zeroed, a bad output codes the input, and a set whose
// Sherman-Morrison denominator, inverse or weights go bad (A^-1 drifting indefinite) restarts from its initial state.
// A^-1 is symmetrized after every update to hold it positive definite.
// Encoder and decoder each own a Stage and feed it the same values; the methods are out-of-line (noinline), so both
// run the same machine code and the decoder repeats every float operation exactly.
#ifndef ONS_STAGE_H_
#define ONS_STAGE_H_

#include <cmath>
#include <cstdint>
#include <cstring>

#include "xz_build.h"

#ifndef XZ_ONS
#define XZ_ONS 0
#endif
#ifndef XZ_ONS_ETA      // step eta in per mille (1000 = the plain Newton step; 2000 diverged on synthetic data)
#define XZ_ONS_ETA 1000
#endif
#ifndef XZ_ONS_LAMLOG   // forgetting lambda = 1 - 2^-LAMLOG
#define XZ_ONS_LAMLOG 16
#endif
#ifndef XZ_ONS_EPS      // A starts as eps * I, eps in per mille; A^-1 never grows past its start (trace cap, a ridge)
#define XZ_ONS_EPS 16000
#endif
#ifndef XZ_ONS_DMAX     // per-coordinate weight step clamp, in 1/65536 (0 = off)
#define XZ_ONS_DMAX 1024
#endif
#ifndef XZ_ONS_FEAT     // inputs: 0 the six below; 1 + p before the semantic-cache end stages; 2 + the 22 layer-0
#define XZ_ONS_FEAT 0     // mixer outputs of the BITLSTM's stage-1 row; 3 + that whole row (path and cache columns)
#endif
#ifndef XZ_ONS_GUARD    // 0 off; else the guard's loss memory is 2^GUARD bits (per set)
#define XZ_ONS_GUARD 12
#endif

namespace ons_stage {

constexpr int kD = XZ_ONS_FEAT == 0 ? 6 : (XZ_ONS_FEAT == 1 ? 7 : (XZ_ONS_FEAT == 2 ? 29 : 48));
constexpr int kSets = 16;   // bit position x skip
constexpr double kZCap = 7.5;
constexpr double kWCap = 4.0;

struct Set {
  double w[kD];
  double ai[kD][kD];   // A^-1
  double dl;           // decayed (ONS code length - input code length), bits
};

struct Stage {
  Set s[kSets];
  double x[kD];
  double pout = 0.5, pin = 0.5;
  int cur = -1;

  __attribute__((noinline, cold)) Stage() {
    for (int k = 0; k < kSets; ++k) Reset(s[k]);
  }

  static double St16(unsigned int p16) {
    if (p16 < 1u) p16 = 1u;
    if (p16 > 65535u) p16 = 65535u;
    return std::log((double)p16 / (65536.0 - (double)p16));
  }
  // non-finite by bit pattern (exponent all ones), immune to -ffinite-math-only
  static bool Bad(double v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return ((u >> 52) & 0x7ffu) == 0x7ffu;
  }
  static double Clip(double v, double c) { return Bad(v) ? 0.0 : (v < -c ? -c : (v > c ? c : v)); }
  __attribute__((noinline, cold)) static void Reset(Set& e) {
    e.dl = 0.0;
    for (int i = 0; i < kD; ++i) {
      e.w[i] = i == 0 ? 1.0 : 0.0;
      for (int j = 0; j < kD; ++j) e.ai[i][j] = i == j ? 1000.0 / XZ_ONS_EPS : 0.0;
    }
  }

  // p32: what the coder would use; p_pre / p_head: 16-bit p before / behind the BITLSTM; m1raw: end-mixer logit;
  // z: adaptive-depth gate logit; skip: the bit took the gate's sigma(z); bp: bit position; ovr: a forced bit
  // p_floor: the floor stage's output (before the semantic-cache end stages); s1 / n1: the BITLSTM's stage-1 row
  __attribute__((noinline)) uint32_t P32(uint32_t p32, unsigned int p_pre, unsigned int p_head, float m1raw, float z,
                                         int skip, int bp, int ovr, uint32_t p_floor, const float* s1, int n1) {
    cur = -1;
    if (ovr) return p32;
    const double q = ((double)p32 + 0.5) * (1.0 / 4294967296.0);
    const double x0 = std::log(q / (1.0 - q));
    if (!(x0 > -kZCap && x0 < kZCap)) return p32;
    cur = (bp & 7) * 2 + (skip ? 1 : 0);
    x[0] = x0;
    x[1] = St16(p_pre);
    x[2] = St16(p_head);
    x[3] = Clip((double)m1raw, 8.0);
    x[4] = Clip((double)z, 8.0);
    x[5] = 1.0;
    if (kD > 6) {
      const double qf = ((double)p_floor + 0.5) * (1.0 / 4294967296.0);
      x[6] = Clip(std::log(qf / (1.0 - qf)), 12.0);
      for (int i = 7; i < kD; ++i) x[i] = (s1 && i - 7 < n1) ? Clip((double)s1[i - 7], 8.0) : 0.0;
    }
    const Set& e = s[cur];
    double t = 0.0;
    for (int i = 0; i < kD; ++i) t += e.w[i] * x[i];
    pin = q;
    if (Bad(t)) { pout = q; cur = -1; return p32; }
    t = Clip(t, 12.0);
    pout = 1.0 / (1.0 + std::exp(-t));
    if (XZ_ONS_GUARD && e.dl > 0.0) return p32;   // behind its input: code the input
    double r = std::floor(pout * 4294967296.0);
    if (r < 4096.0) r = 4096.0;
    if (r > 4294967296.0 - 4096.0) r = 4294967296.0 - 4096.0;
    return (uint32_t)r;
  }

  __attribute__((noinline)) void Update(int bit) {
    if (cur < 0) return;
    Set& e = s[cur];
    const double err = pout - (double)bit;   // d(-log p(bit)) / dt
    if (XZ_ONS_GUARD) {
      const double lo = -std::log2(bit ? pout : 1.0 - pout), li = -std::log2(bit ? pin : 1.0 - pin);
      e.dl += (lo - li) - e.dl * std::ldexp(1.0, -XZ_ONS_GUARD);
    }
    double g[kD], ag[kD];
    for (int i = 0; i < kD; ++i) g[i] = err * x[i];
    double den = 0.0;
    for (int i = 0; i < kD; ++i) {
      double a = 0.0;
      for (int j = 0; j < kD; ++j) a += e.ai[i][j] * g[j];
      ag[i] = a;
      den += g[i] * a;
    }
    // A <- lambda A + g g^T  =>  A^-1 <- (A^-1 - ag ag^T / (lambda + g^T A^-1 g)) / lambda; the 1/lambda inflation
    // is held back while trace(A^-1) is above its start, so a quiet set does not blow up its step
    const double lam = 1.0 - std::ldexp(1.0, -XZ_ONS_LAMLOG);
    if (Bad(den) || !(den >= 0.0) || den > 1e12) { Reset(e); return; }   // A^-1 no longer positive definite
    const double inv = 1.0 / (lam + den);
    double tr = 0.0;
    for (int i = 0; i < kD; ++i) {
      for (int j = 0; j < kD; ++j) e.ai[i][j] -= ag[i] * ag[j] * inv;
      tr += e.ai[i][i];
    }
    for (int i = 0; i < kD; ++i)
      for (int j = i + 1; j < kD; ++j) e.ai[i][j] = e.ai[j][i] = 0.5 * (e.ai[i][j] + e.ai[j][i]);
    if (Bad(tr) || !(tr > 0.0)) { Reset(e); return; }
    if (tr < kD * 1000.0 / XZ_ONS_EPS)
      for (int i = 0; i < kD; ++i)
        for (int j = 0; j < kD; ++j) e.ai[i][j] *= 1.0 / lam;
    // Newton step with the updated inverse (A_t^-1 g_t, as ONS)
    const double eta = XZ_ONS_ETA / 1000.0;
    for (int i = 0; i < kD; ++i) {
      double a = 0.0;
      for (int j = 0; j < kD; ++j) a += e.ai[i][j] * g[j];
      double d = eta * a;
      if (XZ_ONS_DMAX) d = Clip(d, XZ_ONS_DMAX / 65536.0);
      e.w[i] = Clip(e.w[i] - d, kWCap);
    }
    if (Bad(e.dl)) e.dl = 0.0;
  }
};

}  // namespace ons_stage
#endif
