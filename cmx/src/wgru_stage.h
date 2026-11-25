// xzip research stage (build define XZ_WGRU, off = the tree as released): the coder-side consumer of the recurrent word
// history (src/word_gru.h).  Ported from cmix-views 1.0's cubic_overlay::Expert (kzwrx, GPL-3.0-only;
// github.com/kzwrx/cmix-views), the calibrator it puts behind word_order: one coefficient alpha on the cue u (the GRU
// logit minus the word model's logit), alpha = clip(-score / (1 + curvature), +-1) from the running Newton sums of the
// log loss, the shifted probability sigmoid(logit(p) + alpha u), and a running log-likelihood ratio of shifted
// against unshifted that sets the share used (w = sigmoid(llr)).  Adapted to this tree's 32-bit probability (its
// sums keep cmix-views' units), with bit-pattern checks instead of exit() for non-finite values (-ffp-model=fast).
#ifndef WGRU_STAGE_H_
#define WGRU_STAGE_H_

#include <cmath>
#include <cstdint>
#include <cstring>

#include "xz_build.h"
#include "wgru_feature.h"

#ifndef XZ_WGRU
#define XZ_WGRU 0
#endif

namespace wgru_stage {

struct Stage {
  double score = 0, curvature = 0, alpha = 0, phi = 0, probability = .5, log_odds = 0;
  double base = .5, peer = .5;
  bool pending = false;

  static bool Bad(double v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return ((u >> 52) & 0x7ffu) == 0x7ffu;
  }
  static double Sig(double z) {
    if (z >= 0) return 1. / (1. + std::exp(-z));
    const double e = std::exp(z);
    return e / (1. + e);
  }
  static uint32_t To32(double p) {
    double r = std::floor(p * 4294967296.0);
    if (r < 4096.0) r = 4096.0;
    if (r > 4294967296.0 - 4096.0) r = 4294967296.0 - 4096.0;
    return (uint32_t)r;
  }

  __attribute__((noinline)) uint32_t P32(uint32_t p32, bool eligible) {
    pending = false;
    if (!eligible || !word_gru::feature_live) return p32;
    const double u = word_gru::feature;
    if (Bad(u) || !(u >= -4.0 && u <= 4.0)) return p32;
    const unsigned q = p32 >> 16;
    if (q < 1u || q > 65534u) return p32;
    base = ((double)p32 + 0.5) * (1.0 / 4294967296.0);
    phi = u;
    alpha = -score / (1. + curvature);
    alpha = alpha < -1. ? -1. : (alpha > 1. ? 1. : alpha);
    probability = base;
    peer = base;
    if (alpha != 0 && phi != 0) {
      probability = Sig(std::log(base) - std::log1p(-base) + alpha * phi);
      if (Bad(probability) || !(probability > 0. && probability < 1.)) return p32;
      peer = probability;
    }
    pending = true;
    if (peer == base) return p32;
    const double mixed = base + Sig(log_odds) * (peer - base);
    if (Bad(mixed) || !(mixed > 0. && mixed < 1.)) return p32;
    return To32(mixed);
  }

  __attribute__((noinline)) void Update(int bit) {
    if (!pending) return;
    pending = false;
    const double v = probability * (1. - probability), g = probability - (double)bit, h = v * phi * phi;
    const double ns = score + g * phi - h * alpha, nc = curvature + h;
    if (!Bad(ns) && !Bad(nc) && nc >= 0) { score = ns; curvature = nc; }
    if (peer != base) {
      const double a = bit ? base : 1. - base, b = bit ? peer : 1. - peer;
      const double d = std::log(b / a);
      if (!Bad(d)) log_odds += d;
    }
  }
};

}  // namespace wgru_stage
#endif
