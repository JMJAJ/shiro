// xzip research stage (build define XZ_WRES, off = the tree as released): word-context residual correction.
//
// Ported from cmix-views 1.0 (kzwrx, GPL-3.0-only, src/word_residual.cpp; github.com/kzwrx/cmix-views), where it sits
// in the terminal chain behind the view router.  Mechanism, unchanged: on a codeword bit the word model predicts
// (word_label live), a cell keyed by (the last W completed words, the exact trie node of this bit in its codeword)
// keeps exact integer sums of the residual r = 65536 * bit - q and the Fisher weight v = q (65536 - q), plain and
// times the signed confidence s = 2q - 65536.  The correction is the two-coefficient Newton solve of that quadratic
// with a unit prior (beta0 + beta1 * s / 65536, added in the logit domain), and a per-word-context running log
// likelihood ratio of corrected against uncorrected sets how much of it is used (w = sigmoid(llr)).
// Adaptations here: the stage reads and writes the 32-bit probability of this tree's coder (behind the ONS stage when
// that is on), its statistics stay on the 16-bit q as in cmix-views; W is a build knob (cmix-views: 3); near-certain
// bits (q16 at 0 or 65535) and forced or gate-coded bits pass untouched; non-finite values are caught by bit pattern
// (the tree builds with -ffp-model=fast) instead of assert/abort.  Tables are capped as in cmix-views (2^20 cells,
// 2^18 word contexts; a new context or cell past the cap is simply not corrected).
#ifndef WRES_STAGE_H_
#define WRES_STAGE_H_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

#include "xz_build.h"

#ifndef XZ_WRES
#define XZ_WRES 0
#endif
#ifndef XZ_WRES_W        // completed words in the cell key (1..3)
#define XZ_WRES_W 3
#endif
#ifndef XZ_WRES_LR0      // starting gate log-odds per word context, in 1/100 (cmix-views 0 = w 0.5)
#define XZ_WRES_LR0 0
#endif
#ifndef XZ_WRES_CELLLOG  // cell cap 2^CELLLOG
#define XZ_WRES_CELLLOG 20
#endif

namespace wres_stage {

struct Key {
  uint64_t c;
  uint32_t n;
  bool operator==(const Key& b) const { return c == b.c && n == b.n; }
};
struct Hash {
  size_t operator()(const Key& k) const {
    uint64_t v = k.c ^ ((uint64_t)k.n * 0x9e3779b97f4a7c15ull);
    v ^= v >> 30;
    v *= 0xbf58476d1ce4e5b9ull;
    return (size_t)(v ^ (v >> 27));
  }
};
using I128 = __int128_t;
using U128 = __uint128_t;
struct Cell {
  int64_t r = 0;
  uint64_t v = 0;
  I128 r1 = 0, v1 = 0;
  U128 v2 = 0;
};

struct Stage {
  std::unordered_map<Key, Cell, Hash> cells;
  std::unordered_map<uint64_t, double> contexts;
  Key key{};
  bool pending = false;
  unsigned original = 0;
  int64_t signed_q = 0;
  double base = 0.5, expert = 0.5;

  static bool Bad(double v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return ((u >> 52) & 0x7ffu) == 0x7ffu;
  }
  static double Sig(double z) {
    if (z >= 0) return 1.0 / (1.0 + std::exp(-z));
    const double e = std::exp(z);
    return e / (1.0 + e);
  }

  // p32: the probability the coder would use; live / ctx / node: word_label's state of this bit; eligible: a mixer-path
  // bit (not forced, not coded by the adaptive-depth gate)
  __attribute__((noinline)) uint32_t P32(uint32_t p32, bool eligible, bool live, uint64_t ctx, uint32_t node) {
    pending = false;
    if (!eligible || !live) return p32;
    const unsigned q = p32 >> 16;
    if (q < 1u || q > 65534u) return p32;
    key = Key{ctx, node};
    auto it = cells.find(key);
    if (it == cells.end() &&
        !(cells.size() < ((size_t)1 << XZ_WRES_CELLLOG) && (contexts.count(ctx) || contexts.size() < (1u << 18))))
      return p32;
    original = q;
    signed_q = 2 * (int64_t)q - 65536;
    base = ((double)p32 + 0.5) * (1.0 / 4294967296.0);
    expert = base;
    if (it != cells.end() && (it->second.r || it->second.r1)) {
      const Cell& c = it->second;
      // exact integer sums until this fixed double solve (cmix-views' units)
      const double A = 1. + (double)c.v / 4294967296.;
      const double B = (double)c.v1 / 281474976710656.;
      const double C = 1. + (double)c.v2 / 18446744073709551616.;
      const double b0 = (double)c.r / 65536.;
      const double b1 = (double)c.r1 / 4294967296.;
      const double ratio = B / A, D = C - B * ratio;
      if (A > 0 && D > 0 && !Bad(D)) {
        const double beta1 = (b1 - ratio * b0) / D, beta0 = (b0 - B * beta1) / A;
        const double shift = beta0 + beta1 * ((double)signed_q / 65536.);
        if (!Bad(shift)) {
          const double z = std::log(base) - std::log1p(-base) + shift;
          double pe = Sig(z);
          if (!(pe >= 1.0 / 65536.)) pe = 1.0 / 65536.;
          if (!(pe <= 65535.0 / 65536.)) pe = 65535.0 / 65536.;
          if (!Bad(pe)) expert = pe;
        }
      }
    }
    auto st = contexts.find(ctx);
    const double lr = (st == contexts.end() ? 0.0 : st->second) + XZ_WRES_LR0 / 100.0;
    const double w = Sig(lr);
    double mixed = base + w * (expert - base);
    if (Bad(mixed) || !(mixed > 0.0 && mixed < 1.0)) return p32;
    pending = true;
    if (mixed == base) return p32;
    double r = std::floor(mixed * 4294967296.0);
    if (r < 4096.0) r = 4096.0;
    if (r > 4294967296.0 - 4096.0) r = 4294967296.0 - 4096.0;
    return (uint32_t)r;
  }

  __attribute__((noinline)) void Update(int bit) {
    if (!pending) return;
    pending = false;
    Cell& c = cells[key];
    double& lr = contexts[key.c];
    const double pb = bit ? base : 1 - base, pe = bit ? expert : 1 - expert;
    const double d = std::log(pe) - std::log(pb);
    if (!Bad(d)) lr += d;
    const int64_t r = 65536ll * bit - (int64_t)original;
    const uint64_t v = (uint64_t)original * (65536u - original);
    c.r += r;
    c.v += v;
    c.r1 += (I128)r * signed_q;
    c.v1 += (I128)v * signed_q;
    c.v2 += (U128)v * (U128)(signed_q * signed_q);
  }
};

}  // namespace wres_stage
#endif
