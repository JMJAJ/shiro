// xzip research (XZ_WGRU): the per-bit feature of the recurrent word history (src/word_gru.h, inside word_label.cpp)
// handed to the coder's terminal stage (src/wgru_stage.h).  Defined in word_label.cpp.
#pragma once
namespace word_gru {
extern double feature;
extern bool feature_live;
inline void ClearFeature() { feature_live = false; feature = 0.; }
}
