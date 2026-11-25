#ifndef DECODER_H
#define DECODER_H

#include <stdio.h>

#include "../predictor.h"
#include "../char_rate.h"
#include "../run_mode.h"
#include "../floor_stage.h"
#include "../ons_stage.h"
#include "../wres_stage.h"
#include "../wgru_stage.h"

class Decoder {
 public:
  Decoder(FILE* is, Predictor* p);
  int Decode();
  int DecodeRawBit(unsigned int p = 32768);
  inline void ObserveKnownBit(int bit) { p_->ObserveKnownBit(bit); }
  void ObserveKnownByte(unsigned int byte);
  inline void ObserveTfOnlyByte(unsigned int byte) {
    p_->ObserveTfOnlyByte(byte);
  }

 private:
  int ReadByte();
  unsigned int Discretize(float p);

  FILE* is_;
  unsigned int x1_, x2_, x_;
  Predictor* p_;
  floor_stage::Stage floor_;
#if XZ_ONS
  ons_stage::Stage ons_;   // xzip research stage (src/ons_stage.h)
#endif
#if XZ_WRES
  wres_stage::Stage wres_;   // xzip research stage (src/wres_stage.h, after cmix-views)
#endif
#if XZ_WGRU
  wgru_stage::Stage wgru_;   // xzip research stage (src/wgru_stage.h, after cmix-views)
#endif
};

#endif
