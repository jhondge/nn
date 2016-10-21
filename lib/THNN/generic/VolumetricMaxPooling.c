#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxPooling.c"
#else

void THNN_(VolumetricMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          intptr_t kT,
          intptr_t kW,
          intptr_t kH,
          intptr_t dT,
          intptr_t dW,
          intptr_t dH,
          intptr_t pT,
          intptr_t pW,
          intptr_t pH,
          intptr_t ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateOutput)(
          state, input, output, indices,
          kT, kW, kH, dT, dW, dH,
          pT, pW, pH, 1, 1, 1, ceilMode);
}

void THNN_(VolumetricMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          intptr_t dT,
          intptr_t dW,
          intptr_t dH,
          intptr_t pT,
          intptr_t pW,
          intptr_t pH)
{
  THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          state, input, gradOutput, gradInput, indices,
          dT, dW, dH, pT, pW, pH, 1, 1, 1);
}

#endif
