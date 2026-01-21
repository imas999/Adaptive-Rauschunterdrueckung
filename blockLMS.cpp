//blockLMS.cpp
#include "LMS.hpp"
#include "base_types.h"

void blockLMS(float *sinus_LUT, float *inputBlock, float *outputBlock, float *filterCoeffs, 
              int anzSamples, int M, float mu, float stoerFreq, float abtastFreq) {
    
    static float momentanePhase = 0.0f;
    const float phasenIncrement = ZWEI_PI * stoerFreq / abtastFreq;

    // Verarbeiten des Blocks von anzSamples Samples
    for (int i = 0; i < anzSamples; i++) {
        float inputSample = inputBlock[i];
        
        // Berechnung des Filteroutputs (Schätzung der Störung)
        float filterOutput = 0.0f;
        for (int k = 0; k < M; k++) {
            float pastReference = getSineLUT(sinus_LUT, momentanePhase - k * phasenIncrement);
            filterOutput += filterCoeffs[k] * pastReference;
        }
        
        // Fehlersignal berechnen (gewünschtes Signal minus geschätzte Störung)
        float fehlerSignal = inputSample - filterOutput;
        
        // Fehlersignal als Output speichern
        outputBlock[i] = fehlerSignal;
        
        // Koeffizienten-Update nach der LMS-Regel
        for (int k = 0; k < M; k++) {
            float pastReference = getSineLUT(sinus_LUT, momentanePhase - k * phasenIncrement);
            filterCoeffs[k] += 2.0f * mu * fehlerSignal * pastReference;
        }
        
        // Phase für nächstes Sample aktualisieren
        momentanePhase += phasenIncrement;
        if (momentanePhase > ZWEI_PI) {
            momentanePhase -= ZWEI_PI;  // Wrap Phase in [0, 2π]
        }
    }
}

// clippingCheck wird nicht mehr benötigt, kann gelöscht werden

void quaterLUT(float *sinusLUT) {
    for (int i = 0; i < LUT_SIZE; i++) {
        sinusLUT[i] = sinf(i * M_PI / (2.0f * (LUT_SIZE - 1)));
    }
}

float getSineLUT(float *sinus_LUT, float phase) {
    // Phase normalisieren
    while (phase < 0.0f) phase += ZWEI_PI;
    while (phase >= ZWEI_PI) phase -= ZWEI_PI;
    
    float lutPhase;
    int vorzeichen = 1;
    
    if (phase < HALF_PI) {
        lutPhase = phase;
    } else if (phase < M_PI) {
        lutPhase = M_PI - phase;
    } else if (phase < 1.5f * M_PI) {
        lutPhase = phase - M_PI;
        vorzeichen = -1;
    } else {
        lutPhase = ZWEI_PI - phase;
        vorzeichen = -1;
    }
    
    int index = (int)(lutPhase * (LUT_SIZE - 1) / HALF_PI);
    if (index >= LUT_SIZE) index = LUT_SIZE - 1;
    
    return vorzeichen * sinus_LUT[index];
}