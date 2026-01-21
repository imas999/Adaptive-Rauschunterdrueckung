#include "LMS.hpp"

void blockLMS(float *sinus_LUT, float *inputBlock, float *outputBlock, float *filterCoeffs, int anzSamples, int M, float mu, float stoerFreq, float abtastFreq) {
    // Phasenänderung des Sinusreferenzsignals pro Sample (Abtasten eines Sinuses)
    static float momentanePhase = 0.0f;
    static float signalEnergie = 0.0f; // Gebraucht um die Signalenergie zu schätzen 
    
    const float phasenIncrement = ZWEI_PI * stoerFreq / abtastFreq;

    // Verarbeiten des Blocks von anzSamples Samples
    for (int i = 0; i < anzSamples; i++) {
        float inputSample = inputBlock[i];
        
        // Signalenergie-Schätzung (exponentiell gewichteter gleitender Mittelwert)
        // Mathematisch: E(n) = a * E(n-1) + (1 - a) * x(n)^2  | E(n) = signalEnergie, x(n) = inputSample
        signalEnergie = ENERGY_SMOOTHING * signalEnergie + ENERGY_SMOOTHING_INV * (inputSample * inputSample);
        
        // Adaptive Step-Size: reduziert bei schwachem Signal
        // Verhindert Instabilität und Entgleisen der Koeffizienten bei Stille
        float adaptiveMu = mu;
        bool istStille = (signalEnergie < SILENCE_THRESHOLD);
        if (istStille) {
            // Bei Stille: Step-Size stark reduzieren, um Entgleisen zu verhindern
            adaptiveMu = mu * 0.01f;
        }
        const float zweiMu = 2.0f * adaptiveMu;
        
        // Berechnung des Filteroutputs als gewichtete Summe der Sinuswerte
        float filterOutput = 0.0f;
        for (int k = 0; k < M; k++) {
            filterOutput += filterCoeffs[k] * getSineLUT(sinus_LUT, (momentanePhase - k * phasenIncrement));
        }

        // Berechnung des Fehlersignals (bereinigtes Signal)
        float fehlerSignal = inputSample - filterOutput;
        outputBlock[i] = fehlerSignal;

        // Koeffizienten-Update nach der LMS-Regel
        // Vorberechnung des Update-Faktors (außerhalb der Schleife)
        float updateFaktor = zweiMu * fehlerSignal;
        
        // Leakage NUR bei Stille (außerhalb der inneren Schleife entscheiden)
        if (!istStille) {
            for (int k = 0; k < M; k++) {
                filterCoeffs[k] += updateFaktor * getSineLUT(sinus_LUT, (momentanePhase - k * phasenIncrement));
                
                filterCoeffs[k] = clippingCheck(filterCoeffs[k]);
            }
        }

        // Aktualisierung der momentanen Phase für das nächste Sample
        // Phase größer 2Pi --> Zyklus Zurücksetzen
        momentanePhase += phasenIncrement;
        if (momentanePhase >= ZWEI_PI) {
            momentanePhase -= ZWEI_PI; 
        }
    }
}

float clippingCheck(float coeff) {
    if (coeff > MAX_COEFF_VALUE) {
        return MAX_COEFF_VALUE;
    } else if (coeff < -MAX_COEFF_VALUE) {
        return -MAX_COEFF_VALUE;
    }
    return coeff;
}

void quaterLUT(float *sinusLUT) {
    for (int i = 0; i < LUT_SIZE; i++) {
        sinusLUT[i] = sinf(i * M_PI / (2.0f * (LUT_SIZE - 1)));
    }
}

float getSineLUT(float *sinus_LUT, float phase) {
    while (phase < 0.0f) phase += ZWEI_PI;
    while (phase >= ZWEI_PI) phase -= ZWEI_PI;

    float lutPhase;
    int vorzeichen = 1;
    
    if (phase < HALF_PI) {
        // Quadrant 1: direkt aus LUT
        lutPhase = phase;
    } else if (phase < M_PI) {
        // Quadrant 2: gespiegelt an π/2
        lutPhase = M_PI - phase;
    } else if (phase < 1.5f * M_PI) {
        // Quadrant 3: wie Q1, aber negativ
        lutPhase = phase - M_PI;
        vorzeichen = -1;
    } else {
        // Quadrant 4: gespiegelt an 3π/2, negativ
        lutPhase = ZWEI_PI - phase;
        vorzeichen = -1;
    }

    int index = (int)(lutPhase * (LUT_SIZE - 1) / HALF_PI);

    if (index >= LUT_SIZE) index = LUT_SIZE - 1;

    return vorzeichen * sinus_LUT[index];
}