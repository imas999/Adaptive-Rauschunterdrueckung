#include "LMS.hpp"

void blockLMS(float *inputBlock, float *outputBlock, float *filterCoeffs, int anzSamples, int M, float mu, float stoerFreq, float abtastFreq) {
    // Phasenänderung des Sinusreferenzsignals pro Sample (Abtasten eines Sinuses)
    static const float zweiPI = 2.0f * M_PI;  
    static float momentanePhase = 0.0f;
    static float signalEnergie = 0.0f; // Gebraucht um die Signalenergie zu schätzen 
    
    const float phasenIncrement = zweiPI * stoerFreq / abtastFreq;

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
            float sinReferenz = sinf(momentanePhase - k * phasenIncrement);
            filterOutput += filterCoeffs[k] * sinReferenz;
        }

        // Berechnung des Fehlersignals (bereinigtes Signal)
        float fehlerSignal = inputSample - filterOutput;
        outputBlock[i] = fehlerSignal;

        // Koeffizienten-Update nach der LMS-Regel
        // Vorberechnung des Update-Faktors (außerhalb der Schleife)
        float updateFaktor = zweiMu * fehlerSignal;
        
        // Leakage NUR bei Stille (außerhalb der inneren Schleife entscheiden)
        if (istStille) {
            for (int k = 0; k < M; k++) {
                float sinReferenz = sinf(momentanePhase - k * phasenIncrement);
                filterCoeffs[k] = LEAKAGE_FACTOR * filterCoeffs[k] + updateFaktor * sinReferenz;
                
                filterCoeffs[k] = clippingCheck(filterCoeffs[k]);
            }
        } else {
            for (int k = 0; k < M; k++) {
                float sinReferenz = sinf(momentanePhase - k * phasenIncrement);
                filterCoeffs[k] += updateFaktor * sinReferenz;
                
                filterCoeffs[k] = clippingCheck(filterCoeffs[k]);
            }
        }

        // Aktualisierung der momentanen Phase für das nächste Sample
        // Phase größer 2Pi --> Zyklus Zurücksetzen
        momentanePhase += phasenIncrement;
        if (momentanePhase >= 2.0f * M_PI) {
            momentanePhase -= 2.0f * M_PI; 
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