#ifndef LMS_HPP
#define LMS_HPP

#define _USE_MATH_DEFINES
#include <cmath>

// Konfigurationsparameter für Stabilität
#define SILENCE_THRESHOLD    1e-7f    // Schwellwert für Stille-Erkennung
#define LEAKAGE_FACTOR       0.999f  // Koeffizienten-Decay (je näher 1, desto kleiner der Einfluss, nur bemerkbar bei langer Stille) --> Verhindert Driften der Koeffizieten bei Stille / keinem Signal
#define MAX_COEFF_VALUE      10.0f    // Maximale Koeffizientengröße --> Verhindert Instabilität durch entgleisende Koeffizienten
#define ENERGY_SMOOTHING     0.99f    // Glättung für Energieschätzung
#define ENERGY_SMOOTHING_INV 0.01f    // (1 - ENERGY_SMOOTHING) vorberechnet
#define M_PI                3.14159265f

/**
 * @brief Adaptive Rauschunterdrückung mittels Block-LMS-Algorithmus für sinusförmige Störsignale
 * 
 * Funktionsweise:
 * - Für jedes Sample wird eine Schätzung der Störung durch gewichtete Summe von Sinuswerten berechnet
 * - Das bereinigtes Signal (Fehlersignal) ergibt sich aus der Differenz von Eingang und Schätzung
 * - Die Filterkoeffizienten werden nach der LMS-Regel adaptiv angepasst: w(n+1) = w(n) + 2*μ*e(n)*x(n)
 * 
 * @param inputBlock     Zeiger auf den Eingangspuffer mit gestörten Samples
 * @param outputBlock    Zeiger auf den Ausgangspuffer für bereinigte Samples (muss allokiert sein)
 * @param filterCoeffs   Zeiger auf Array der adaptiven Filterkoeffizienten (Größe: M)
 * @param anzSamples     Anzahl der zu verarbeitenden Samples im aktuellen Block
 * @param M              Filterordnung (Anzahl der Koeffizienten)
 * @param mu             Schrittweite für die Koeffizienten-Adaption
 *                       Kleinere Werte: stabiler aber langsamer
 *                       Größere Werte: schneller aber instabiler
 * @param stoerFreq      Frequenz der zu unterdrückenden Störung in Hz (50 Hz für Netzbrummen)
 * @param abtastFreq     Abtastfrequenz des Signals in Hz 
 * 
 * @note filterCoeffs sollte vor dem ersten Aufruf mit Nullen initialisiert werden
 */
void blockLMS(float *inputBlock, float *outputBlock, float *filterCoeffs, int anzSamples, int M, float mu, float stoerFreq, float abtastFreq);

/**
 * @brief Überprüft und begrenzt falls nötig den Koeffizientenwert um entgleisen zu verhindern
 * 
 * Funktionsweise:
 * - Wenn der Betrag des Koeffizienten den definierten Maximalwert überschreitet, wird er auf diesen Maximalwert begrenzt
 * - Dies verhindert Instabilität im adaptiven Filter durch zu große Koeffizientenwerte
 * 
 * @param coeff   Der zu überprüfende Koeffizientenwert
 * @return coeff  Der ggf. begrenzte Koeffizientenwert
 */
float clippingCheck(float coeff);


#endif