"""
Test-Skript für den Block-LMS Algorithmus
Testet verschiedene Werte von M und mu zur Unterdrückung einer 50Hz Störung

Verwendet test.wav (Sprachsignal mit 50Hz Netzbrummen, 8kHz Abtastrate)

Autor: DSP Projekt Gruppe 4
Datum: 17.01.2026

Zielplattform: FM4-176-S6E2CC-ETH DSP-Board
"""

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from typing import Tuple
import os

# =============================================================================
# KONFIGURATIONSPARAMETER (exakt wie in LMS.hpp)
# =============================================================================
SILENCE_THRESHOLD = 1e-7      # Schwellwert für Stille-Erkennung
LEAKAGE_FACTOR = 0.999        # Koeffizienten-Decay bei Stille
MAX_COEFF_VALUE = 10.0        # Maximale Koeffizientengröße
ENERGY_SMOOTHING = 0.99       # Glättung für Energieschätzung
ENERGY_SMOOTHING_INV = 0.01   # (1 - ENERGY_SMOOTHING) vorberechnet


# =============================================================================
# OPERATIONS-ZÄHLER für DSP-Board Analyse
# =============================================================================
class OperationsZaehler:
    """Zählt alle relevanten Operationen für DSP-Performance-Analyse."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.additionen = 0
        self.multiplikationen = 0
        self.divisionen = 0
        self.sin_aufrufe = 0
        self.vergleiche = 0
        self.zuweisungen = 0
        self.block_aufrufe = 0
        self.sample_iterationen = 0
        self.koeff_iterationen = 0
    
    def zusammenfassung(self, M: int, block_groesse: int, num_blocks: int) -> dict:
        """Berechnet Zusammenfassung der Operationen."""
        total_samples = num_blocks * block_groesse
        return {
            'block_aufrufe': self.block_aufrufe,
            'total_samples': total_samples,
            'additionen': self.additionen,
            'multiplikationen': self.multiplikationen,
            'divisionen': self.divisionen,
            'sin_aufrufe': self.sin_aufrufe,
            'vergleiche': self.vergleiche,
            'zuweisungen': self.zuweisungen,
            'koeff_iterationen': self.koeff_iterationen,
            # Pro Sample Statistiken
            'add_pro_sample': self.additionen / total_samples if total_samples > 0 else 0,
            'mult_pro_sample': self.multiplikationen / total_samples if total_samples > 0 else 0,
            'sin_pro_sample': self.sin_aufrufe / total_samples if total_samples > 0 else 0,
            # Pro Block Statistiken
            'add_pro_block': self.additionen / self.block_aufrufe if self.block_aufrufe > 0 else 0,
            'mult_pro_block': self.multiplikationen / self.block_aufrufe if self.block_aufrufe > 0 else 0,
            'sin_pro_block': self.sin_aufrufe / self.block_aufrufe if self.block_aufrufe > 0 else 0,
        }


# Globaler Zähler
ops = OperationsZaehler()


# =============================================================================
# HILFSFUNKTIONEN (exakt wie in blockLMS.cpp)
# =============================================================================
def clipping_check(coeff: float) -> float:
    """
    Überprüft und begrenzt falls nötig den Koeffizientenwert.
    Exakt wie in blockLMS.cpp
    """
    ops.vergleiche += 2  # Zwei Vergleiche im worst case
    if coeff > MAX_COEFF_VALUE:
        return MAX_COEFF_VALUE
    elif coeff < -MAX_COEFF_VALUE:
        return -MAX_COEFF_VALUE
    return coeff


# =============================================================================
# BLOCK-LMS ALGORITHMUS (1:1 Übersetzung von blockLMS.cpp)
# =============================================================================
def block_lms(input_block: np.ndarray, filter_coeffs: np.ndarray, 
              M: int, mu: float, stoer_freq: float, abtast_freq: float,
              state: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block-LMS Algorithmus zur adaptiven Störunterdrückung.
    
    EXAKTE 1:1 Übersetzung von blockLMS.cpp!
    
    C++ Original:
    void blockLMS(float *inputBlock, float *outputBlock, float *filterCoeffs, 
                  int anzSamples, int M, float mu, float stoerFreq, float abtastFreq)
    
    Unterschied zu C++: 
    - static Variablen werden über state-Dictionary zwischen Aufrufen erhalten
    """
    ops.block_aufrufe += 1
    
    # Konstanten (wie in C++)
    zwei_pi = 2.0 * np.pi
    ops.multiplikationen += 1
    
    # phasenIncrement = zweiPI * stoerFreq / abtastFreq (C++ Zeile 9)
    phasen_increment = zwei_pi * stoer_freq / abtast_freq
    ops.multiplikationen += 1
    ops.divisionen += 1
    
    anz_samples = len(input_block)
    output_block = np.zeros(anz_samples, dtype=np.float32)
    
    # static Variablen aus C++ (über state-Dictionary)
    momentane_phase = state['momentane_phase']
    signal_energie = state['signal_energie']
    
    # Hauptschleife: for (int i = 0; i < anzSamples; i++) (C++ Zeile 12)
    for i in range(anz_samples):
        ops.sample_iterationen += 1
        
        input_sample = input_block[i]
        ops.zuweisungen += 1
        
        # Signalenergie-Schätzung (C++ Zeile 17)
        # signalEnergie = ENERGY_SMOOTHING * signalEnergie + ENERGY_SMOOTHING_INV * (inputSample * inputSample)
        signal_energie = ENERGY_SMOOTHING * signal_energie + ENERGY_SMOOTHING_INV * (input_sample * input_sample)
        ops.multiplikationen += 3  # 3 Multiplikationen
        ops.additionen += 1
        
        # Adaptive Step-Size (C++ Zeile 21-27)
        adaptive_mu = mu
        ops.zuweisungen += 1
        
        # bool istStille = (signalEnergie < SILENCE_THRESHOLD)
        ist_stille = signal_energie < SILENCE_THRESHOLD
        ops.vergleiche += 1
        
        if ist_stille:
            # adaptiveMu = mu * 0.01f (C++ Zeile 25)
            adaptive_mu = mu * 0.01
            ops.multiplikationen += 1
        
        # const float zweiMu = 2.0f * adaptiveMu (C++ Zeile 27)
        zwei_mu = 2.0 * adaptive_mu
        ops.multiplikationen += 1
        
        # Berechnung des Filteroutputs (C++ Zeile 30-34)
        # float filterOutput = 0.0f;
        # for (int k = 0; k < M; k++) {
        #     float sinReferenz = sinf(momentanePhase - k * phasenIncrement);
        #     filterOutput += filterCoeffs[k] * sinReferenz;
        # }
        filter_output = 0.0
        for k in range(M):
            ops.koeff_iterationen += 1
            # sinf(momentanePhase - k * phasenIncrement)
            sin_referenz = np.sin(momentane_phase - k * phasen_increment)
            ops.multiplikationen += 1  # k * phasenIncrement
            ops.additionen += 1        # momentanePhase - ...
            ops.sin_aufrufe += 1
            
            # filterOutput += filterCoeffs[k] * sinReferenz
            filter_output += filter_coeffs[k] * sin_referenz
            ops.multiplikationen += 1
            ops.additionen += 1
        
        # Fehlersignal (C++ Zeile 37-38)
        # float fehlerSignal = inputSample - filterOutput;
        # outputBlock[i] = fehlerSignal;
        fehler_signal = input_sample - filter_output
        ops.additionen += 1
        output_block[i] = fehler_signal
        ops.zuweisungen += 1
        
        # Update-Faktor (C++ Zeile 42)
        # float updateFaktor = zweiMu * fehlerSignal;
        update_faktor = zwei_mu * fehler_signal
        ops.multiplikationen += 1
        
        # Koeffizienten-Update (C++ Zeile 45-59)
        if ist_stille:
            # Mit Leakage (C++ Zeile 46-51)
            for k in range(M):
                ops.koeff_iterationen += 1
                # float sinReferenz = sinf(momentanePhase - k * phasenIncrement);
                sin_referenz = np.sin(momentane_phase - k * phasen_increment)
                ops.multiplikationen += 1
                ops.additionen += 1
                ops.sin_aufrufe += 1
                
                # filterCoeffs[k] = LEAKAGE_FACTOR * filterCoeffs[k] + updateFaktor * sinReferenz;
                filter_coeffs[k] = LEAKAGE_FACTOR * filter_coeffs[k] + update_faktor * sin_referenz
                ops.multiplikationen += 2
                ops.additionen += 1
                ops.zuweisungen += 1
                
                # filterCoeffs[k] = clippingCheck(filterCoeffs[k]);
                filter_coeffs[k] = clipping_check(filter_coeffs[k])
        else:
            # Ohne Leakage (C++ Zeile 53-58)
            for k in range(M):
                ops.koeff_iterationen += 1
                # float sinReferenz = sinf(momentanePhase - k * phasenIncrement);
                sin_referenz = np.sin(momentane_phase - k * phasen_increment)
                ops.multiplikationen += 1
                ops.additionen += 1
                ops.sin_aufrufe += 1
                
                # filterCoeffs[k] += updateFaktor * sinReferenz;
                filter_coeffs[k] += update_faktor * sin_referenz
                ops.multiplikationen += 1
                ops.additionen += 1
                ops.zuweisungen += 1
                
                # filterCoeffs[k] = clippingCheck(filterCoeffs[k]);
                filter_coeffs[k] = clipping_check(filter_coeffs[k])
        
        # Phase aktualisieren (C++ Zeile 62-65)
        # momentanePhase += phasenIncrement;
        momentane_phase += phasen_increment
        ops.additionen += 1
        
        # if (momentanePhase >= 2.0f * M_PI) { momentanePhase -= 2.0f * M_PI; }
        ops.vergleiche += 1
        if momentane_phase >= zwei_pi:
            momentane_phase -= zwei_pi
            ops.additionen += 1
    
    # Zustand für nächsten Block speichern (entspricht static in C++)
    state['momentane_phase'] = momentane_phase
    state['signal_energie'] = signal_energie
    
    return output_block, filter_coeffs


# =============================================================================
# ANALYSE-FUNKTIONEN
# =============================================================================
def berechne_stoerunterdrueckung_db(input_signal: np.ndarray, output_signal: np.ndarray,
                                     stoer_freq: float, abtast_freq: float,
                                     bandbreite: float = 5.0) -> float:
    """Berechnet die Störunterdrückung in dB durch Vergleich der Leistung bei 50Hz."""
    n = len(input_signal)
    
    freq_input = np.fft.fft(input_signal)
    freq_output = np.fft.fft(output_signal)
    freqs = np.fft.fftfreq(n, 1/abtast_freq)
    
    idx_stoer = np.where((np.abs(freqs - stoer_freq) < bandbreite) | 
                         (np.abs(freqs + stoer_freq) < bandbreite))[0]
    
    if len(idx_stoer) == 0:
        return 0.0
    
    power_input = np.mean(np.abs(freq_input[idx_stoer])**2)
    power_output = np.mean(np.abs(freq_output[idx_stoer])**2)
    
    if power_output == 0:
        return float('inf')
    
    return 10 * np.log10(power_input / power_output)


def teste_parameter_kombination(signal: np.ndarray, M: int, mu: float,
                                 block_groesse: int, abtast_freq: float,
                                 stoer_freq: float) -> dict:
    """Testet eine bestimmte M/mu-Kombination und zählt Operationen."""
    
    # Operations-Zähler zurücksetzen
    ops.reset()
    
    # Filter initialisieren
    filter_coeffs = np.zeros(M, dtype=np.float32)
    state = {'momentane_phase': 0.0, 'signal_energie': 0.0}
    
    # Signal blockweise verarbeiten
    num_blocks = len(signal) // block_groesse
    output_signal = np.zeros_like(signal)
    
    for block_idx in range(num_blocks):
        start = block_idx * block_groesse
        end = start + block_groesse
        
        input_block = signal[start:end]
        output_block, filter_coeffs = block_lms(
            input_block, filter_coeffs, M, mu, stoer_freq, abtast_freq, state
        )
        output_signal[start:end] = output_block
    
    # Restliche Samples verarbeiten
    rest_start = num_blocks * block_groesse
    if rest_start < len(signal):
        input_block = signal[rest_start:]
        output_block, filter_coeffs = block_lms(
            input_block, filter_coeffs, M, mu, stoer_freq, abtast_freq, state
        )
        output_signal[rest_start:] = output_block
    
    # Störunterdrückung berechnen (erste Hälfte überspringen für Einschwingphase)
    start_analyse = len(signal) // 4
    unterdrueckung_db = berechne_stoerunterdrueckung_db(
        signal[start_analyse:], output_signal[start_analyse:],
        stoer_freq, abtast_freq
    )
    
    # Operations-Statistik
    op_stats = ops.zusammenfassung(M, block_groesse, num_blocks)
    
    return {
        'M': M,
        'mu': mu,
        'unterdrueckung_db': unterdrueckung_db,
        'output_signal': output_signal,
        'filter_coeffs': filter_coeffs.copy(),
        'operations': op_stats
    }


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================
def main():
    # ==================== PARAMETER ====================
    STOER_FREQ = 50      # Hz (Netzbrummen)
    BLOCK_GROESSE = 128  # Samples pro Block
    
    # Test-Parameter
    M_WERTE = [8, 10, 12, 14, 16, 20, 24, 32]
    MU_WERTE = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    # ==================== WAV-DATEI LADEN ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    wav_datei = os.path.join(parent_dir, "test.wav")
    
    print(f"{'='*80}")
    print("Block-LMS Algorithmus Test für DSP-Board FM4-176-S6E2CC-ETH")
    print(f"{'='*80}")
    print(f"\nLade WAV-Datei: {wav_datei}")
    
    if not os.path.exists(wav_datei):
        print(f"FEHLER: Datei '{wav_datei}' nicht gefunden!")
        return
    
    sr, data = wavfile.read(wav_datei)
    
    # Falls Stereo, nur ersten Kanal verwenden
    if len(data.shape) > 1:
        data = data[:, 0]
        print(f"  Stereo-Datei erkannt, verwende nur linken Kanal")
    
    # Normalisieren auf [-1, 1]
    if data.dtype == np.int16:
        signal = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        signal = (data.astype(np.float32) - 128) / 128.0
    else:
        signal = data.astype(np.float32)
    
    ABTAST_FREQ = float(sr)
    SIGNAL_LAENGE = len(signal)
    SIGNAL_DAUER = SIGNAL_LAENGE / ABTAST_FREQ
    
    print(f"  Abtastfrequenz: {ABTAST_FREQ:.0f} Hz")
    print(f"  Signallänge: {SIGNAL_LAENGE} Samples ({SIGNAL_DAUER:.2f} s)")
    print(f"  Störfrequenz: {STOER_FREQ} Hz")
    print(f"  Blockgröße: {BLOCK_GROESSE} Samples")
    print(f"  Anzahl Blöcke: {SIGNAL_LAENGE // BLOCK_GROESSE}")
    print(f"\nTest-Parameter:")
    print(f"  M-Werte: {M_WERTE}")
    print(f"  mu-Werte: {MU_WERTE}")
    print(f"{'='*80}\n")
    
    # ==================== TESTS DURCHFÜHREN ====================
    ergebnisse = []
    total_tests = len(M_WERTE) * len(MU_WERTE)
    test_counter = 0
    
    for M in M_WERTE:
        for mu in MU_WERTE:
            test_counter += 1
            print(f"Test {test_counter:2d}/{total_tests}: M={M:2d}, mu={mu:.4f}...", end=" ")
            
            try:
                ergebnis = teste_parameter_kombination(
                    signal.copy(), M, mu, BLOCK_GROESSE, ABTAST_FREQ, STOER_FREQ
                )
                ergebnisse.append(ergebnis)
                
                db = ergebnis['unterdrueckung_db']
                if db == float('inf'):
                    print(f"Unterdrückung: ∞ dB")
                else:
                    print(f"Unterdrückung: {db:6.2f} dB")
            except Exception as e:
                print(f"Fehler: {e}")
                ergebnisse.append({
                    'M': M, 'mu': mu, 'unterdrueckung_db': float('-inf'),
                    'output_signal': None, 'filter_coeffs': None, 'operations': None
                })
    
    # ==================== ERGEBNIS-MATRIX ====================
    print(f"\n{'='*80}")
    print("ERGEBNIS-MATRIX: Störunterdrückung bei 50Hz in dB")
    print("(Höhere Werte = bessere Unterdrückung)")
    print(f"{'='*80}\n")
    
    header = f"{'M \\ mu':>8}"
    for mu in MU_WERTE:
        header += f" {mu:>8.4f}"
    print(header)
    print("-" * len(header))
    
    ergebnis_matrix = np.zeros((len(M_WERTE), len(MU_WERTE)))
    
    for i, M in enumerate(M_WERTE):
        zeile = f"{M:>8}"
        for j, mu in enumerate(MU_WERTE):
            for e in ergebnisse:
                if e['M'] == M and e['mu'] == mu:
                    db = e['unterdrueckung_db']
                    ergebnis_matrix[i, j] = db
                    if db == float('inf'):
                        zeile += f" {'∞':>8}"
                    elif db == float('-inf'):
                        zeile += f" {'ERR':>8}"
                    else:
                        zeile += f" {db:>8.2f}"
                    break
        print(zeile)
    
    # Beste Kombination finden
    beste_idx = np.unravel_index(np.argmax(ergebnis_matrix), ergebnis_matrix.shape)
    bestes_M = M_WERTE[beste_idx[0]]
    bestes_mu = MU_WERTE[beste_idx[1]]
    beste_unterdrueckung = ergebnis_matrix[beste_idx]
    
    print(f"\n{'='*80}")
    print(f"BESTE KOMBINATION: M={bestes_M}, mu={bestes_mu}")
    print(f"Störunterdrückung: {beste_unterdrueckung:.2f} dB")
    print(f"{'='*80}")
    
    # ==================== DSP-BOARD ANALYSE ====================
    print(f"\n{'='*80}")
    print("DSP-BOARD ANALYSE: Rechenaufwand pro Parameterkombination")
    print("Zielplattform: FM4-176-S6E2CC-ETH")
    print(f"{'='*80}\n")
    
    # Tabelle mit Operationen für jedes M (mit bestem mu)
    print(f"{'M':>4} | {'Blöcke':>8} | {'sin()/Samp':>10} | {'Mult/Samp':>10} | {'Add/Samp':>10} | {'sin()/Block':>12} | {'Mult/Block':>12}")
    print("-" * 90)
    
    for M in M_WERTE:
        # Finde Ergebnis für dieses M mit bestem mu
        for e in ergebnisse:
            if e['M'] == M and e['mu'] == bestes_mu and e['operations']:
                op = e['operations']
                print(f"{M:>4} | {op['block_aufrufe']:>8} | {op['sin_pro_sample']:>10.1f} | "
                      f"{op['mult_pro_sample']:>10.1f} | {op['add_pro_sample']:>10.1f} | "
                      f"{op['sin_pro_block']:>12.0f} | {op['mult_pro_block']:>12.0f}")
                break
    
    # Detaillierte Analyse für beste Kombination
    for e in ergebnisse:
        if e['M'] == bestes_M and e['mu'] == bestes_mu and e['operations']:
            op = e['operations']
            print(f"\n{'='*80}")
            print(f"DETAILLIERTE ANALYSE für beste Kombination (M={bestes_M}, mu={bestes_mu})")
            print(f"{'='*80}")
            print(f"\nGesamt-Operationen für {SIGNAL_DAUER:.2f}s Audio:")
            print(f"  Block-Aufrufe:       {op['block_aufrufe']:>12,}")
            print(f"  Sample-Iterationen:  {op['total_samples']:>12,}")
            print(f"  Koeff-Iterationen:   {op['koeff_iterationen']:>12,}")
            print(f"\n  Additionen:          {op['additionen']:>12,}")
            print(f"  Multiplikationen:    {op['multiplikationen']:>12,}")
            print(f"  Divisionen:          {op['divisionen']:>12,}")
            print(f"  sin() Aufrufe:       {op['sin_aufrufe']:>12,}")
            print(f"  Vergleiche:          {op['vergleiche']:>12,}")
            print(f"  Zuweisungen:         {op['zuweisungen']:>12,}")
            
            print(f"\nPro Sample (bei {ABTAST_FREQ:.0f} Hz = {ABTAST_FREQ:.0f} Samples/s):")
            print(f"  Additionen:          {op['add_pro_sample']:>12.1f}")
            print(f"  Multiplikationen:    {op['mult_pro_sample']:>12.1f}")
            print(f"  sin() Aufrufe:       {op['sin_pro_sample']:>12.1f}")
            
            print(f"\nPro Block ({BLOCK_GROESSE} Samples):")
            print(f"  Additionen:          {op['add_pro_block']:>12.0f}")
            print(f"  Multiplikationen:    {op['mult_pro_block']:>12.0f}")
            print(f"  sin() Aufrufe:       {op['sin_pro_block']:>12.0f}")
            
            # Theoretische Berechnung
            print(f"\n{'='*80}")
            print("THEORETISCHE KOMPLEXITÄT pro Sample:")
            print(f"{'='*80}")
            print(f"  sin() Aufrufe:       2 * M = 2 * {bestes_M} = {2*bestes_M}")
            print(f"  Multiplikationen:    ~6 + 4*M = ~6 + 4*{bestes_M} = ~{6+4*bestes_M}")
            print(f"  Additionen:          ~4 + 4*M = ~4 + 4*{bestes_M} = ~{4+4*bestes_M}")
            
            # Speicherbedarf
            print(f"\n{'='*80}")
            print("SPEICHERBEDARF:")
            print(f"{'='*80}")
            print(f"  Filterkoeffizienten: M * 4 Bytes = {bestes_M} * 4 = {bestes_M*4} Bytes")
            print(f"  Input-Buffer:        {BLOCK_GROESSE} * 4 Bytes = {BLOCK_GROESSE*4} Bytes")
            print(f"  Output-Buffer:       {BLOCK_GROESSE} * 4 Bytes = {BLOCK_GROESSE*4} Bytes")
            print(f"  State-Variablen:     2 * 4 Bytes = 8 Bytes")
            print(f"  GESAMT:              {bestes_M*4 + BLOCK_GROESSE*4*2 + 8} Bytes")
            break
    
    # ==================== VISUALISIERUNG ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heatmap der Ergebnisse
    ax1 = axes[0, 0]
    im = ax1.imshow(ergebnis_matrix, aspect='auto', cmap='RdYlGn')
    ax1.set_xticks(range(len(MU_WERTE)))
    ax1.set_xticklabels([f'{mu:.4f}' for mu in MU_WERTE], rotation=45, ha='right')
    ax1.set_yticks(range(len(M_WERTE)))
    ax1.set_yticklabels(M_WERTE)
    ax1.set_xlabel('mu (Schrittweite)')
    ax1.set_ylabel('M (Filterordnung)')
    ax1.set_title('50Hz Störunterdrückung [dB]')
    plt.colorbar(im, ax=ax1, label='dB')
    
    for i in range(len(M_WERTE)):
        for j in range(len(MU_WERTE)):
            value = ergebnis_matrix[i, j]
            if not np.isinf(value):
                ax1.text(j, i, f'{value:.1f}', ha='center', va='center', 
                        fontsize=7, color='black' if abs(value) < 15 else 'white')
    
    # 2. Spektrum Vergleich (beste Kombination)
    ax2 = axes[0, 1]
    
    for e in ergebnisse:
        if e['M'] == bestes_M and e['mu'] == bestes_mu:
            output_best = e['output_signal']
            break
    
    # Analyse nach Einschwingphase
    analyse_start = int(0.5 * ABTAST_FREQ)
    n = len(signal) - analyse_start
    freqs = np.fft.fftfreq(n, 1/ABTAST_FREQ)[:n//2]
    
    fft_input = np.abs(np.fft.fft(signal[analyse_start:]))[:n//2]
    fft_output = np.abs(np.fft.fft(output_best[analyse_start:]))[:n//2]
    
    # Nur bis 500 Hz anzeigen
    freq_limit = 500
    idx_limit = np.argmin(np.abs(freqs - freq_limit))
    
    ax2.semilogy(freqs[:idx_limit], fft_input[:idx_limit], 'b-', alpha=0.7, label='Eingang (mit 50Hz Störung)')
    ax2.semilogy(freqs[:idx_limit], fft_output[:idx_limit], 'g-', alpha=0.7, label='Ausgang (gefiltert)')
    ax2.axvline(x=50, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax2.annotate('50 Hz', xy=(50, fft_input[np.argmin(np.abs(freqs - 50))]), 
                xytext=(80, fft_input[np.argmin(np.abs(freqs - 50))]),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    ax2.set_xlabel('Frequenz [Hz]')
    ax2.set_ylabel('Amplitude (log)')
    ax2.set_title(f'Spektrum (M={bestes_M}, μ={bestes_mu}) - nach Einschwingphase')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, freq_limit)
    
    # 3. Rechenaufwand vs. M
    ax3 = axes[1, 0]
    
    sin_pro_sample = []
    mult_pro_sample = []
    for M in M_WERTE:
        for e in ergebnisse:
            if e['M'] == M and e['mu'] == bestes_mu and e['operations']:
                sin_pro_sample.append(e['operations']['sin_pro_sample'])
                mult_pro_sample.append(e['operations']['mult_pro_sample'])
                break
    
    ax3.plot(M_WERTE, sin_pro_sample, 'ro-', label='sin() Aufrufe', linewidth=2, markersize=8)
    ax3.plot(M_WERTE, mult_pro_sample, 'bs-', label='Multiplikationen', linewidth=2, markersize=8)
    ax3.set_xlabel('M (Filterordnung)')
    ax3.set_ylabel('Operationen pro Sample')
    ax3.set_title('Rechenaufwand vs. Filterordnung (DSP-Board Analyse)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Unterdrückung vs. M für verschiedene mu
    ax4 = axes[1, 1]
    for j, mu in enumerate(MU_WERTE[::2]):
        mu_idx = MU_WERTE.index(mu)
        ax4.plot(M_WERTE, ergebnis_matrix[:, mu_idx], 'o-', label=f'μ={mu}', linewidth=2)
    ax4.set_xlabel('M (Filterordnung)')
    ax4.set_ylabel('Störunterdrückung [dB]')
    ax4.set_title('Unterdrückung vs. Filterordnung')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern
    output_dir = os.path.join(parent_dir, 'Output')
    output_path = os.path.join(output_dir, 'blockLMS_testergebnis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGrafik gespeichert: {output_path}")
    
    plt.show()
    
    # CSV speichern
    csv_path = os.path.join(output_dir, 'blockLMS_testergebnis.csv')
    with open(csv_path, 'w') as f:
        f.write('M,mu,Unterdrueckung_dB,sin_pro_sample,mult_pro_sample,add_pro_sample\n')
        for e in ergebnisse:
            if e['operations']:
                op = e['operations']
                f.write(f"{e['M']},{e['mu']},{e['unterdrueckung_db']:.4f},"
                       f"{op['sin_pro_sample']:.1f},{op['mult_pro_sample']:.1f},{op['add_pro_sample']:.1f}\n")
    print(f"CSV gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
