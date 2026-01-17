"""
Kontinuierliche Block-LMS Verarbeitung
Zeigt wie der Filter über einen längeren Zeitraum arbeitet und einschwingt.

Beste Kombination: M=20, mu=0.05
"""

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import os

# Konfigurationsparameter (entsprechend LMS.hpp)
SILENCE_THRESHOLD = 1e-7
LEAKAGE_FACTOR = 0.999
MAX_COEFF_VALUE = 10.0
ENERGY_SMOOTHING = 0.99
ENERGY_SMOOTHING_INV = 0.01


def clipping_check(coeff: float) -> float:
    """Begrenzt Koeffizientenwerte zur Stabilitätssicherung."""
    if coeff > MAX_COEFF_VALUE:
        return MAX_COEFF_VALUE
    elif coeff < -MAX_COEFF_VALUE:
        return -MAX_COEFF_VALUE
    return coeff


def block_lms(input_block: np.ndarray, filter_coeffs: np.ndarray, 
              M: int, mu: float, stoer_freq: float, abtast_freq: float,
              state: dict) -> np.ndarray:
    """
    Block-LMS Algorithmus zur adaptiven Störunterdrückung.
    Verarbeitet einen Block und aktualisiert den Zustand für den nächsten Block.
    """
    zwei_pi = 2.0 * np.pi
    phasen_increment = zwei_pi * stoer_freq / abtast_freq
    
    anz_samples = len(input_block)
    output_block = np.zeros(anz_samples, dtype=np.float32)
    
    momentane_phase = state['momentane_phase']
    signal_energie = state['signal_energie']
    
    for i in range(anz_samples):
        input_sample = input_block[i]
        
        # Signalenergie-Schätzung
        signal_energie = ENERGY_SMOOTHING * signal_energie + ENERGY_SMOOTHING_INV * (input_sample * input_sample)
        
        # Adaptive Step-Size
        adaptive_mu = mu
        ist_stille = signal_energie < SILENCE_THRESHOLD
        if ist_stille:
            adaptive_mu = mu * 0.01
        zwei_mu = 2.0 * adaptive_mu
        
        # Berechnung des Filteroutputs
        filter_output = 0.0
        for k in range(M):
            sin_referenz = np.sin(momentane_phase - k * phasen_increment)
            filter_output += filter_coeffs[k] * sin_referenz
        
        # Fehlersignal (bereinigtes Signal)
        fehler_signal = input_sample - filter_output
        output_block[i] = fehler_signal
        
        # Koeffizienten-Update
        update_faktor = zwei_mu * fehler_signal
        
        if ist_stille:
            for k in range(M):
                sin_referenz = np.sin(momentane_phase - k * phasen_increment)
                filter_coeffs[k] = LEAKAGE_FACTOR * filter_coeffs[k] + update_faktor * sin_referenz
                filter_coeffs[k] = clipping_check(filter_coeffs[k])
        else:
            for k in range(M):
                sin_referenz = np.sin(momentane_phase - k * phasen_increment)
                filter_coeffs[k] += update_faktor * sin_referenz
                filter_coeffs[k] = clipping_check(filter_coeffs[k])
        
        # Phase aktualisieren
        momentane_phase += phasen_increment
        if momentane_phase >= zwei_pi:
            momentane_phase -= zwei_pi
    
    # Zustand für nächsten Block speichern
    state['momentane_phase'] = momentane_phase
    state['signal_energie'] = signal_energie
    
    return output_block


def main():
    # ==================== PARAMETER ====================
    ABTAST_FREQ = 48000  # Hz
    STOER_FREQ = 50      # Hz (Netzbrummen)
    BLOCK_GROESSE = 128  # Samples pro Block
    
    # Beste Kombination aus vorherigem Test
    M = 20
    MU = 0.05
    
    # ==================== WAV-DATEI LADEN ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    wav_datei = os.path.join(parent_dir, "test.wav")
    
    print(f"Lade WAV-Datei: {wav_datei}")
    
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
        input_signal = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        input_signal = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        input_signal = (data.astype(np.float32) - 128) / 128.0
    else:
        input_signal = data.astype(np.float32)
    
    ABTAST_FREQ = float(sr)
    SIGNAL_LAENGE = len(input_signal)
    SIGNAL_DAUER = SIGNAL_LAENGE / ABTAST_FREQ
    
    print(f"  Abtastfrequenz: {ABTAST_FREQ:.0f} Hz")
    print(f"  Signallänge: {SIGNAL_LAENGE} Samples ({SIGNAL_DAUER:.2f} s)")
    print()
    
    print(f"{'='*70}")
    print("Kontinuierliche Block-LMS Verarbeitung")
    print(f"{'='*70}")
    print(f"Parameter: M={M}, mu={MU}")
    print(f"Abtastfrequenz: {ABTAST_FREQ:.0f} Hz")
    print(f"Störfrequenz: {STOER_FREQ} Hz")
    print(f"Blockgröße: {BLOCK_GROESSE} Samples")
    print(f"Signaldauer: {SIGNAL_DAUER:.2f} s ({SIGNAL_LAENGE} Samples)")
    print(f"Anzahl Blöcke: {SIGNAL_LAENGE // BLOCK_GROESSE}")
    print(f"{'='*70}\n")
    
    # ==================== KONTINUIERLICHE BLOCK-VERARBEITUNG ====================
    print("Starte kontinuierliche Blockverarbeitung...")
    
    # Filter initialisieren
    filter_coeffs = np.zeros(M, dtype=np.float32)
    state = {'momentane_phase': 0.0, 'signal_energie': 0.0}
    
    # Ausgangssignal vorbereiten
    output_signal = np.zeros(SIGNAL_LAENGE, dtype=np.float32)
    
    # Blockweise Verarbeitung
    num_blocks = SIGNAL_LAENGE // BLOCK_GROESSE
    
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_GROESSE
        end = start + BLOCK_GROESSE
        
        # Block verarbeiten
        input_block = input_signal[start:end]
        output_block = block_lms(input_block, filter_coeffs, M, MU, 
                                  STOER_FREQ, ABTAST_FREQ, state)
        output_signal[start:end] = output_block
        
        # Fortschritt anzeigen
        if (block_idx + 1) % 500 == 0:
            progress = (block_idx + 1) / num_blocks * 100
            print(f"  Block {block_idx + 1}/{num_blocks} ({progress:.1f}%)")
    
    # Restliche Samples verarbeiten
    rest_start = num_blocks * BLOCK_GROESSE
    if rest_start < SIGNAL_LAENGE:
        input_block = input_signal[rest_start:]
        output_block = block_lms(input_block, filter_coeffs, M, MU, 
                                  STOER_FREQ, ABTAST_FREQ, state)
        output_signal[rest_start:] = output_block
    
    print("Verarbeitung abgeschlossen!\n")
    
    # ==================== ANALYSE ====================
    # Störunterdrückung berechnen (nach Einschwingphase)
    analyse_start = int(0.5 * ABTAST_FREQ)  # Nach 0.5 Sekunden
    
    # FFT für Spektralanalyse
    n_analyse = SIGNAL_LAENGE - analyse_start
    freqs = np.fft.fftfreq(n_analyse, 1/ABTAST_FREQ)[:n_analyse//2]
    
    fft_input = np.abs(np.fft.fft(input_signal[analyse_start:]))[:n_analyse//2]
    fft_output = np.abs(np.fft.fft(output_signal[analyse_start:]))[:n_analyse//2]
    
    # 50Hz Unterdrückung berechnen
    idx_50hz = np.argmin(np.abs(freqs - 50))
    
    unterdrueckung_50hz = 20 * np.log10(fft_input[idx_50hz] / max(fft_output[idx_50hz], 1e-10))
    
    print(f"Störunterdrückung (nach Einschwingphase):")
    print(f"  50 Hz:  {unterdrueckung_50hz:.1f} dB")
    print()
    
    # ==================== VISUALISIERUNG ====================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Frequenzspektrum (nach Einschwingphase)
    freq_limit = 2000  # Hz
    idx_limit = np.argmin(np.abs(freqs - freq_limit))
    
    ax.semilogy(freqs[:idx_limit], fft_input[:idx_limit], 'b-', alpha=0.7, linewidth=1.5, label='Eingangssignal (gestört)')
    ax.semilogy(freqs[:idx_limit], fft_output[:idx_limit], 'g-', alpha=0.9, linewidth=1.5, label='Ausgangssignal (gefiltert)')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.annotate('50 Hz Störung', xy=(50, fft_input[idx_50hz]), xytext=(100, fft_input[idx_50hz]*1.5),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    ax.set_xlabel('Frequenz [Hz]', fontsize=12)
    ax.set_ylabel('Amplitude (log)', fontsize=12)
    ax.set_title(f'Frequenzspektrum nach Einschwingphase (M={M}, μ={MU}) - 50Hz Unterdrückung: {unterdrueckung_50hz:.1f} dB', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, freq_limit)
    
    plt.tight_layout()
    
    # Speichern
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Output')
    output_path = os.path.join(output_dir, 'blockLMS_kontinuierlich.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grafik gespeichert: {output_path}")
    
    plt.show()
    
    # ==================== ZUSAMMENFASSUNG ====================
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"Parameter: M={M}, mu={MU}, Blockgröße={BLOCK_GROESSE}")
    print(f"Verarbeitete Blöcke: {num_blocks}")
    print(f"Gesamtdauer: {SIGNAL_DAUER} Sekunden")
    print(f"\nStörunterdrückung:")
    print(f"  50 Hz Netzbrummen:  {unterdrueckung_50hz:.1f} dB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
