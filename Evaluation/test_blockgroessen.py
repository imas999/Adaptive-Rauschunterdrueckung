"""
Block-LMS Test: Verschiedene Blockgrößen und Abtastfrequenzen
Feste Parameter: M=24, μ=0.05
Blockgrößen: 64, 128, 256, 512
Abtastfrequenzen: 8kHz, 48kHz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
import os
from datetime import datetime

# Feste Parameter
M = 24  # Filterordnung
MU = 0.05  # Schrittweite
NOISE_FREQ = 50.0  # Hz

# Testparameter
BLOCK_SIZES = [64, 128, 256, 512]
SAMPLE_RATES = [8000, 48000]

# Algorithmus-Konstanten (wie in C++)
SILENCE_THRESHOLD = 1e-7
LEAKAGE_FACTOR = 0.999
MAX_COEFF_VALUE = 10.0
ENERGY_SMOOTHING = 0.99

class OperationsZaehler:
    """Zählt DSP-Operationen"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.multiplikationen = 0
        self.additionen = 0
        self.vergleiche = 0
        self.speicherzugriffe = 0
    
    def get_total(self):
        return self.multiplikationen + self.additionen + self.vergleiche

def blockLMS(input_block, block_size, sample_rate, M, mu, ops, state):
    """
    Block-LMS Algorithmus - 1:1 wie C++ Implementierung
    """
    output = np.zeros(block_size, dtype=np.float32)
    
    # Referenzsignal generieren (Sinuswelle bei NOISE_FREQ)
    reference = np.zeros((block_size, M), dtype=np.float32)
    
    for n in range(block_size):
        for m in range(M):
            t = (state['sample_count'] + n - m) / sample_rate
            reference[n, m] = np.sin(2 * np.pi * NOISE_FREQ * t)
            ops.multiplikationen += 2
            ops.additionen += 1
            ops.speicherzugriffe += 1
    
    # Filterung und Adaption für jeden Sample im Block
    for n in range(block_size):
        # Gefilterte Ausgabe berechnen (y = w^T * x)
        y = 0.0
        for m in range(M):
            y += state['weights'][m] * reference[n, m]
            ops.multiplikationen += 1
            ops.additionen += 1
            ops.speicherzugriffe += 2
        
        # Fehlersignal (gewünschtes Signal - Filterausgang)
        error = input_block[n] - y
        ops.additionen += 1
        
        # Ausgabe ist das Fehlersignal (Störung entfernt)
        output[n] = error
        
        # Energie des Referenzsignals berechnen
        ref_energy = 0.0
        for m in range(M):
            ref_energy += reference[n, m] * reference[n, m]
            ops.multiplikationen += 1
            ops.additionen += 1
            ops.speicherzugriffe += 1
        
        # Geglättete Energie
        state['smoothed_energy'] = (ENERGY_SMOOTHING * state['smoothed_energy'] + 
                                    (1 - ENERGY_SMOOTHING) * ref_energy)
        ops.multiplikationen += 3
        ops.additionen += 1
        
        # Normalisierungsfaktor
        norm_factor = state['smoothed_energy'] + 1e-10
        ops.additionen += 1
        
        # Stille-Erkennung
        ops.vergleiche += 1
        if abs(input_block[n]) > SILENCE_THRESHOLD:
            # Gewichte aktualisieren (LMS Update)
            step = mu * error / norm_factor
            ops.multiplikationen += 1
            ops.vergleiche += 1
            
            for m in range(M):
                # Leakage anwenden
                state['weights'][m] *= LEAKAGE_FACTOR
                ops.multiplikationen += 1
                ops.speicherzugriffe += 2
                
                # LMS Update
                state['weights'][m] += step * reference[n, m]
                ops.multiplikationen += 1
                ops.additionen += 1
                ops.speicherzugriffe += 2
                
                # Clipping
                ops.vergleiche += 2
                if state['weights'][m] > MAX_COEFF_VALUE:
                    state['weights'][m] = MAX_COEFF_VALUE
                elif state['weights'][m] < -MAX_COEFF_VALUE:
                    state['weights'][m] = -MAX_COEFF_VALUE
    
    state['sample_count'] += block_size
    return output

def berechne_50hz_unterdrueckung(original, gefiltert, sample_rate):
    """Berechnet die Unterdrückung bei 50Hz in dB"""
    n = len(original)
    
    # FFT berechnen
    freq = np.fft.rfftfreq(n, 1/sample_rate)
    fft_original = np.abs(np.fft.rfft(original))
    fft_gefiltert = np.abs(np.fft.rfft(gefiltert))
    
    # 50Hz Index finden
    idx_50hz = np.argmin(np.abs(freq - 50))
    
    # Bereich um 50Hz (±5Hz)
    bandwidth = 5
    idx_low = np.argmin(np.abs(freq - (50 - bandwidth)))
    idx_high = np.argmin(np.abs(freq - (50 + bandwidth)))
    
    # Energie im 50Hz Bereich
    energie_original = np.sum(fft_original[idx_low:idx_high+1]**2)
    energie_gefiltert = np.sum(fft_gefiltert[idx_low:idx_high+1]**2)
    
    if energie_gefiltert > 0 and energie_original > 0:
        unterdrueckung_db = 10 * np.log10(energie_original / energie_gefiltert)
    else:
        unterdrueckung_db = float('inf')
    
    return unterdrueckung_db

def erstelle_testsignal(sample_rate, dauer_sekunden=5.0):
    """
    Erstellt ein synthetisches Testsignal mit Sprache-ähnlichem Rauschen + 50Hz Störung
    """
    n_samples = int(sample_rate * dauer_sekunden)
    t = np.arange(n_samples) / sample_rate
    
    # Sprache-ähnliches Signal (gefiltetes Rauschen mit Formanten)
    np.random.seed(42)  # Reproduzierbar
    rauschen = np.random.randn(n_samples) * 0.3
    
    # Bandpass-Filter für Sprach-ähnliches Spektrum (300-3400 Hz)
    nyq = sample_rate / 2
    low = min(300 / nyq, 0.99)
    high = min(3400 / nyq, 0.99)
    if high > low:
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        sprache = scipy_signal.filtfilt(b, a, rauschen)
    else:
        sprache = rauschen * 0.5
    
    # 50Hz Störsignal
    stoerung = 0.15 * np.sin(2 * np.pi * 50 * t)
    
    # Kombiniertes Signal
    signal = (sprache + stoerung).astype(np.float32)
    
    # Normalisieren
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal

def lade_oder_erstelle_signal(sample_rate, wav_path):
    """
    Lädt test.wav für 8kHz oder erstellt/resamplet für andere Raten
    """
    if sample_rate == 8000 and os.path.exists(wav_path):
        # Original test.wav laden
        sr, data = wavfile.read(wav_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # Nur erste 5 Sekunden für konsistente Tests
        max_samples = min(len(data), 5 * sample_rate)
        return data[:max_samples]
    
    elif sample_rate == 48000 and os.path.exists(wav_path):
        # test.wav laden und auf 48kHz resampling
        sr, data = wavfile.read(wav_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # Resampling von 8kHz auf 48kHz
        resampling_factor = sample_rate / sr
        n_samples_neu = int(len(data) * resampling_factor)
        data_resampled = scipy_signal.resample(data, n_samples_neu)
        
        # 50Hz Störung muss neu hinzugefügt werden (wurde beim Resampling verzerrt)
        t = np.arange(len(data_resampled)) / sample_rate
        # Alte Störung entfernen (Hochpass bei 60Hz)
        nyq = sample_rate / 2
        b, a = scipy_signal.butter(4, 60/nyq, btype='high')
        data_clean = scipy_signal.filtfilt(b, a, data_resampled)
        
        # Neue 50Hz Störung hinzufügen
        stoerung = 0.1 * np.sin(2 * np.pi * 50 * t)
        data_mit_stoerung = (data_clean + stoerung).astype(np.float32)
        
        # Nur erste 5 Sekunden
        max_samples = min(len(data_mit_stoerung), 5 * sample_rate)
        return data_mit_stoerung[:max_samples]
    
    else:
        # Synthetisches Signal erstellen
        return erstelle_testsignal(sample_rate)

def run_tests():
    """Führt alle Tests durch"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    wav_path = os.path.join(parent_dir, "test.wav")
    
    print("=" * 70)
    print("Block-LMS Test: Blockgrößen und Abtastfrequenzen")
    print(f"Feste Parameter: M={M}, μ={MU}")
    print("=" * 70)
    
    ergebnisse = []
    
    for sample_rate in SAMPLE_RATES:
        print(f"\n{'='*70}")
        print(f"Abtastfrequenz: {sample_rate} Hz")
        print("=" * 70)
        
        # Testsignal laden/erstellen
        signal = lade_oder_erstelle_signal(sample_rate, wav_path)
        signal_dauer = len(signal) / sample_rate
        print(f"Signallänge: {len(signal)} Samples ({signal_dauer:.2f} s)")
        
        for block_size in BLOCK_SIZES:
            print(f"\n--- Blockgröße: {block_size} ---")
            
            # State initialisieren
            state = {
                'weights': np.zeros(M, dtype=np.float32),
                'smoothed_energy': 0.0,
                'sample_count': 0
            }
            
            ops = OperationsZaehler()
            
            # Signal in Blöcke aufteilen und verarbeiten
            n_blocks = len(signal) // block_size
            output = np.zeros(n_blocks * block_size, dtype=np.float32)
            
            for i in range(n_blocks):
                start = i * block_size
                end = start + block_size
                input_block = signal[start:end]
                output[start:end] = blockLMS(input_block, block_size, sample_rate, 
                                             M, MU, ops, state)
            
            # Unterdrückung berechnen (nur ab 20% des Signals - nach Konvergenz)
            start_analyse = int(0.2 * len(output))
            unterdrueckung = berechne_50hz_unterdrueckung(
                signal[start_analyse:n_blocks*block_size],
                output[start_analyse:],
                sample_rate
            )
            
            # Operationen pro Sample
            total_samples = n_blocks * block_size
            ops_pro_sample = ops.get_total() / total_samples
            ops_pro_block = ops.get_total() / n_blocks
            
            # Latenz berechnen (1 Block Verzögerung)
            latenz_ms = (block_size / sample_rate) * 1000
            
            ergebnis = {
                'sample_rate': sample_rate,
                'block_size': block_size,
                'unterdrueckung_db': unterdrueckung,
                'ops_pro_sample': ops_pro_sample,
                'ops_pro_block': ops_pro_block,
                'multiplikationen': ops.multiplikationen,
                'additionen': ops.additionen,
                'latenz_ms': latenz_ms,
                'signal': signal[:n_blocks*block_size].copy(),
                'output': output.copy()
            }
            ergebnisse.append(ergebnis)
            
            print(f"  50Hz Unterdrückung: {unterdrueckung:.2f} dB")
            print(f"  Operationen/Sample: {ops_pro_sample:.1f}")
            print(f"  Operationen/Block:  {ops_pro_block:.1f}")
            print(f"  Latenz:             {latenz_ms:.2f} ms")
    
    return ergebnisse

def erstelle_plots(ergebnisse, output_dir):
    """Erstellt Visualisierungen der Ergebnisse"""
    
    # Plot 1: Unterdrückung vs Blockgröße für verschiedene Sample-Raten
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Unterdrückung
    ax1 = axes[0, 0]
    for sr in SAMPLE_RATES:
        sr_ergebnisse = [e for e in ergebnisse if e['sample_rate'] == sr]
        block_sizes = [e['block_size'] for e in sr_ergebnisse]
        unterdrueckung = [e['unterdrueckung_db'] for e in sr_ergebnisse]
        ax1.plot(block_sizes, unterdrueckung, 'o-', label=f'{sr/1000:.0f} kHz', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Blockgröße [Samples]')
    ax1.set_ylabel('50Hz Unterdrückung [dB]')
    ax1.set_title('50Hz Unterdrückung vs. Blockgröße')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(BLOCK_SIZES)
    ax1.set_xticklabels(BLOCK_SIZES)
    
    # Latenz
    ax2 = axes[0, 1]
    for sr in SAMPLE_RATES:
        sr_ergebnisse = [e for e in ergebnisse if e['sample_rate'] == sr]
        block_sizes = [e['block_size'] for e in sr_ergebnisse]
        latenz = [e['latenz_ms'] for e in sr_ergebnisse]
        ax2.plot(block_sizes, latenz, 'o-', label=f'{sr/1000:.0f} kHz', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Blockgröße [Samples]')
    ax2.set_ylabel('Latenz [ms]')
    ax2.set_title('Verarbeitungslatenz vs. Blockgröße')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(BLOCK_SIZES)
    ax2.set_xticklabels(BLOCK_SIZES)
    
    # Operationen pro Sample
    ax3 = axes[1, 0]
    for sr in SAMPLE_RATES:
        sr_ergebnisse = [e for e in ergebnisse if e['sample_rate'] == sr]
        block_sizes = [e['block_size'] for e in sr_ergebnisse]
        ops = [e['ops_pro_sample'] for e in sr_ergebnisse]
        ax3.plot(block_sizes, ops, 'o-', label=f'{sr/1000:.0f} kHz', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Blockgröße [Samples]')
    ax3.set_ylabel('Operationen pro Sample')
    ax3.set_title('Rechenaufwand vs. Blockgröße')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(BLOCK_SIZES)
    ax3.set_xticklabels(BLOCK_SIZES)
    
    # Operationen pro Block
    ax4 = axes[1, 1]
    for sr in SAMPLE_RATES:
        sr_ergebnisse = [e for e in ergebnisse if e['sample_rate'] == sr]
        block_sizes = [e['block_size'] for e in sr_ergebnisse]
        ops = [e['ops_pro_block'] for e in sr_ergebnisse]
        ax4.plot(block_sizes, ops, 'o-', label=f'{sr/1000:.0f} kHz', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Blockgröße [Samples]')
    ax4.set_ylabel('Operationen pro Block')
    ax4.set_title('Rechenaufwand pro Block vs. Blockgröße')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    ax4.set_xticks(BLOCK_SIZES)
    ax4.set_xticklabels(BLOCK_SIZES)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'blockgroessen_vergleich.png'), dpi=150)
    plt.close()
    
    # Plot 2: Spektren für beste Konfiguration pro Sample-Rate
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, sr in enumerate(SAMPLE_RATES):
        sr_ergebnisse = [e for e in ergebnisse if e['sample_rate'] == sr]
        # Beste Konfiguration nach Unterdrückung
        best = max(sr_ergebnisse, key=lambda x: x['unterdrueckung_db'])
        
        signal = best['signal']
        output = best['output']
        block_size = best['block_size']
        
        # Zeitbereich (erste 100ms)
        n_plot = int(0.1 * sr)
        t = np.arange(n_plot) / sr * 1000
        
        ax_time = axes[idx, 0]
        ax_time.plot(t, signal[:n_plot], 'b-', alpha=0.7, label='Original')
        ax_time.plot(t, output[:n_plot], 'r-', alpha=0.7, label='Gefiltert')
        ax_time.set_xlabel('Zeit [ms]')
        ax_time.set_ylabel('Amplitude')
        ax_time.set_title(f'{sr/1000:.0f} kHz - Zeitbereich (Block={block_size})')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        # Frequenzbereich
        n_fft = len(signal)
        freq = np.fft.rfftfreq(n_fft, 1/sr)
        fft_orig = 20 * np.log10(np.abs(np.fft.rfft(signal)) + 1e-10)
        fft_filt = 20 * np.log10(np.abs(np.fft.rfft(output)) + 1e-10)
        
        ax_freq = axes[idx, 1]
        ax_freq.plot(freq, fft_orig, 'b-', alpha=0.7, label='Original')
        ax_freq.plot(freq, fft_filt, 'r-', alpha=0.7, label='Gefiltert')
        ax_freq.axvline(x=50, color='g', linestyle='--', alpha=0.5, label='50 Hz')
        ax_freq.set_xlabel('Frequenz [Hz]')
        ax_freq.set_ylabel('Magnitude [dB]')
        ax_freq.set_title(f'{sr/1000:.0f} kHz - Spektrum (Block={block_size})')
        ax_freq.set_xlim([0, min(500, sr/2)])
        ax_freq.legend()
        ax_freq.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'blockgroessen_spektren.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots gespeichert in: {output_dir}")

def erstelle_markdown(ergebnisse, output_path):
    """Erstellt Markdown-Dokumentation"""
    
    md = []
    md.append("# Block-LMS Test: Blockgrößen und Abtastfrequenzen")
    md.append("")
    md.append(f"**Testdatum:** {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    md.append("")
    md.append("## Testparameter")
    md.append("")
    md.append(f"- **Filterordnung (M):** {M}")
    md.append(f"- **Schrittweite (μ):** {MU}")
    md.append(f"- **Störfrequenz:** {NOISE_FREQ} Hz")
    md.append(f"- **Blockgrößen:** {BLOCK_SIZES}")
    md.append(f"- **Abtastfrequenzen:** {[f'{sr/1000:.0f} kHz' for sr in SAMPLE_RATES]}")
    md.append("")
    
    # Ergebnistabelle
    md.append("## Ergebnisübersicht")
    md.append("")
    md.append("| Sample-Rate | Blockgröße | 50Hz Unterdr. | Ops/Sample | Ops/Block | Latenz |")
    md.append("|-------------|------------|---------------|------------|-----------|--------|")
    
    for e in ergebnisse:
        sr_str = f"{e['sample_rate']/1000:.0f} kHz"
        block_str = f"{e['block_size']}"
        unter_str = f"{e['unterdrueckung_db']:.2f} dB"
        ops_sample = f"{e['ops_pro_sample']:.0f}"
        ops_block = f"{e['ops_pro_block']:.0f}"
        latenz_str = f"{e['latenz_ms']:.2f} ms"
        md.append(f"| {sr_str} | {block_str} | {unter_str} | {ops_sample} | {ops_block} | {latenz_str} |")
    
    md.append("")
    
    # Analyse pro Sample-Rate
    for sr in SAMPLE_RATES:
        sr_ergebnisse = [e for e in ergebnisse if e['sample_rate'] == sr]
        best = max(sr_ergebnisse, key=lambda x: x['unterdrueckung_db'])
        
        md.append(f"## Analyse: {sr/1000:.0f} kHz Abtastfrequenz")
        md.append("")
        md.append(f"### Beste Konfiguration")
        md.append(f"- **Blockgröße:** {best['block_size']} Samples")
        md.append(f"- **50Hz Unterdrückung:** {best['unterdrueckung_db']:.2f} dB")
        md.append(f"- **Latenz:** {best['latenz_ms']:.2f} ms")
        md.append(f"- **Operationen pro Sample:** {best['ops_pro_sample']:.0f}")
        md.append("")
        
        md.append("### Latenz-Übersicht")
        md.append("")
        for e in sr_ergebnisse:
            md.append(f"- Block {e['block_size']}: {e['latenz_ms']:.2f} ms")
        md.append("")
    
    # Trade-offs
    md.append("## Trade-off Analyse")
    md.append("")
    md.append("### Blockgröße vs. Latenz")
    md.append("")
    md.append("Die Verarbeitungslatenz ist direkt proportional zur Blockgröße:")
    md.append("")
    md.append("$$\\text{Latenz} = \\frac{\\text{Blockgröße}}{\\text{Abtastfrequenz}}$$")
    md.append("")
    md.append("| Blockgröße | Latenz @ 8 kHz | Latenz @ 48 kHz |")
    md.append("|------------|----------------|-----------------|")
    for bs in BLOCK_SIZES:
        lat_8k = bs / 8000 * 1000
        lat_48k = bs / 48000 * 1000
        md.append(f"| {bs} | {lat_8k:.2f} ms | {lat_48k:.3f} ms |")
    md.append("")
    
    md.append("### Rechenaufwand")
    md.append("")
    md.append("Der Rechenaufwand pro Sample bleibt konstant unabhängig von der Blockgröße,")
    md.append("da der LMS-Algorithmus sample-weise arbeitet.")
    md.append("")
    
    # Empfehlungen
    md.append("## Empfehlungen")
    md.append("")
    md.append("### Für Echtzeit-Anwendungen (niedrige Latenz)")
    md.append("")
    md.append("- **8 kHz:** Blockgröße 64 (8 ms Latenz)")
    md.append("- **48 kHz:** Blockgröße 64 (1.33 ms Latenz)")
    md.append("")
    md.append("### Für maximale Unterdrückung")
    md.append("")
    
    for sr in SAMPLE_RATES:
        sr_ergebnisse = [e for e in ergebnisse if e['sample_rate'] == sr]
        best = max(sr_ergebnisse, key=lambda x: x['unterdrueckung_db'])
        md.append(f"- **{sr/1000:.0f} kHz:** Blockgröße {best['block_size']} ({best['unterdrueckung_db']:.2f} dB)")
    md.append("")
    
    md.append("### DSP-Board Implementierung (FM4-176-S6E2CC-ETH)")
    md.append("")
    md.append("Für das DSP-Board empfehle ich:")
    md.append("")
    md.append("1. **Blockgröße 128** als guter Kompromiss zwischen:")
    md.append("   - Latenz (16 ms @ 8 kHz, 2.67 ms @ 48 kHz)")
    md.append("   - Effizienter DMA-Nutzung")
    md.append("   - Guter Unterdrückungsleistung")
    md.append("")
    md.append("2. **Bei höheren Echtzeitanforderungen:** Blockgröße 64")
    md.append("")
    
    # Visualisierungen
    md.append("## Visualisierungen")
    md.append("")
    md.append("### Vergleichsdiagramme")
    md.append("![Blockgrößen Vergleich](blockgroessen_vergleich.png)")
    md.append("")
    md.append("### Spektralanalyse")
    md.append("![Spektren](blockgroessen_spektren.png)")
    md.append("")
    
    # Zusammenfassung
    md.append("## Zusammenfassung")
    md.append("")
    md.append("Die Tests zeigen, dass die 50Hz-Unterdrückung weitgehend unabhängig von der")
    md.append("Blockgröße ist, da der LMS-Algorithmus sample-weise adaptiert. Die Blockgröße")
    md.append("beeinflusst hauptsächlich:")
    md.append("")
    md.append("1. **Latenz:** Größere Blöcke = höhere Latenz")
    md.append("2. **DMA-Effizienz:** Größere Blöcke = weniger DMA-Transfers")
    md.append("3. **Interrupt-Overhead:** Größere Blöcke = weniger Interrupts")
    md.append("")
    md.append("Die Abtastfrequenz hat einen signifikanten Einfluss:")
    md.append("")
    md.append("- **48 kHz:** Mehr Samples pro Periode der 50Hz-Störung → feinere Adaption")
    md.append("- **8 kHz:** Weniger Samples → gröbere Adaption, aber ausreichend für 50Hz")
    md.append("")
    
    # Schreiben
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    
    print(f"\nMarkdown gespeichert: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, "Output")
    
    # Tests durchführen
    ergebnisse = run_tests()
    
    # Plots erstellen
    erstelle_plots(ergebnisse, output_dir)
    
    # Markdown erstellen
    md_path = os.path.join(output_dir, "Testergebnisse_Blockgroessen.md")
    erstelle_markdown(ergebnisse, md_path)
    
    print("\n" + "=" * 70)
    print("Tests abgeschlossen!")
    print("=" * 70)

if __name__ == "__main__":
    main()
