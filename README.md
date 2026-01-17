# Adaptive Rauschunterdrückung mit Block-LMS

Adaptive 50Hz-Störunterdrückung für DSP-Board FM4-176-S6E2CC-ETH mittels Block-LMS Algorithmus.
Dieses Projekt implementiert einen adaptiven Block-LMS Filter zur Unterdrückung von 50Hz Netzbrummen in Audiosignalen. 
Die Implementierung ist optimiert für Echtzeit-Verarbeitung auf einem Embedded DSP-Board.

### C++ Implementierung (DSP-Board)

Die C++ Dateien (`blockLMS.cpp`, `LMS.hpp`) können direkt in Ihr Embedded-Projekt integriert werden.

### Python Evaluations-Tools

```bash
# Virtual Environment erstellen
python -m venv .venv

# Aktivieren (Windows)
.venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Verwendung

### C++ Algorithmus

```cpp
#include "LMS.hpp"

// In Ihrer Audio-Callback-Funktion:
float input_block[BLOCK_SIZE];  // Input samples
float output_block[BLOCK_SIZE]; // Output samples

// Für jeden Block:
blockLMS(input_block, output_block, BLOCK_SIZE, SAMPLE_RATE);
```

### Python Tests

```bash
# Vollständiger Parameter-Sweep
python Evaluation/test_blockLMS.py

# Blockgrößen-Analyse
python Evaluation/test_blockgroessen.py

# Frequenzselektivität testen
python Evaluation/analyse_frequenzselektivitaet.py

# Kontinuierliche Verarbeitung visualisieren
python Evaluation/test_blockLMS_kontinuierlich.py
```

## Testergebnisse

### Empfohlene Parameter

| Sample-Rate | Filterordnung (M) | Schrittweite (μ) | 50Hz Unterdrückung | Latenz @ Block=128 |
|-------------|-------------------|------------------|--------------------|--------------------|
| **8 kHz**   | 24                | 0.05             | 5.3 dB             | 16.0 ms            |
| **48 kHz**  | 24                | 0.05             | 27.5 dB            | 2.67 ms            |

### Wichtige Erkenntnisse

1. **Höhere Abtastrate = bessere Unterdrückung**
   - 48 kHz: 960 Samples pro 50Hz-Periode → präzisere Adaption
   - 8 kHz: 160 Samples pro 50Hz-Periode → ausreichend für grundlegende Unterdrückung

2. **Blockgröße beeinflusst nur Latenz**
   - Unterdrückungsleistung bleibt konstant (sample-weise Adaption)
   - Trade-off: Latenz vs. DMA-Effizienz

3. **DSP-Rechenaufwand**
   - ~297 Operationen pro Sample bei M=24
   - Skaliert linear mit Filterordnung

Detaillierte Ergebnisse finden Sie in den PDF-Reports im `Output/` Ordner.

## Konfiguration

In `LMS.hpp` können folgende Parameter angepasst werden:

```cpp
#define SAMPLE_RATE 48000           // Abtastfrequenz [Hz]
#define BLOCK_SIZE 128              // Blockgröße [Samples]
#define M 24                        // Filterordnung
#define MU 0.05f                    // Schrittweite
#define NOISE_FREQ 50.0f            // Störfrequenz [Hz]

// Algorithmus-Parameter
#define SILENCE_THRESHOLD 1e-7f     // Schwellwert für Stille-Erkennung
#define LEAKAGE_FACTOR 0.999f       // Leakage für Stabilität
#define MAX_COEFF_VALUE 10.0f       // Koeffizienten-Clipping
#define ENERGY_SMOOTHING 0.99f      // Glättungsfaktor
```

## Algorithmus-Details

Der Block-LMS Filter nutzt eine sinusförmige Referenzsignal bei 50 Hz zur adaptiven Störunterdrückung:

**Updateformel:**
```
w(n+1) = α·w(n) + μ·e(n)·x(n) / (ε + E[x²(n)])
```

Wobei:
- `w(n)`: Filterkoeffizienten
- `α`: Leakage-Faktor (0.999)
- `μ`: Schrittweite (0.05)
- `e(n)`: Fehlersignal
- `x(n)`: Referenzsignal
- `E[x²(n)]`: Geglättete Energie

## 📈 Performance

### DSP-Board: FM4-176-S6E2CC-ETH

- **Operationen/Sample:** ~297 (M=24)
- **Speicherbedarf:**
  - Filterkoeffizienten: 24 × 4 Byte = 96 Byte
  - Zustandsvariablen: ~16 Byte
  - Gesamt: ~112 Byte RAM

- **Latenz:**
  - Block=64: 1.33 ms @ 48 kHz
  - Block=128: 2.67 ms @ 48 kHz


