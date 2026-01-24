# Audio Player mit Spektrum-Analyse und Störsignal-Simulation

Diese Anwendung ist ein Python-basiertes GUI-Tool zur Analyse von Audiodateien. Sie ermöglicht das Abspielen von WAV-Dateien, die Echtzeit-Visualisierung des Frequenzspektrums und die Simulation von verschiedenen Netzbrumm-Störsignalen (50Hz), um Filteralgorithmen oder Audio-Equipment zu testen.

## 1. Überblick \& Abhängigkeiten

Das Programm verbindet eine grafische Oberfläche (`tkinter`) mit numerischer Signalverarbeitung (`numpy`) und Audio-Wiedergabe (`pygame`).

### Verwendete Bibliotheken

- **tkinter**: Für die grafische Benutzeroberfläche (Buttons, Slider, Fenster).
- **numpy**: Für schnelle mathematische Operationen (FFT, Signalgenerierung, Normalisierung).
- **matplotlib**: Zur Darstellung des Frequenzspektrums, integriert in tkinter via `FigureCanvasTkAgg`.
- **scipy.io.wavfile**: Zum Lesen und Schreiben von WAV-Dateien.
- **pygame**: Als Audio-Backend für das Abspielen der Audiodaten.

***

## 2. Funktionsweise der Klasse `AudioSpectrogramPlayer`

Die gesamte Logik ist in dieser Klasse gekapselt. Hier ist, was "unter der Haube" passiert:

### 2.1 Audio-Verarbeitung (`open_file`)

Wenn eine Datei geladen wird, führt der Code folgende Schritte durch:

1. **Import**: Lädt die WAV-Datei mittels `scipy`.
2. **Konvertierung**:
    - **Stereo zu Mono**: Falls nötig, werden Kanäle gemittelt (`np.mean`).
    - **Normalisierung**: Konvertiert alle Formate (int16, int32, uint8) in ein einheitliches `float32`-Format im Wertebereich `[-1.0, 1.0]`. Dies vereinfacht alle nachfolgenden Berechnungen erheblich.
3. **Speicherung**: Hält zwei Kopien im Speicher:
    - `self.audio_data_original`: Das unveränderte Originalsignal.
    - `self.audio_data`: Das aktuelle Signal (ggf. mit hinzugefügtem Rauschen), das abgespielt wird.

### 2.2 Störsignal-Simulation (`generate_noise`, `update_noise`)

Dies ist eine Kernfunktion zum Testen von Filtern. Der Code simuliert Netzbrummen auf drei Arten:

1. **Simple (Reiner Sinus)**:
Erzeugt eine reine 50Hz Sinuswelle.

$$
f(t) = A \cdot \sin(2\pi \cdot 50 \cdot t)
$$
2. **Harmonics (Netzteil-Simulation)**:
Addiert Obertöne (100Hz, 150Hz, etc.) mit abnehmender Amplitude hinzu. Dies simuliert das typische "Sägezahn"-artige Brummen von Gleichrichtern.
3. **Transformer (Trafo-Brummen)**:
Ein komplexes Modell, das Magnetostriktion simuliert.
    - Basis: 50Hz + starke Obertöne.
    - **Modulation**: Das gesamte Signal wird mit einer langsamen 2Hz-Schwingung amplitudenmoduliert (`1 + 0.1 * sin(...)`), was das "Wabern" großer Transformatoren nachahmt.

Das Rauschen wird additiv auf das Originalsignal gelegt. Um digitales Clipping (Werte > 1.0) zu verhindern, wird das Summensignal bei Bedarf neu normalisiert.

### 2.3 Wiedergabe-Engine (`play_pause`, `save_playback_file`)

Da `pygame` Audiodaten nicht direkt aus NumPy-Arrays streamen kann, nutzt der Code einen Trick:

1. Das aktuelle NumPy-Array (`float32`) wird in `int16` zurückkonvertiert.
2. Es wird als **temporäre WAV-Datei** (`tempfile`) auf die Festplatte geschrieben.
3. `pygame.mixer` lädt und spielt diese temporäre Datei ab.

**Synchronisation:**

- Ein Slider (`ttk.Scale`) zeigt den Fortschritt.
- Da Pygame keine präzisen "Live"-Updates an den Slider sendet, wird die Position beim *Pausieren* basierend auf der vergangenen Systemzeit (`time.time()`) berechnet und korrigiert.


### 2.4 Spektrum-Analyse (`update_spectrum`)

Die Visualisierung zeigt, welche Frequenzen im aktuellen Audio-Abschnitt dominieren.

1. **Fensterung**: Es wird ein Ausschnitt (Window) von 4096 Samples an der aktuellen Abspielposition genommen.
2. **Hanning-Fenster**: Der Ausschnitt wird mit einer Hanning-Funktion multipliziert (`np.hanning`), um "Spectral Leakage" (Verschmieren des Spektrums an den Rändern) zu reduzieren.
3. **FFT (Fast Fourier Transform)**: `np.fft.rfft` wandelt das Zeitsignal in den Frequenzbereich um.
4. **Logarithmierung**: Die Amplitude wird in Dezibel (dB) umgerechnet: $20 \cdot \log_{10}(|X|)$.

**Achsen-Skalierung:**
Der Nutzer kann zwischen "Auto-Scale" (passt sich dynamisch an die lautesten Frequenzen an) und manuellen Grenzen (z.B. um gezielt den 50Hz-Bereich zu beobachten) umschalten.

***

## 3. Benutzeroberfläche (GUI) Aufbau

Die GUI ist modular aufgebaut (`create_widgets`):

- **Control Frame (Oben)**: Standard-Player-Steuerung (Play, Stop, Datei öffnen) und Export-Funktion.
- **Rausch-Kontrollbereich (Mitte)**:
    - Checkbox zum Ein-/Ausschalten.
    - Radiobuttons für die drei Rauschmodelle.
    - Slider für die Rauschamplitude (0 bis 20% des Signals).
- **Achsen-Einstellungen**: Erlaubt Zoom in bestimmte Frequenzbereiche (z.B. 0-100Hz zur Analyse des Brummens).
- **Matplotlib Canvas (Unten)**: Einbettung des Plots in das Tkinter-Fenster inklusive der Standard-Matplotlib-Toolbar (Zoom, Pan, Save).


## 4. Export (`export_audio`)

Das bearbeitete Signal (inklusive des simulierten Brummens) kann als neue WAV-Datei gespeichert werden. Dies ist nützlich, um Testdateien für externe Hardware (wie den DSP mit dem LMS-Filter) zu generieren.

***

## 5. Anwendungsbeispiel (Workflow)

1. **Datei laden**: Nutzer öffnet eine saubere Sprachaufnahme.
2. **Störung hinzufügen**: Nutzer aktiviert "Rauschen", wählt "Trafo-Brummen" und setzt Amplitude auf 10%.
3. **Analyse**: Nutzer zoomt im Spektrum auf 0-400Hz, um die 50Hz-Spitze und ihre Obertöne zu sehen.
4. **Export**: Nutzer speichert die Datei als `testfile_brummen.wav`.
5. **DSP-Test**: Diese Datei kann nun in den C++ LMS-Filter eingespeist werden, um zu prüfen, ob er das Brummen entfernen kann.
<span style="display:none">[^1][^2]</span>

<div align="center">⁂</div>

[^1]: blockLMS.cpp

[^2]: LMS.hpp

