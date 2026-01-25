## Audio exportieren

```python
def export_audio(self):
    """Exportiert das aktuelle Audio (mit Rauschen falls aktiviert) als WAV"""
    if self.audio_data is None:
        return  # Nichts zu exportieren

    # Dateiname vorschlagen (abhängig davon, ob eine Datei geladen ist und ob Rauschen aktiv ist)
    if self.audio_file:
        base_name = os.path.splitext(os.path.basename(self.audio_file))
        if self.noise_enabled.get():
            suggested_name = f"{base_name}_mit_rauschen.wav"
        else:
            suggested_name = f"{base_name}_export.wav"
    else:
        suggested_name = "audio_export.wav"

    # Speichern-Dialog öffnen
    filename = filedialog.asksaveasfilename(
        title="Audio exportieren",
        defaultextension=".wav",
        initialfile=suggested_name,
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )

    # Wenn der User nicht abbricht, speichern
    if filename:
        try:
            # Sicherheitsmaßnahme: auf [-1.0, 1.0] begrenzen (verhindert Überlauf beim Int16-Cast)
            audio_normalized = np.clip(self.audio_data, -1.0, 1.0)

            # Float [-1..1] -> PCM int16 [-32768..32767] skalieren und casten
            audio_int16 = (audio_normalized * 32767).astype(np.int16)

            # WAV schreiben (Rate + PCM-Daten)
            wav.write(filename, self.sample_rate, audio_int16)

            # UI/Logging
            self.status_label.config(text=f"Exportiert: {os.path.basename(filename)}")
            print(f"Audio erfolgreich exportiert: {filename}")

        except Exception as e:
            print(f"Fehler beim Export: {e}")
            self.status_label.config(text=f"Export-Fehler: {str(e)}")
```

Beim Export wird das aktuelle Signal (ggf. inklusive hinzugefügter Störung) auf den üblichen 16‑Bit‑PCM-Bereich gebracht und dann als WAV geschrieben.
Wichtig ist das `np.clip(..., -1.0, 1.0)`, weil `int16` nur Werte von -32768 bis +32767 darstellen kann und Übersteuerungen sonst beim Konvertieren “kaputt” quantisiert würden.

## Störung erzeugen

```python
def generate_noise(self, duration, sample_rate):
    """Generiert 50Hz Rauschen basierend auf Modell und Frequenz-Shift"""
    # Zeitachse: 0 .. duration (in Sekunden), abgetastet mit sample_rate
    t = np.arange(int(duration * sample_rate)) / sample_rate
    amplitude = self.noise_amplitude.get()

    # Basisfrequenz: 50 Hz plus optionaler Shift
    base_freq = 50.0
    if self.shift_enabled.get():
        base_freq += self.freq_shift.get()

    # Negative Frequenzen verhindern (hier nicht sinnvoll)
    if base_freq < 0:
        base_freq = 0

    if self.noise_model.get() == "simple":
        # Modell 1: reiner Sinus bei base_freq
        noise = amplitude * np.sin(2 * np.pi * base_freq * t)

    elif self.noise_model.get() == "harmonics":
        # Modell 2: base_freq plus Harmonische (2x..5x), typische Netzteil-/Gleichrichter-Artefakte
        noise = (amplitude * np.sin(2 * np.pi * base_freq * t) +
                 amplitude * 0.6 * np.sin(2 * np.pi * (2 * base_freq) * t) +
                 amplitude * 0.3 * np.sin(2 * np.pi * (3 * base_freq) * t) +
                 amplitude * 0.2 * np.sin(2 * np.pi * (4 * base_freq) * t) +
                 amplitude * 0.15 * np.sin(2 * np.pi * (5 * base_freq) * t))

    elif self.noise_model.get() == "transformer":
        # Modell 3: “Trafo-Brummen”: andere Gewichtung der Harmonischen
        noise = (amplitude * 0.4 * np.sin(2 * np.pi * base_freq * t) +
                 amplitude * np.sin(2 * np.pi * (2 * base_freq) * t) +
                 amplitude * 0.5 * np.sin(2 * np.pi * (4 * base_freq) * t) +
                 amplitude * 0.35 * np.sin(2 * np.pi * (6 * base_freq) * t) +
                 amplitude * 0.25 * np.sin(2 * np.pi * (8 * base_freq) * t))

        # Langsame Amplitudenmodulation (z.B. mechanisches “Wabern”), 2 Hz
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 2 * t)
        noise *= modulation

    return noise
```

Dieser Block synthetisiert definierte Störsignale, damit du reproduzierbar testen kannst, ob dein Filter (z.B. LMS/Notch) das Brummen wirklich unterdrückt.
Die Frequenz-Verschiebung ist wichtig, weil reale Netzstörungen oft nicht exakt 50.00 Hz sind und Filter sonst “daneben greifen” (Test auf Robustheit).

## Temp.-Datei speichern

```python
def save_playback_file(self):
    """Speichert die aktuelle audio_data als temporäre WAV-Datei für Wiedergabe"""
    # Alte temporäre Datei löschen (damit sich keine Dateien ansammeln)
    if self.playback_file and os.path.exists(self.playback_file):
        try:
            os.remove(self.playback_file)
        except:
            pass

    # Neue temporäre Datei erzeugen
    fd, self.playback_file = tempfile.mkstemp(suffix='.wav')
    os.close(fd)  # Wir brauchen nur den Pfad, nicht den offenen File-Descriptor

    # Audio auf sicheren Bereich begrenzen
    audio_normalized = np.clip(self.audio_data, -1.0, 1.0)

    # Float [-1..1] -> int16 PCM skalieren
    audio_int16 = (audio_normalized * 32767).astype(np.int16)

    # WAV-Datei schreiben (wird später von pygame abgespielt)
    wav.write(self.playback_file, self.sample_rate, audio_int16)
```

Hier wird das aktuell zu spielende Signal als temporäre WAV-Datei erzeugt und überschrieben, sobald sich das Signal ändert (z.B. Rauschen an/aus).
Wichtig ist wieder die PCM-Umwandlung, weil `wav.write` typischerweise Integer-PCM erwartet bzw. du für sauberes 16‑Bit‑PCM explizit `int16` schreibst.

## Datei öffnen

```python
def open_file(self):
    filename = filedialog.askopenfilename(
        title="WAV-Datei auswählen",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )

    if filename:
        try:
            self.audio_file = filename

            # WAV lesen: liefert sample_rate und Rohdaten (z.B. int16/int32/uint8/float)
            self.sample_rate, audio_raw = wav.read(filename)

            # Stereo -> Mono: Kanäle mitteln
            if len(audio_raw.shape) > 1:
                audio_raw = np.mean(audio_raw, axis=1)

            # Auf float32 normalisieren in einen einheitlichen Bereich [-1, 1]
            if audio_raw.dtype == np.int16:
                audio_raw = audio_raw.astype(np.float32) / 32768.0
            elif audio_raw.dtype == np.int32:
                audio_raw = audio_raw.astype(np.float32) / 2147483648.0
            elif audio_raw.dtype == np.uint8:
                audio_raw = (audio_raw.astype(np.float32) - 128) / 128.0
            elif audio_raw.dtype == np.float32 or audio_raw.dtype == np.float64:
                audio_raw = audio_raw.astype(np.float32)
                max_val = np.max(np.abs(audio_raw))
                if max_val > 1.0:
                    audio_raw = audio_raw / max_val

            # Original + Arbeitskopie
            self.audio_data_original = audio_raw
            self.audio_data = audio_raw.copy()

            # Länge / UI vorbereiten
            self.audio_length = len(self.audio_data) / self.sample_rate
            self.progress_bar.config(to=self.audio_length)
            self.current_position = 0

            # X-Achsen-Default: nicht über Nyquist
            self.x_max.set(min(8000, self.sample_rate / 2))

            # Playback-Datei und Spektrum initial erzeugen
            self.save_playback_file()
            self.update_spectrum()

            # Buttons aktivieren
            self.play_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)

            info_text = f"Geladen: {os.path.basename(filename)} | {self.sample_rate}Hz | {self.format_time(self.audio_length)}"
            self.status_label.config(text=info_text)

        except Exception as e:
            print(f"FEHLER beim Laden: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"Fehler: {str(e)}")
```

Der Block standardisiert alle WAV-Formate (int16/int32/uint8/float) auf `float32`, damit späteres Addieren von Rauschen, Normalisieren und FFT immer gleich funktioniert.
Wichtig ist auch die Begrenzung der X‑Achse auf `sample_rate / 2` (Nyquist), weil darüber keine echten Spektralanteile existieren.

## Spektrum berechnen und zeichnen

```python
def update_spectrum(self):
    """Aktualisiert das Frequenzspektrum an aktueller Position"""
    if self.audio_data is None:
        return

    # Abspielposition (Sekunden) -> Sampleindex
    sample_pos = int(self.current_position * self.sample_rate)
    window_size = 4096

    # Nur wenn wir genug Samples für das Fenster haben
    if sample_pos + window_size < len(self.audio_data):
        # Zeitfenster aus dem Signal ausschneiden
        window = self.audio_data[sample_pos:sample_pos + window_size]

        # Fensterfunktion (Hann/Hanning) reduziert Spectral Leakage vor der FFT
        fft = np.fft.rfft(window * np.hanning(len(window)))

        # Betragsspektrum und Umrechnung in dB
        magnitude = np.abs(fft)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)  # epsilon verhindert log(0)

        # Frequenzachse passend zu rfft
        freqs = np.fft.rfftfreq(len(window), 1/self.sample_rate)

        # Plot neu zeichnen
        self.ax_spectrum.clear()
        self.ax_spectrum.plot(freqs, magnitude_db, color='cyan', linewidth=1.5)

        # Plot-Styling
        self.ax_spectrum.set_xlabel("Frequenz (Hz)", color='white', fontsize=11)
        self.ax_spectrum.set_ylabel("Amplitude (dB)", color='white', fontsize=11)
        self.ax_spectrum.set_facecolor('black')
        self.ax_spectrum.tick_params(colors='white')

        # Manuelle Skalierung aus den GUI-Feldern
        try:
            self.ax_spectrum.set_xlim(self.x_min.get(), self.x_max.get())
            self.ax_spectrum.set_ylim(self.y_min.get(), self.y_max.get())
        except:
            pass

        self.ax_spectrum.grid(True, alpha=0.3, color='gray')

        # Titel mit Zeit + optionalem Hinweis auf Frequenz-Shift
        time_str = self.format_time(self.current_position)
        title_extra = ""
        if self.noise_enabled.get() and self.shift_enabled.get():
            shift_val = self.freq_shift.get()
            sign = "+" if shift_val >= 0 else ""
            title_extra = f" | Störung: 50Hz {sign}{shift_val:.1f}Hz"

        self.ax_spectrum.set_title(
            f"Frequenzspektrum bei {time_str}{title_extra}",
            color='white', fontsize=12
        )

        # Canvas aktualisieren
        self.canvas_spectrum.draw()
```

Die Hann-Fensterung ist wichtig, weil sie Spectral Leakage reduziert, also das “Verschmieren” von Energie auf benachbarte Frequenz-Bins, wenn die analysierte Frequenz nicht perfekt auf einen FFT-Bin fällt
`np.fft.rfftfreq(...)` liefert dazu passend die Frequenz-Bin-Mitten für die Real-FFT (`rfft`) und sorgt dafür, dass die X-Achse korrekt in Hz beschriftet ist.
