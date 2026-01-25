import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import scipy.io.wavfile as wav
import pygame
import os
import tempfile


class AudioSpectrogramPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Player mit Spektrum-Analyse")
        self.root.geometry("1000x800")  # Etwas höher für neue Controls
        
        # Audio-Variablen
        self.audio_file = None
        self.playback_file = None
        self.sample_rate = None
        self.audio_data = None
        self.audio_data_original = None
        self.is_playing = False
        self.current_position = 0
        self.audio_length = 0
        self.playback_start_time = 0
        self.playback_position_start = 0
        
        # Rausch-Variablen
        self.noise_enabled = tk.BooleanVar(value=False)
        self.noise_model = tk.StringVar(value="simple")
        self.noise_amplitude = tk.DoubleVar(value=0.05)
        
        # Frequenz-Shift Variablen (NEU)
        self.shift_enabled = tk.BooleanVar(value=False)
        self.freq_shift = tk.DoubleVar(value=0.0)
        
        # Achsen-Einstellungen (Nur noch manuell)
        self.x_min = tk.DoubleVar(value=0)
        self.x_max = tk.DoubleVar(value=8000)
        self.y_min = tk.DoubleVar(value=-60)
        self.y_max = tk.DoubleVar(value=60)
        
        # Pygame initialisieren mit flexibler Konfiguration
        pygame.mixer.quit()
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
        
        # Dark theme für matplotlib
        plt.style.use('dark_background')
        
        # GUI erstellen
        self.create_widgets()
        
    def create_widgets(self):
        # Top Frame für Steuerung
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # Buttons
        self.open_btn = tk.Button(control_frame, text="Datei öffnen", 
                                  command=self.open_file, width=12, height=2)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = tk.Button(control_frame, text="Play", 
                                  command=self.play_pause, width=12, height=2, 
                                  state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="Stop", 
                                 command=self.stop, width=12, height=2, 
                                 state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = tk.Button(control_frame, text="Exportieren", 
                                    command=self.export_audio, width=12, height=2,
                                    state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Status-Label mit mehr Platz
        self.status_label = tk.Label(control_frame, text="Keine Datei geladen", 
                                     font=("Arial", 9), width=40, anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Rausch-Kontrollbereich
        noise_frame = tk.LabelFrame(self.root, text="50Hz Störung-Simulation", 
                                    padx=10, pady=10)
        noise_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Checkbox zum Aktivieren
        self.noise_check = tk.Checkbutton(noise_frame, text="Rauschen aktivieren", 
                                         variable=self.noise_enabled, 
                                         command=self.update_noise)
        self.noise_check.pack(anchor=tk.W)
        
        # Rauschmodell-Auswahl
        model_frame = tk.Frame(noise_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(model_frame, text="Rauschmodell:").pack(side=tk.LEFT, padx=5)
        
        models = [
            ("Einfaches Sinus (Pure)", "simple"),
            ("Netzteil mit Harmonischen", "harmonics"),
            ("Trafo-Brummen (Magnetostriktion)", "transformer")
        ]
        
        for text, value in models:
            tk.Radiobutton(model_frame, text=text, variable=self.noise_model, 
                          value=value, command=self.update_noise).pack(side=tk.LEFT, padx=5)
        
        # Amplitude-Slider
        amp_frame = tk.Frame(noise_frame)
        amp_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(amp_frame, text="Amplitude:").pack(side=tk.LEFT, padx=5)
        self.amp_slider = tk.Scale(amp_frame, from_=0, to=0.2, resolution=0.01,
                                   orient=tk.HORIZONTAL, variable=self.noise_amplitude,
                                   command=lambda x: self.update_noise())
        self.amp_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Frequenz-Shift (NEU)
        shift_frame = tk.Frame(noise_frame)
        shift_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Checkbox für Shift
        self.shift_check = tk.Checkbutton(shift_frame, text="Frequenz-Drift aktivieren", 
                                         variable=self.shift_enabled, 
                                         command=self.update_noise)
        self.shift_check.pack(side=tk.LEFT, padx=0)
        
        tk.Label(shift_frame, text="Offset (Hz):").pack(side=tk.LEFT, padx=(15, 5))
        
        # Slider für +/- 10Hz
        self.shift_slider = tk.Scale(shift_frame, from_=-10.0, to=10.0, resolution=0.1,
                                     orient=tk.HORIZONTAL, variable=self.freq_shift,
                                     command=lambda x: self.update_noise())
        self.shift_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Achsen-Einstellungen Frame
        axis_frame = tk.LabelFrame(self.root, text="Spektrum-Achsen (Manuell)", 
                                   padx=10, pady=10)
        axis_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # X-Achse Einstellungen
        x_axis_frame = tk.Frame(axis_frame)
        x_axis_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(x_axis_frame, text="X-Achse (Frequenz in Hz):").pack(side=tk.LEFT, padx=5)
        
        tk.Label(x_axis_frame, text="Min:").pack(side=tk.LEFT, padx=(20, 2))
        self.x_min_entry = tk.Entry(x_axis_frame, textvariable=self.x_min, width=10)
        self.x_min_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(x_axis_frame, text="Max:").pack(side=tk.LEFT, padx=(20, 2))
        self.x_max_entry = tk.Entry(x_axis_frame, textvariable=self.x_max, width=10)
        self.x_max_entry.pack(side=tk.LEFT, padx=2)
        
        # Y-Achse Einstellungen
        y_axis_frame = tk.Frame(axis_frame)
        y_axis_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(y_axis_frame, text="Y-Achse (Amplitude in dB):").pack(side=tk.LEFT, padx=5)
        
        tk.Label(y_axis_frame, text="Min:").pack(side=tk.LEFT, padx=(20, 2))
        self.y_min_entry = tk.Entry(y_axis_frame, textvariable=self.y_min, width=10)
        self.y_min_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(y_axis_frame, text="Max:").pack(side=tk.LEFT, padx=(20, 2))
        self.y_max_entry = tk.Entry(y_axis_frame, textvariable=self.y_max, width=10)
        self.y_max_entry.pack(side=tk.LEFT, padx=2)
        
        # Apply Button
        apply_btn = tk.Button(axis_frame, text="Achsen anwenden", command=self.apply_axis_settings,
                             width=15)
        apply_btn.pack(pady=5)
        
        # Fortschrittsbalken
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.time_label = tk.Label(progress_frame, text="0:00 / 0:00", font=("Arial", 10))
        self.time_label.pack()
        
        self.progress_bar = ttk.Scale(progress_frame, from_=0, to=100, 
                                     orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Spektrum-Anzeige
        spectrum_container = tk.Frame(self.root)
        spectrum_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig_spectrum = Figure(figsize=(10, 5), facecolor='black')
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.ax_spectrum.set_title("Aktuelles Frequenzspektrum (nur bei Pause)", 
                                   color='white', fontsize=14)
        self.ax_spectrum.set_xlabel("Frequenz (Hz)", color='white')
        self.ax_spectrum.set_ylabel("Amplitude (dB)", color='white')
        self.ax_spectrum.set_facecolor('black')
        self.ax_spectrum.tick_params(colors='white')
        self.ax_spectrum.grid(True, alpha=0.3, color='gray')
        
        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, spectrum_container)
        self.canvas_spectrum.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar für Spektrum
        self.toolbar_spectrum = NavigationToolbar2Tk(self.canvas_spectrum, spectrum_container,
                                                      pack_toolbar=False)
        self.toolbar_spectrum.update()
        self.toolbar_spectrum.pack(side=tk.BOTTOM, fill=tk.X)
    
    def apply_axis_settings(self):
        """Wendet die Achsen-Einstellungen an"""
        if self.audio_data is not None:
            self.update_spectrum()
    
    def export_audio(self):
        """Exportiert das aktuelle Audio (mit Rauschen falls aktiviert) als WAV"""
        if self.audio_data is None:
            return
        
        # Dateiname vorschlagen
        if self.audio_file:
            base_name = os.path.splitext(os.path.basename(self.audio_file))[0]
            if self.noise_enabled.get():
                suggested_name = f"{base_name}_mit_rauschen.wav"
            else:
                suggested_name = f"{base_name}_export.wav"
        else:
            suggested_name = "audio_export.wav"
        
        # Speichern-Dialog
        filename = filedialog.asksaveasfilename(
            title="Audio exportieren",
            defaultextension=".wav",
            initialfile=suggested_name,
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Audio in int16 konvertieren
                audio_normalized = np.clip(self.audio_data, -1.0, 1.0)
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
                
                # Speichern
                wav.write(filename, self.sample_rate, audio_int16)
                
                self.status_label.config(text=f"Exportiert: {os.path.basename(filename)}")
                print(f"Audio erfolgreich exportiert: {filename}")
                
            except Exception as e:
                print(f"Fehler beim Export: {e}")
                self.status_label.config(text=f"Export-Fehler: {str(e)}")
        
    def generate_noise(self, duration, sample_rate):
        """Generiert 50Hz Rauschen basierend auf Modell und Frequenz-Shift"""
        t = np.arange(int(duration * sample_rate)) / sample_rate
        amplitude = self.noise_amplitude.get()
        
        # Basis-Frequenz bestimmen (50Hz +/- Shift)
        base_freq = 50.0
        if self.shift_enabled.get():
            base_freq += self.freq_shift.get()
            
        # Um negative Frequenzen zu vermeiden (physikalisch unsinnig hier)
        if base_freq < 0: base_freq = 0
            
        if self.noise_model.get() == "simple":
            # Modell 1: Einfaches Sinus
            noise = amplitude * np.sin(2 * np.pi * base_freq * t)
            
        elif self.noise_model.get() == "harmonics":
            # Modell 2: Basis mit Harmonischen (2x, 3x, 4x...)
            noise = (amplitude * np.sin(2 * np.pi * base_freq * t) +
                    amplitude * 0.6 * np.sin(2 * np.pi * (2 * base_freq) * t) +
                    amplitude * 0.3 * np.sin(2 * np.pi * (3 * base_freq) * t) +
                    amplitude * 0.2 * np.sin(2 * np.pi * (4 * base_freq) * t) +
                    amplitude * 0.15 * np.sin(2 * np.pi * (5 * base_freq) * t))
            
        elif self.noise_model.get() == "transformer":
            # Modell 3: Trafo-Brummen (Harmonische + Modulation)
            noise = (amplitude * 0.4 * np.sin(2 * np.pi * base_freq * t) +
                    amplitude * np.sin(2 * np.pi * (2 * base_freq) * t) +
                    amplitude * 0.5 * np.sin(2 * np.pi * (4 * base_freq) * t) +
                    amplitude * 0.35 * np.sin(2 * np.pi * (6 * base_freq) * t) +
                    amplitude * 0.25 * np.sin(2 * np.pi * (8 * base_freq) * t))
            
            # Modulation bleibt bei festen 2Hz, da mechanisch bedingt
            modulation = 1 + 0.1 * np.sin(2 * np.pi * 2 * t)
            noise *= modulation
            
        return noise
    
    def save_playback_file(self):
        """Speichert die aktuelle audio_data als temporäre WAV-Datei für Wiedergabe"""
        # Alte temporäre Datei löschen
        if self.playback_file and os.path.exists(self.playback_file):
            try:
                os.remove(self.playback_file)
            except:
                pass
        
        # Neue temporäre Datei erstellen
        fd, self.playback_file = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        # Audio sicher in int16 konvertieren
        audio_normalized = np.clip(self.audio_data, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # Mit ursprünglicher Sample-Rate speichern
        wav.write(self.playback_file, self.sample_rate, audio_int16)
    
    def update_noise(self):
        """Aktualisiert das Audio mit oder ohne Rauschen"""
        if self.audio_data_original is None:
            return
        
        was_playing = self.is_playing
        if was_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
        
        if self.noise_enabled.get():
            duration = len(self.audio_data_original) / self.sample_rate
            noise = self.generate_noise(duration, self.sample_rate)
            self.audio_data = self.audio_data_original + noise[:len(self.audio_data_original)]
            
            # Normalisieren um Clipping zu vermeiden
            max_val = np.max(np.abs(self.audio_data))
            if max_val > 1.0:
                self.audio_data = self.audio_data / max_val
        else:
            self.audio_data = self.audio_data_original.copy()
        
        # Playback-Datei neu erstellen mit aktuellem audio_data
        self.save_playback_file()
        
        # Spektrum aktualisieren wenn nicht am Spielen
        if not was_playing:
            self.update_spectrum()
        
        # Wenn vorher abgespielt wurde, fortsetzen
        if was_playing:
            self.play_pause()
            
    def open_file(self):
        filename = filedialog.askopenfilename(
            title="WAV-Datei auswählen",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.audio_file = filename
                self.sample_rate, audio_raw = wav.read(filename)
                
                # Mono-Conversion falls Stereo
                if len(audio_raw.shape) > 1:
                    audio_raw = np.mean(audio_raw, axis=1)
                
                # Normalisieren auf float32 im Bereich [-1.0, 1.0]
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
                
                self.audio_data_original = audio_raw
                self.audio_data = audio_raw.copy()
                
                self.audio_length = len(self.audio_data) / self.sample_rate
                self.progress_bar.config(to=self.audio_length)
                self.current_position = 0
                
                # Standardwerte für X-Achse basierend auf Sample-Rate setzen
                self.x_max.set(min(8000, self.sample_rate / 2))
                
                # Playback-Datei erstellen
                self.save_playback_file()
                
                # Initial Spektrum anzeigen
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
            
    def play_pause(self):
        if not self.playback_file:
            return
            
        if self.is_playing:
            # Pausieren - NUR HIER wird Position aktualisiert
            pygame.mixer.music.pause()
            self.is_playing = False
            self.play_btn.config(text="Play")
            
            # Position berechnen und aktualisieren
            import time
            elapsed = time.time() - self.playback_start_time
            self.current_position = self.playback_position_start + elapsed
            
            # Auf Audio-Länge begrenzen
            if self.current_position > self.audio_length:
                self.current_position = self.audio_length
            
            # UI aktualisieren
            self.progress_bar.set(self.current_position)
            self.time_label.config(
                text=f"{self.format_time(self.current_position)} / {self.format_time(self.audio_length)}"
            )
            
            # Spektrum beim Pausieren aktualisieren
            self.update_spectrum()
        else:
            # Abspielen - KEINE Position-Updates während Wiedergabe
            try:
                import time
                self.playback_start_time = time.time()
                self.playback_position_start = self.current_position
                
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.unpause()
                else:
                    pygame.mixer.music.load(self.playback_file)
                    pygame.mixer.music.play(start=self.current_position)
                
                self.is_playing = True
                self.play_btn.config(text="⏸ Pause")
                
            except Exception as e:
                print(f"FEHLER bei Wiedergabe: {e}")
                self.status_label.config(text=f"Wiedergabe-Fehler: {str(e)}")
            
    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.current_position = 0
        self.progress_bar.set(0)
        self.play_btn.config(text="Play")
        self.time_label.config(text=f"0:00 / {self.format_time(self.audio_length)}")
        self.update_spectrum()
        
    def on_slider_move(self, value):
        if not self.playback_file:
            return
        
        self.current_position = float(value)
        
        self.time_label.config(
            text=f"{self.format_time(self.current_position)} / {self.format_time(self.audio_length)}"
        )
        
        # Spektrum nur aktualisieren wenn pausiert
        if not self.is_playing:
            self.update_spectrum()
        
        if self.is_playing:
            import time
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.playback_file)
            pygame.mixer.music.play(start=self.current_position)
            # Reset der Timer für Position-Tracking
            self.playback_start_time = time.time()
            self.playback_position_start = self.current_position
            
    def update_spectrum(self):
        """Aktualisiert das Frequenzspektrum an aktueller Position"""
        if self.audio_data is None:
            return
            
        sample_pos = int(self.current_position * self.sample_rate)
        window_size = 4096
        
        if sample_pos + window_size < len(self.audio_data):
            window = self.audio_data[sample_pos:sample_pos + window_size]
            
            # FFT berechnen mit Hanning-Fenster
            fft = np.fft.rfft(window * np.hanning(len(window)))
            magnitude = np.abs(fft)
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            
            freqs = np.fft.rfftfreq(len(window), 1/self.sample_rate)
            
            # Plotten
            self.ax_spectrum.clear()
            self.ax_spectrum.plot(freqs, magnitude_db, color='cyan', linewidth=1.5)
            self.ax_spectrum.set_xlabel("Frequenz (Hz)", color='white', fontsize=11)
            self.ax_spectrum.set_ylabel("Amplitude (dB)", color='white', fontsize=11)
            self.ax_spectrum.set_facecolor('black')
            self.ax_spectrum.tick_params(colors='white')
            
            # Manuelle Skalierung
            try:
                self.ax_spectrum.set_xlim(self.x_min.get(), self.x_max.get())
                self.ax_spectrum.set_ylim(self.y_min.get(), self.y_max.get())
            except:
                pass  # Falls ungültige Werte eingegeben wurden
            
            self.ax_spectrum.grid(True, alpha=0.3, color='gray')
            
            # Titel mit Zeitstempel
            time_str = self.format_time(self.current_position)
            
            # Info über Shift im Titel anzeigen falls aktiv
            title_extra = ""
            if self.noise_enabled.get() and self.shift_enabled.get():
                shift_val = self.freq_shift.get()
                sign = "+" if shift_val >= 0 else ""
                title_extra = f" | Störung: 50Hz {sign}{shift_val:.1f}Hz"
                
            self.ax_spectrum.set_title(f"Frequenzspektrum bei {time_str}{title_extra}", 
                                       color='white', fontsize=12)
            
            self.canvas_spectrum.draw()
            
    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
        
    def on_closing(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        # Temporäre Datei löschen
        if self.playback_file and os.path.exists(self.playback_file):
            try:
                os.remove(self.playback_file)
            except:
                pass
        
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioSpectrogramPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
