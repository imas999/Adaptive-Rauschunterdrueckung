import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy.io.wavfile as wav
import sounddevice as sd
import queue
import threading
import time

class RealtimeDSPAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Echtzeit DSP-Analyse mit Rauschsimulation")
        self.root.geometry("1400x900")
        
        # Audio-Variablen
        self.audio_file = None
        self.original_data = None
        self.distorted_data = None
        self.sample_rate = None
        self.is_playing = False
        self.playback_position = 0  # Aktuelle Position in Samples
        self.playback_start_time = 0
        
        # Rausch-Variablen
        self.noise_enabled = tk.BooleanVar(value=False)
        self.noise_model = tk.StringVar(value="simple")
        self.noise_amplitude = tk.DoubleVar(value=0.05)
        
        # Aufnahme-Variablen
        self.is_recording = False
        self.recorded_buffer = []
        self.audio_queue = queue.Queue()
        self.delay_compensation = tk.DoubleVar(value=0.0)
        
        # Streams
        self.output_stream = None
        self.input_stream = None
        self.stop_flag = False
        
        # Input-Device
        self.input_device = tk.IntVar(value=sd.default.device[0])
        
        # Sweep-Variablen
        self.sweep_active = False
        
        # FFT-Update
        self.update_interval = 200  # ms
        self.fft_update_running = False
        
        # Dark theme
        plt.style.use('dark_background')
        
        # GUI erstellen
        self.create_widgets()
        self.update_device_list()
        
    def create_widgets(self):
        # === CONTROL PANEL ===
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # Datei-Laden Frame
        file_frame = tk.LabelFrame(control_frame, text="Audio-Datei", padx=10, pady=10)
        file_frame.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(file_frame, text="Keine Datei", width=30, anchor='w', relief=tk.SUNKEN)
        self.file_label.pack()
        tk.Button(file_frame, text="Laden", command=self.load_file, width=12).pack(pady=5)
        
        # Rausch-Frame
        noise_frame = tk.LabelFrame(control_frame, text="Störsignal", padx=10, pady=10)
        noise_frame.pack(side=tk.LEFT, padx=5)
        
        self.noise_check = tk.Checkbutton(noise_frame, text="Aktivieren", 
                                         variable=self.noise_enabled,
                                         command=self.update_distortion)
        self.noise_check.pack()
        
        models = [("50Hz Sinus", "simple"), ("Netzteil", "harmonics"), ("Trafo", "transformer")]
        for text, value in models:
            tk.Radiobutton(noise_frame, text=text, variable=self.noise_model,
                          value=value, command=self.update_distortion).pack(anchor='w')
        
        tk.Label(noise_frame, text="Amplitude:").pack()
        tk.Scale(noise_frame, from_=0, to=0.2, resolution=0.01, orient=tk.HORIZONTAL,
                variable=self.noise_amplitude, command=lambda x: self.update_distortion()).pack()
        
        # Input-Device Frame
        device_frame = tk.LabelFrame(control_frame, text="Aufnahme-Eingang", padx=10, pady=10)
        device_frame.pack(side=tk.LEFT, padx=5)
        
        self.device_listbox = tk.Listbox(device_frame, height=6, width=30)
        self.device_listbox.pack()
        self.device_listbox.bind('<<ListboxSelect>>', self.on_device_select)
        tk.Button(device_frame, text="Aktualisieren", command=self.update_device_list, width=12).pack(pady=5)
        
        # Delay-Kompensation Frame
        delay_frame = tk.LabelFrame(control_frame, text="Delay-Kompensation", padx=10, pady=10)
        delay_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(delay_frame, text="Verzögerung (ms):").pack()
        delay_scale = tk.Scale(delay_frame, from_=0, to=500, resolution=1, orient=tk.HORIZONTAL,
                              variable=self.delay_compensation)
        delay_scale.pack()
        
        self.delay_label = tk.Label(delay_frame, textvariable=self.delay_compensation)
        self.delay_label.pack()
        
        # Sweep-Test Button
        tk.Button(delay_frame, text="🔊 Sweep-Test\n100-4000Hz", 
                 command=self.start_sweep_test, width=15, height=2).pack(pady=5)
        
        # Auto-Sync Button im Delay-Frame
        tk.Button(delay_frame, text="⚡ Auto-Sync", 
                 command=self.auto_sync_delay, width=15, bg='#444444', fg='white').pack(pady=2)
        
        # === PLAYBACK CONTROLS ===
        playback_frame = tk.LabelFrame(self.root, text="Wiedergabe & Aufnahme", padx=10, pady=10)
        playback_frame.pack(pady=10, fill=tk.X, padx=10)
        
        btn_frame = tk.Frame(playback_frame)
        btn_frame.pack()
        
        self.play_btn = tk.Button(btn_frame, text="▶ Start", command=self.start_playback_recording,
                                  width=15, height=2, state=tk.DISABLED, bg='green', fg='white')
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop", command=self.stop_playback_recording,
                                 width=15, height=2, state=tk.DISABLED, bg='red', fg='white')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(playback_frame, text="Bereit", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=5)
        
        # === FFT PLOT ===
        plot_frame = tk.LabelFrame(self.root, text="Echtzeit-Frequenzspektrum (0-4000 Hz)", 
                                   padx=10, pady=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = Figure(figsize=(12, 6), facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Frequenz (Hz)", color='white', fontsize=12)
        self.ax.set_ylabel("Amplitude (dB)", color='white', fontsize=12)
        self.ax.set_xlim(0, 2000)
        self.ax.set_ylim(-20, 40)  # BIS +20 dB
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, alpha=0.3, color='gray')
        
        # Legende vorbereiten
        self.line_original, = self.ax.plot([], [], color='cyan', linewidth=2, label='Original', alpha=0.8)
        self.line_distorted, = self.ax.plot([], [], color='yellow', linewidth=2, label='Verzerrt (abgespielt)', alpha=0.8)
        self.line_recorded, = self.ax.plot([], [], color='lime', linewidth=2, label='Aufgenommen (gefiltert)', alpha=0.8)
        
        self.ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
        
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_device_list(self):
        """Aktualisiert die Liste der verfügbaren Eingabegeräte"""
        self.device_listbox.delete(0, tk.END)
        devices = sd.query_devices()
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                self.device_listbox.insert(tk.END, f"{idx}: {device['name']}")
        
        try:
            default_idx = sd.default.device[0]
            for i in range(self.device_listbox.size()):
                if self.device_listbox.get(i).startswith(f"{default_idx}:"):
                    self.device_listbox.selection_set(i)
                    self.input_device.set(default_idx)
                    break
        except:
            pass
    
    def on_device_select(self, event):
        """Callback für Device-Auswahl"""
        selection = self.device_listbox.curselection()
        if selection:
            text = self.device_listbox.get(selection[0])
            device_id = int(text.split(':')[0])
            self.input_device.set(device_id)
            print(f"Input-Device gewählt: {device_id}")
    
    def load_file(self):
        """Lädt eine WAV-Datei"""
        filename = filedialog.askopenfilename(
            title="WAV-Datei auswählen",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.sample_rate, audio_raw = wav.read(filename)
                
                # Mono-Conversion
                if len(audio_raw.shape) > 1:
                    audio_raw = np.mean(audio_raw, axis=1)
                
                # Normalisieren
                if audio_raw.dtype == np.int16:
                    audio_raw = audio_raw.astype(np.float32) / 32768.0
                elif audio_raw.dtype == np.int32:
                    audio_raw = audio_raw.astype(np.float32) / 2147483648.0
                elif audio_raw.dtype == np.float32 or audio_raw.dtype == np.float64:
                    audio_raw = audio_raw.astype(np.float32)
                    max_val = np.max(np.abs(audio_raw))
                    if max_val > 1.0:
                        audio_raw = audio_raw / max_val
                
                self.original_data = audio_raw
                self.audio_file = filename
                self.file_label.config(text=f"{filename.split('/')[-1][:30]}...")
                
                self.update_distortion()
                self.play_btn.config(state=tk.NORMAL)
                
                print(f"Geladen: {self.sample_rate}Hz, {len(self.original_data)} samples, Dauer: {len(self.original_data)/self.sample_rate:.2f}s")
                
            except Exception as e:
                print(f"Fehler beim Laden: {e}")
                import traceback
                traceback.print_exc()
                self.file_label.config(text="Fehler beim Laden!")
    
    def generate_noise(self, duration, sample_rate):
        """Generiert 50Hz Rauschen"""
        t = np.arange(int(duration * sample_rate)) / sample_rate
        amplitude = self.noise_amplitude.get()
        
        if self.noise_model.get() == "simple":
            noise = amplitude * np.sin(2 * np.pi * 50 * t)
        elif self.noise_model.get() == "harmonics":
            noise = (amplitude * np.sin(2 * np.pi * 50 * t) +
                    amplitude * 0.6 * np.sin(2 * np.pi * 100 * t) +
                    amplitude * 0.3 * np.sin(2 * np.pi * 150 * t) +
                    amplitude * 0.2 * np.sin(2 * np.pi * 200 * t) +
                    amplitude * 0.15 * np.sin(2 * np.pi * 250 * t))
        elif self.noise_model.get() == "transformer":
            noise = (amplitude * 0.4 * np.sin(2 * np.pi * 50 * t) +
                    amplitude * np.sin(2 * np.pi * 100 * t) +
                    amplitude * 0.5 * np.sin(2 * np.pi * 200 * t) +
                    amplitude * 0.35 * np.sin(2 * np.pi * 300 * t) +
                    amplitude * 0.25 * np.sin(2 * np.pi * 400 * t))
            modulation = 1 + 0.1 * np.sin(2 * np.pi * 2 * t)
            noise *= modulation
        
        return noise
    
    def update_distortion(self):
        """Aktualisiert das verzerrte Signal"""
        if self.original_data is None:
            return
        
        if self.noise_enabled.get():
            duration = len(self.original_data) / self.sample_rate
            noise = self.generate_noise(duration, self.sample_rate)
            self.distorted_data = self.original_data + noise[:len(self.original_data)]
            
            max_val = np.max(np.abs(self.distorted_data))
            if max_val > 1.0:
                self.distorted_data = self.distorted_data / max_val
        else:
            self.distorted_data = self.original_data.copy()
        
        print(f"Distorted signal updated: {len(self.distorted_data)} samples")
    
    def generate_sweep(self, duration=6.0, f_start=100, f_stop=4000):
        """Generiert Frequency-Sweep"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        half = len(t) // 2
        t1 = t[:half]
        phase1 = 2 * np.pi * (f_start * t1 + (f_stop - f_start) / (duration) * t1**2)
        
        t2 = t[half:] - t[half]
        phase2 = 2 * np.pi * (f_stop * t2 + (f_start - f_stop) / (duration) * t2**2)
        
        sweep = np.concatenate([np.sin(phase1), np.sin(phase2)])
        return sweep * 0.5
    
    def auto_sync_delay(self):
        """Berechnet automatisch das Delay mittels Kreuzkorrelation"""
        if not self.is_recording or len(self.recorded_buffer) < self.sample_rate:
            print("Zu wenig Daten für Sync. Bitte erst Start drücken und kurz warten.")
            return

        print("Starte Auto-Sync...")
        
        # 1. Datenvorbereitung (wir nehmen max. 1 Sekunde für die Berechnung)
        analyze_duration = 1.0  # Sekunden
        analyze_samples = int(analyze_duration * self.sample_rate)
        
        # Holen der aufgenommenen Daten (das Ende des Buffers)
        if len(self.recorded_buffer) > analyze_samples:
            rec_chunk = np.array(self.recorded_buffer[-analyze_samples:])
        else:
            rec_chunk = np.array(self.recorded_buffer)
            analyze_samples = len(rec_chunk)

        # Holen der Original-Daten (entsprechend der aktuellen Playback-Position ZURÜCK)
        # Wir müssen weit genug zurückschauen, um das Delay zu finden.
        # Wir nehmen das Original-Segment, das "gerade eben" bis "vor 1s" abgespielt wurde.
        if self.playback_position > analyze_samples:
            # Wir nehmen ein etwas größeres Fenster vom Original, um Versatz zu finden
            # Sagen wir, wir suchen ein Delay bis zu 500ms
            max_delay_search = int(0.5 * self.sample_rate) 
            
            start_pos = self.playback_position - analyze_samples - max_delay_search
            end_pos = self.playback_position
            
            if start_pos < 0: start_pos = 0
            
            ref_chunk = self.distorted_data[start_pos:end_pos]
        else:
            print("Playback noch nicht weit genug fortgeschritten.")
            return

        # 2. Normalisierung (Wichtig für gute Korrelation)
        rec_chunk = rec_chunk - np.mean(rec_chunk)
        ref_chunk = ref_chunk - np.mean(ref_chunk)
        
        if np.std(rec_chunk) > 0: rec_chunk /= np.std(rec_chunk)
        if np.std(ref_chunk) > 0: ref_chunk /= np.std(ref_chunk)

        # 3. Kreuzkorrelation berechnen
        # mode='valid' schiebt das kürzere Signal (rec) über das längere (ref)
        correlation = np.correlate(ref_chunk, rec_chunk, mode='valid')
        
        # 4. Maximum finden
        best_idx = np.argmax(correlation)
        max_corr_val = correlation[best_idx]
        
        # Der Index im Correlation-Array entspricht dem Offset im Suchfenster
        # Da wir ref_chunk = rec_chunk + max_delay Samples haben:
        # Das gefundene Delay ist: (Länge Suchbereich - gefundener Index) ? 
        # Einfacher: Das Delay ist der Abstand vom "Jetzt"
        
        # Bei mode='valid' mit len(ref) > len(rec):
        # Index 0 bedeutet: rec passt ganz am Anfang von ref (altes Signal -> großes Delay)
        # Index max bedeutet: rec passt ganz am Ende von ref (neues Signal -> kleines Delay)
        
        # Das Delay in Samples ist: (Länge des Such-Überhangs) - best_idx
        # ref_chunk ist länger als rec_chunk um 'max_delay_search'
        
        calculated_delay_samples = max_delay_search - best_idx
        
        # In Millisekunden umrechnen
        calculated_delay_ms = (calculated_delay_samples / self.sample_rate) * 1000
        
        print(f"Sync Result: Peak @ Idx {best_idx}, Val {max_corr_val:.2f}, Delay {calculated_delay_ms:.2f}ms")

        # 5. Plausibilitäts-Check und Anwenden
        if max_corr_val > 10.0: # Arbiträrer Threshold, hängt von chunk-Länge ab
            # Begrenzen auf Slider-Range
            final_delay = max(0, min(500, calculated_delay_ms))
            self.delay_compensation.set(final_delay)
            self.status_label.config(text=f"Sync: {final_delay:.1f}ms", fg='green')
        else:
            self.status_label.config(text="Sync fehlgeschlagen (Signal zu schwach?)", fg='orange')


    def start_sweep_test(self):
        """Startet Sweep-Test"""
        if self.is_playing:
            return
        
        self.sweep_active = True
        sr = self.sample_rate if self.sample_rate else 44100
        self.sample_rate = sr
        
        self.original_data = self.generate_sweep(duration=6.0)
        self.distorted_data = self.original_data.copy()
        
        self.file_label.config(text="Sweep-Test 100-4000Hz (6s)")
        self.play_btn.config(state=tk.NORMAL)
        self.noise_check.config(state=tk.DISABLED)
        print(f"Sweep generated: {len(self.original_data)} samples @ {self.sample_rate}Hz")
    
    def audio_input_callback(self, indata, frames, time_info, status):
        """Callback für Audio-Aufnahme"""
        if status:
            print(f"Input Status: {status}")
        self.audio_queue.put(indata.copy())
    
    def audio_output_callback(self, outdata, frames, time_info, status):
        """Callback für Audio-Wiedergabe"""
        if status:
            print(f"Output Status: {status}")
        
        if self.playback_position >= len(self.distorted_data):
            outdata.fill(0)
            raise sd.CallbackStop()
        
        remaining = len(self.distorted_data) - self.playback_position
        if remaining < frames:
            outdata[:remaining, 0] = self.distorted_data[self.playback_position:]
            outdata[remaining:, 0] = 0
            self.playback_position += remaining
        else:
            outdata[:, 0] = self.distorted_data[self.playback_position:self.playback_position + frames]
            self.playback_position += frames
    
    def start_playback_recording(self):
        """Startet Wiedergabe und Aufnahme"""
        if self.distorted_data is None:
            return
        
        self.is_playing = True
        self.is_recording = True
        self.stop_flag = False
        self.playback_position = 0
        self.recorded_buffer = []
        self.playback_start_time = time.time()
        
        # Leere Queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        try:
            # Output-Stream (Wiedergabe)
            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_output_callback,
                blocksize=1024
            )
            
            # Input-Stream (Aufnahme)
            self.input_stream = sd.InputStream(
                device=self.input_device.get(),
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_input_callback,
                blocksize=1024
            )
            
            self.output_stream.start()
            self.input_stream.start()
            
            print(f"Playback & Recording started @ {self.sample_rate}Hz")
            
            # Start FFT-Update
            self.fft_update_running = True
            self.update_fft_display()
            
            # UI Update
            self.play_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="⏺ Spielt ab & nimmt auf...", fg='red')
            
        except Exception as e:
            print(f"Fehler beim Start: {e}")
            import traceback
            traceback.print_exc()
            self.stop_playback_recording()
    
    def stop_playback_recording(self):
        """Stoppt Wiedergabe und Aufnahme"""
        self.stop_flag = True
        self.is_playing = False
        self.is_recording = False
        self.fft_update_running = False
        
        try:
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
        except:
            pass
        
        # UI Update
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Gestoppt", fg='black')
        
        if self.sweep_active:
            self.sweep_active = False
            self.noise_check.config(state=tk.NORMAL)
        
        print("Stopped")
    
    def compute_fft(self, data, sample_rate, window_size=4096):
        """Berechnet FFT effizient"""
        if len(data) < window_size:
            return None, None
        
        # FFT mit Hanning-Fenster
        windowed = data * np.hanning(window_size)
        fft_data = np.fft.rfft(windowed)
        magnitude = np.abs(fft_data)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        freqs = np.fft.rfftfreq(window_size, 1/sample_rate)
        
        # Nur 0-4000 Hz
        mask = freqs <= 4000
        return freqs[mask], magnitude_db[mask]
    
    def update_fft_display(self):
        """Aktualisiert FFT-Display"""
        if not self.fft_update_running:
            return

        try:
            window_size = 4096
            
            # Berechne Delay in Samples
            delay_samples = int(self.delay_compensation.get() * self.sample_rate / 1000)
            
            # === KORREKTUR ===
            # Wir verschieben die Position für Original/Verzerrt nach HINTEN (in die Vergangenheit),
            # damit sie synchron zu der verzögerten Aufnahme angezeigt werden.
            display_position = self.playback_position - delay_samples
            
            # 1. Original-FFT (Verzögert angezeigt)
            if self.original_data is not None and display_position > 0:
                start_pos = max(0, display_position - window_size)
                # Sicherstellen, dass wir nicht über das Ende lesen
                if start_pos + window_size <= len(self.original_data):
                    orig_window = self.original_data[start_pos:start_pos + window_size]
                    freqs_orig, mag_orig = self.compute_fft(orig_window, self.sample_rate)
                    
                    if freqs_orig is not None:
                        self.line_original.set_data(freqs_orig, mag_orig)
            
            # 2. Verzerrtes Signal FFT (Verzögert angezeigt)
            if self.distorted_data is not None and display_position > 0:
                start_pos = max(0, display_position - window_size)
                
                if start_pos + window_size <= len(self.distorted_data):
                    dist_window = self.distorted_data[start_pos:start_pos + window_size]
                    freqs_dist, mag_dist = self.compute_fft(dist_window, self.sample_rate)
                    
                    if freqs_dist is not None:
                        self.line_distorted.set_data(freqs_dist, mag_dist)
            
            # 3. Aufgenommenes Signal FFT (Immer das neueste!)
            # Hier nehmen wir einfach immer das Ende des Buffers (das "Jetzt" des Mikrofons)
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                self.recorded_buffer.extend(chunk.flatten())
            
            if len(self.recorded_buffer) >= window_size:
                # Nimm einfach die allerletzten Daten (das "Frischeste")
                rec_window = np.array(self.recorded_buffer[-window_size:])
                
                freqs_rec, mag_rec = self.compute_fft(rec_window, self.sample_rate)
                
                if freqs_rec is not None:
                    self.line_recorded.set_data(freqs_rec, mag_rec)
            
            # Canvas aktualisieren
            self.canvas.draw_idle()
            
            # Check ob Wiedergabe beendet
            if self.playback_position >= len(self.distorted_data):
                print("Playback finished")
                self.root.after(500, self.stop_playback_recording)
                return
            
        except Exception as e:
            print(f"FFT-Update Fehler: {e}")
            import traceback
            traceback.print_exc()
        
        # Nächstes Update planen
        if self.fft_update_running:
            self.root.after(self.update_interval, self.update_fft_display)
    
    def on_closing(self):
        """Cleanup"""
        self.stop_playback_recording()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealtimeDSPAnalyzer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
