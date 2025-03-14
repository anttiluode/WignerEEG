#!/usr/bin/env python3
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mne
from scipy import signal
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------
# Wigner Transform
# -------------------------------
class WignerTransform:
    """Computes Wigner-Ville distributions (auto or cross) for signals."""
    @staticmethod
    def compute_wigner(signal1, signal2=None, fs=1000.0, n_freq=None):
        signal1 = np.asarray(signal1).flatten()
        if signal2 is None:
            signal2 = signal1
        else:
            signal2 = np.asarray(signal2).flatten()

        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

        analytic1 = signal.hilbert(signal1)
        analytic2 = signal.hilbert(signal2)

        if n_freq is None:
            n_freq = 1
            while n_freq < min_len:
                n_freq *= 2

        t = np.arange(min_len) / fs
        f = np.fft.fftshift(np.fft.fftfreq(n_freq, 1/fs))

        wvd = np.zeros((min_len, n_freq), dtype=complex)
        half_len = min_len // 2

        for tau in range(-half_len, half_len):
            t_indices = np.arange(max(0, -tau), min(min_len, min_len - tau))
            t_shifted = t_indices + tau
            valid = (t_shifted >= 0) & (t_shifted < min_len)
            if not np.any(valid):
                continue

            t_indices = t_indices[valid]
            t_shifted = t_shifted[valid]
            product = analytic1[t_indices] * np.conjugate(analytic2[t_shifted])

            padded = np.zeros(n_freq, dtype=complex)
            padded[:len(product)] = product
            F = np.fft.fftshift(np.fft.fft(padded))
            tau_idx = tau + half_len
            if 0 <= tau_idx < min_len:
                wvd[tau_idx] = F

        if np.array_equal(signal1, signal2):
            wvd = np.real(wvd)
        return wvd, t, f

    @staticmethod
    def compute_simple_wigner(signal1, signal2=None, fs=1000.0):
        if signal2 is None:
            signal2 = signal1

        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

        if not np.array_equal(signal1, signal2):
            sig_sum = signal1 + signal2
            sig_diff = signal1 - signal2
            auto_sum, t, f = WignerTransform.compute_simple_wigner(sig_sum, fs=fs)
            auto_diff, _, _ = WignerTransform.compute_simple_wigner(sig_diff, fs=fs)
            return (auto_sum - auto_diff)/4, t, f

        n_fft = 1
        while n_fft < min_len:
            n_fft *= 2

        win_length = min(min_len // 2, 256)
        if win_length % 2 == 0:
            win_length += 1
        window = signal.windows.hann(win_length)

        f, t, Sxx = signal.spectrogram(
            signal1,
            fs=fs,
            window=window,
            nperseg=win_length,
            noverlap=win_length-1,
            nfft=n_fft,
            detrend=False,
            return_onesided=False,
            scaling='spectrum'
        )
        Sxx = np.fft.fftshift(Sxx, axes=0)
        f = np.fft.fftshift(f)
        return Sxx.T, t, f

    @staticmethod
    def plot_wigner(wigner, t, f, ax=None, vmin=None, vmax=None, cmap='RdBu_r'):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if np.isrealobj(wigner):
            data = wigner
        else:
            data = np.real(wigner)

        if len(t) != data.shape[0]:
            t = np.linspace(0, data.shape[0], data.shape[0])
        if len(f) != data.shape[1]:
            f = np.linspace(f[0], f[-1], data.shape[1])

        extent = [t[0], t[-1], f[0], f[-1]]
        im = ax.imshow(
            data.T,
            origin='lower',
            aspect='auto',
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        return fig, im

# -------------------------------
# Neural Coherence Calculator
# -------------------------------
class NeuralCoherenceCalculator:
    """Toy 'quantum-inspired' coherence: Wigner negativity + PLV."""
    def __init__(self):
        self.wigner = WignerTransform()

    def compute_coherence_index(self, signals, fs=1000.0):
        channels = list(signals.keys())
        if len(channels) < 2:
            return 0.0, {}

        negativity_sum = 0.0
        plv_sum = 0.0
        pair_count = 0
        details = {}

        for i in range(len(channels)):
            for j in range(i+1, len(channels)):
                ch1 = channels[i]
                ch2 = channels[j]
                sig1 = signals[ch1]
                sig2 = signals[ch2]

                try:
                    wvd, _, _ = self.wigner.compute_wigner(sig1, sig2, fs=fs)
                except Exception as e:
                    logging.warning(f"Wigner failed for {ch1}-{ch2}: {e}")
                    wvd, _, _ = self.wigner.compute_simple_wigner(sig1, sig2, fs=fs)

                neg_mask = (wvd < 0)
                neg_sum = np.sum(np.abs(wvd[neg_mask]))
                total_sum = np.sum(np.abs(wvd))
                negativity = neg_sum / total_sum if total_sum > 1e-9 else 0

                plv = self.phase_locking_value(sig1, sig2)

                negativity_sum += negativity
                plv_sum += plv
                pair_count += 1

                details[f"{ch1}-{ch2}"] = {"negativity": negativity, "plv": plv}

        if pair_count > 0:
            avg_neg = negativity_sum / pair_count
            avg_plv = plv_sum / pair_count
            coherence_index = 0.5 * avg_neg + 0.5 * avg_plv
        else:
            avg_neg = 0
            avg_plv = 0
            coherence_index = 0

        details["avg_negativity"] = avg_neg
        details["avg_plv"] = avg_plv
        details["coherence_index"] = coherence_index
        return coherence_index, details

    def phase_locking_value(self, sig1, sig2):
        analytic1 = signal.hilbert(sig1)
        analytic2 = signal.hilbert(sig2)
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        phase_diff = phase1 - phase2
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        return plv

# -------------------------------
# EEG Processor
# -------------------------------
class EEGProcessor:
    """Loads EEG from EDF with MNE, extracts data by time window."""
    def __init__(self):
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        self.window_size = 2.0
        self.channel_types = {}

    def load_file(self, filepath: str) -> bool:
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            
            # Try to determine channel types
            self.channel_types = {}
            for ch_name in self.raw.ch_names:
                if "EEG" in ch_name.upper():
                    self.channel_types[ch_name] = "EEG"
                elif any(sensor in ch_name.upper() for sensor in ["TEMP", "TEMPERATURE"]):
                    self.channel_types[ch_name] = "Temperature"
                elif any(sensor in ch_name.upper() for sensor in ["EMG", "MUSC"]):
                    self.channel_types[ch_name] = "EMG"
                elif any(sensor in ch_name.upper() for sensor in ["EOG", "EYE"]):
                    self.channel_types[ch_name] = "EOG"
                elif any(sensor in ch_name.upper() for sensor in ["ECG", "HEART", "CARD"]):
                    self.channel_types[ch_name] = "ECG"
                elif "STRENGTH" in ch_name.upper() or "SIGNAL" in ch_name.upper():
                    self.channel_types[ch_name] = "Signal Strength"
                else:
                    self.channel_types[ch_name] = "Other"
            
            logging.info(f"Loaded EEG file: {filepath}")
            logging.info(f"Sampling frequency: {self.sfreq} Hz, Duration: {self.duration:.2f} s")
            logging.info(f"Channels: {self.raw.ch_names}")
            logging.info(f"Channel types: {self.channel_types}")
            return True
        except Exception as e:
            logging.error(f"Failed to load EEG file: {e}")
            return False

    def get_channels(self):
        if self.raw:
            return self.raw.ch_names
        return []
    
    def get_channel_types(self):
        if not self.channel_types:
            return set(["Unknown"])
        return set(self.channel_types.values())

    def get_channels_by_type(self, channel_type):
        return [ch for ch, tp in self.channel_types.items() if tp == channel_type]

    def get_data(self, channels, start_time: float) -> dict:
        if self.raw is None:
            return {}
        try:
            start_sample = int(start_time * self.sfreq)
            samples_needed = int(self.window_size * self.sfreq)
            end_sample = start_sample + samples_needed
            if end_sample > self.raw.n_times:
                end_sample = int(self.raw.n_times)

            data_dict = {}
            for channel in channels:
                if isinstance(channel, int):
                    ch_name = self.raw.ch_names[channel]
                    ch_idx = channel
                else:
                    ch_name = channel
                    ch_idx = self.raw.ch_names.index(channel)

                data, _ = self.raw[ch_idx, start_sample:end_sample]
                data_dict[ch_name] = data.flatten()

            return data_dict
        except Exception as e:
            logging.error(f"Error getting EEG data: {e}")
            return {}

# -------------------------------
# Channel Selection Dialog
# -------------------------------
class ChannelSelectorDialog:
    def __init__(self, parent, channel_types, initial_selection=None):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Channel Type Selection")
        self.dialog.geometry("400x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.channel_types = channel_types
        self.checkboxes = {}
        self.vars = {}
        self.result = initial_selection or set()
        
        # Create UI
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Select channel types to include in coherence calculation:").pack(anchor="w", pady=(0, 10))
        
        # Scrollable frame for checkboxes
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create a checkbox for each channel type
        for ch_type in sorted(channel_types):
            var = tk.BooleanVar(value=ch_type in self.result)
            self.vars[ch_type] = var
            checkbox = ttk.Checkbutton(
                scrollable_frame, 
                text=ch_type,
                variable=var,
                command=lambda t=ch_type: self.toggle_selection(t)
            )
            checkbox.pack(anchor="w", pady=2)
            self.checkboxes[ch_type] = checkbox
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(button_frame, text="Select All", command=self.select_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="EEG Only", command=self.eeg_only).pack(side="left", padx=5)
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side="right", padx=5)
        
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def toggle_selection(self, channel_type):
        if self.vars[channel_type].get():
            self.result.add(channel_type)
        else:
            self.result.discard(channel_type)
    
    def select_all(self):
        for ch_type, var in self.vars.items():
            var.set(True)
            self.result.add(ch_type)
    
    def deselect_all(self):
        for ch_type, var in self.vars.items():
            var.set(False)
        self.result.clear()
    
    def eeg_only(self):
        self.deselect_all()
        for ch_type in self.channel_types:
            if ch_type == "EEG":
                self.vars[ch_type].set(True)
                self.result.add(ch_type)
    
    def on_ok(self):
        self.dialog.destroy()
    
    def on_cancel(self):
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        self.dialog.wait_window()
        return self.result

# -------------------------------
# Main EEG Wigner Viewer
# -------------------------------
class EEGWignerViewer:
    """
    A GUI that:
      - Loads an EEG file
      - Displays EEG signal + Wigner transform
      - Plays EEG data (via main-thread scheduling)
      - Computes 'quantum-inspired' coherence with channel type filtering
    """
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Wigner Viewer (Quantum Decoherence Simulator)")

        self.eeg = EEGProcessor()
        self.wigner = WignerTransform()
        self.coherence_calc = NeuralCoherenceCalculator()

        self.selected_ch1 = tk.StringVar()
        self.selected_ch2 = tk.StringVar()
        self.current_time = tk.DoubleVar(value=0.0)
        self.auto_wigner = tk.BooleanVar(value=False)
        self.playing = False
        self.speed_var = tk.DoubleVar(value=1.0)

        # Selected channel types for coherence calculation
        self.selected_channel_types = set(["EEG"])

        self.last_wigner_time = 0
        self.wigner_interval = 0.5

        self.wigner_cbar = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None

        self.setup_gui()

    def setup_gui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")

        viz_frame = ttk.Frame(self.root, padding="10")
        viz_frame.grid(row=1, column=0, sticky="nsew")

        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # File load
        file_frame = ttk.LabelFrame(control_frame, text="EEG File", padding="5")
        file_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(file_frame, text="Load EEG File", command=self.load_eeg_file).grid(
            row=0, column=0, padx=5, pady=5
        )
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Channels
        ch_frame = ttk.LabelFrame(control_frame, text="Channel Selection", padding="5")
        ch_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Label(ch_frame, text="Channel 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ch1_combo = ttk.Combobox(ch_frame, textvariable=self.selected_ch1, state="readonly")
        self.ch1_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(ch_frame, text="Channel 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ch2_combo = ttk.Combobox(ch_frame, textvariable=self.selected_ch2, state="readonly")
        self.ch2_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Checkbutton(ch_frame, text="Auto-Wigner (single channel)",
                        variable=self.auto_wigner, command=self.toggle_auto_wigner
                       ).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Time nav
        time_frame = ttk.LabelFrame(control_frame, text="Time Navigation", padding="5")
        time_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)

        self.play_button = ttk.Button(time_frame, text="Play", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        ttk.Label(time_frame, text="Speed:").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        speed_combo = ttk.Combobox(time_frame, textvariable=self.speed_var,
                                   values=[0.25, 0.5, 1.0, 2.0, 4.0], width=5)
        speed_combo.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        speed_combo.current(2)

        ttk.Label(time_frame, text="Time (s):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.time_scale = ttk.Scale(time_frame, from_=0, to=100,
                                    variable=self.current_time,
                                    command=self.on_time_changed)
        self.time_scale.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        time_frame.columnconfigure(3, weight=1)

        self.time_entry = ttk.Entry(time_frame, width=10)
        self.time_entry.grid(row=1, column=4, padx=5, pady=5)
        self.time_entry.bind("<Return>", self.on_time_entry)

        # Action Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)

        ttk.Button(button_frame, text="Compute Wigner", command=self.compute_wigner).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(button_frame, text="Save Figure", command=self.save_figure).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        
        coherence_frame = ttk.Frame(button_frame)
        coherence_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(coherence_frame, text="Compute Coherence", command=self.compute_coherence).pack(
            side=tk.LEFT
        )
        ttk.Button(coherence_frame, text="Channel Types", command=self.select_channel_types).pack(
            side=tk.LEFT, padx=5
        )
        
        # Add a label to show currently selected channel types
        self.ch_types_label = ttk.Label(button_frame, text="Selected types: EEG")
        self.ch_types_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax1.set_title("EEG Signal")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")

        self.ax2.set_title("Wigner-Ville Distribution")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Frequency (Hz)")

        try:
            self.fig.tight_layout()
        except Exception as e:
            logging.warning(f"tight_layout error on init: {e}")

        self.canvas.draw()

    def select_channel_types(self):
        if not self.eeg.raw:
            messagebox.showwarning("No Data", "Please load an EEG file first")
            return
        
        channel_types = self.eeg.get_channel_types()
        selector = ChannelSelectorDialog(self.root, channel_types, self.selected_channel_types)
        result = selector.show()
        
        if result is not None:
            self.selected_channel_types = result
            self.update_channel_types_label()
    
    def update_channel_types_label(self):
        types_str = ", ".join(sorted(self.selected_channel_types)) if self.selected_channel_types else "None"
        self.ch_types_label.config(text=f"Selected types: {types_str}")

    def toggle_auto_wigner(self):
        if self.auto_wigner.get():
            self.ch2_combo.config(state="disabled")
        else:
            self.ch2_combo.config(state="readonly")

    def toggle_play(self):
        if not self.eeg.raw:
            messagebox.showwarning("No Data", "Please load an EEG file first")
            return
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.last_update_play = time.time()
            self.playback_update()

    def playback_update(self):
        if not self.playing:
            return

        current_time = time.time()
        if not hasattr(self, 'last_update_play'):
            self.last_update_play = current_time

        elapsed = current_time - self.last_update_play
        self.last_update_play = current_time

        new_time = self.current_time.get() + elapsed * self.speed_var.get()
        max_time = max(0, self.eeg.duration - self.eeg.window_size)
        if new_time >= max_time:
            new_time = 0.0
        self.current_time.set(new_time)
        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, f"{new_time:.2f}")

        self.update_signal_plot()

        if new_time - self.last_wigner_time >= self.wigner_interval:
            self.compute_wigner()
            self.last_wigner_time = new_time

        self.root.after(30, self.playback_update)

    def load_eeg_file(self):
        filepath = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if not filepath:
            return
        if self.eeg.load_file(filepath):
            self.file_label.config(text=os.path.basename(filepath))
            channels = self.eeg.get_channels()
            self.ch1_combo['values'] = channels
            self.ch2_combo['values'] = channels
            if channels:
                self.ch1_combo.current(0)
                if len(channels) > 1:
                    self.ch2_combo.current(1)

            max_time = max(0, self.eeg.duration - self.eeg.window_size)
            self.time_scale.config(to=max_time)
            self.time_scale.set(0)
            self.time_entry.delete(0, tk.END)
            self.time_entry.insert(0, "0.0")

            # Update channel types label (set to EEG by default)
            channel_types = self.eeg.get_channel_types()
            if "EEG" in channel_types:
                self.selected_channel_types = set(["EEG"])
            else:
                # If no EEG channels, select the first available type
                self.selected_channel_types = set([next(iter(channel_types))])
            self.update_channel_types_label()

            self.update_signal_plot()
        else:
            messagebox.showerror("Error", "Failed to load EEG file")

    def on_time_changed(self, event=None):
        val = self.current_time.get()
        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, f"{val:.2f}")
        self.update_signal_plot()

    def on_time_entry(self, event=None):
        try:
            val = float(self.time_entry.get())
            max_time = max(0, self.eeg.duration - self.eeg.window_size)
            if 0 <= val <= max_time:
                self.current_time.set(val)
                self.update_signal_plot()
            else:
                messagebox.showwarning("Invalid Time",
                                       f"Time must be between 0 and {max_time:.2f} seconds")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number")

    def update_signal_plot(self):
        if not self.eeg.raw:
            return
        try:
            ch1 = self.selected_ch1.get()
            ch2 = self.selected_ch2.get()
            if not ch1:
                return

            time_val = self.current_time.get()
            channels = [ch1]
            if (not self.auto_wigner.get()) and ch2:
                channels.append(ch2)

            data_dict = self.eeg.get_data(channels, time_val)
            if not data_dict:
                return

            self.ax1.clear()
            t_vals = np.arange(len(data_dict[ch1])) / self.eeg.sfreq
            self.ax1.plot(t_vals, data_dict[ch1], label=ch1)
            if (not self.auto_wigner.get()) and ch2 in data_dict:
                self.ax1.plot(t_vals, data_dict[ch2], label=ch2)
                self.ax1.legend()

            self.ax1.set_title("EEG Signal")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.set_ylabel("Amplitude")

            try:
                self.fig.tight_layout()
            except Exception as e:
                logging.warning(f"tight_layout error (signal plot): {e}")

            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error updating signal plot: {e}")

    def compute_wigner(self):
        if not self.eeg.raw:
            messagebox.showwarning("No Data", "Please load an EEG file first")
            return
        try:
            ch1 = self.selected_ch1.get()
            ch2 = self.selected_ch2.get()
            if not ch1:
                return

            time_val = self.current_time.get()
            channels = [ch1]
            if (not self.auto_wigner.get()) and ch2:
                channels.append(ch2)

            data_dict = self.eeg.get_data(channels, time_val)
            if not data_dict:
                return

            if self.auto_wigner.get() or ch2 not in data_dict:
                signal1 = data_dict[ch1]
                signal2 = None
                title = f"Auto-Wigner: {ch1}"
            else:
                signal1 = data_dict[ch1]
                signal2 = data_dict[ch2]
                title = f"Cross-Wigner: {ch1} × {ch2}"

            try:
                wigner_result, t, f = self.wigner.compute_wigner(signal1, signal2, fs=self.eeg.sfreq)
            except Exception as e:
                logging.warning(f"Full Wigner transform failed: {e}. Using simplified version.")
                wigner_result, t, f = self.wigner.compute_simple_wigner(signal1, signal2, fs=self.eeg.sfreq)

            self.ax2.clear()
            _, im = self.wigner.plot_wigner(wigner_result, t, f, ax=self.ax2,
                                            cmap='viridis' if self.auto_wigner.get() else 'RdBu_r')
            self.ax2.set_title(title)

            # Remove old colorbar carefully
            if self.wigner_cbar:
                try:
                    self.wigner_cbar.remove()
                except Exception as e:
                    logging.warning(f"Error removing old colorbar: {e}")
                self.wigner_cbar = None

            # Create new colorbar with use_gridspec=False
            self.wigner_cbar = self.fig.colorbar(im, ax=self.ax2, use_gridspec=False)

            try:
                self.fig.tight_layout()
            except Exception as e:
                logging.warning(f"tight_layout error (wigner): {e}")

            self.canvas.draw()
            logging.info(f"Computed Wigner transform with shape {wigner_result.shape}")

        except Exception as e:
            logging.error(f"Error computing Wigner transform: {e}")
            messagebox.showerror("Error", f"Failed to compute Wigner transform: {str(e)}")

    def compute_coherence(self):
        if not self.eeg.raw:
            messagebox.showwarning("No Data", "Please load an EEG file first")
            return
        
        if not self.selected_channel_types:
            messagebox.showwarning("No Channel Types", "Please select at least one channel type")
            return

        # Get all channels of the selected types
        selected_channels = []
        for ch_type in self.selected_channel_types:
            channels = self.eeg.get_channels_by_type(ch_type)
            selected_channels.extend(channels)
        
        if len(selected_channels) < 2:
            messagebox.showwarning("Not Enough Channels", 
                                  f"Need at least 2 channels of types {', '.join(self.selected_channel_types)}")
            return

        # If in auto-wigner mode, only use the selected channel if it's in the selected types
        if self.auto_wigner.get():
            ch1 = self.selected_ch1.get()
            ch_type = self.eeg.channel_types.get(ch1, "Unknown")
            if ch_type not in self.selected_channel_types:
                messagebox.showwarning("Invalid Channel", 
                                      f"Selected channel {ch1} is not of the selected types")
                return
            used_channels = [ch1]
        else:
            # If in cross-wigner mode, use both selected channels if they're of the right types
            ch1 = self.selected_ch1.get()
            ch2 = self.selected_ch2.get()
            ch1_type = self.eeg.channel_types.get(ch1, "Unknown")
            ch2_type = self.eeg.channel_types.get(ch2, "Unknown")
            
            # Build the list of channels to use
            used_channels = []
            if ch1_type in self.selected_channel_types:
                used_channels.append(ch1)
            if ch2_type in self.selected_channel_types:
                used_channels.append(ch2)
                
            if len(used_channels) < 2:
                # If we don't have 2 channels from selected types, add more from the selected types
                for ch in selected_channels:
                    if ch not in used_channels:
                        used_channels.append(ch)
                        if len(used_channels) >= 2:
                            break

        time_val = self.current_time.get()
        data_dict = self.eeg.get_data(used_channels, time_val)
        if not data_dict:
            messagebox.showwarning("No Data", "Failed to get EEG data for coherence")
            return

        # Create results dialog
        result_dialog = tk.Toplevel(self.root)
        result_dialog.title("Neural Coherence Results")
        result_dialog.geometry("600x500")
        result_dialog.transient(self.root)
        result_dialog.grab_set()
        
        # Create scrollable text widget for results
        result_text = scrolledtext.ScrolledText(result_dialog, wrap=tk.WORD, width=70, height=25)
        result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Compute coherence and display results
        coherence_index, details = self.coherence_calc.compute_coherence_index(data_dict, fs=self.eeg.sfreq)
        
        result_text.insert(tk.END, f"Coherence Index: {coherence_index:.4f}\n\n")
        result_text.insert(tk.END, f"Channel Pairs Analysis:\n")
        result_text.insert(tk.END, f"{'Channel Pair':<20} {'Negativity':<15} {'PLV':<15}\n")
        result_text.insert(tk.END, f"{'-'*50}\n")
        
        for pair, vals in details.items():
            if isinstance(vals, dict) and 'negativity' in vals:
                result_text.insert(tk.END, f"{pair:<20} {vals['negativity']:.4f}{' '*10} {vals['plv']:.4f}\n")
        
        result_text.insert(tk.END, f"\nSummary Statistics:\n")
        result_text.insert(tk.END, f"{'-'*50}\n")
        if "avg_negativity" in details:
            result_text.insert(tk.END, f"Average Negativity: {details['avg_negativity']:.4f}\n")
        if "avg_plv" in details:
            result_text.insert(tk.END, f"Average PLV: {details['avg_plv']:.4f}\n")
        result_text.insert(tk.END, f"Coherence Index: {details.get('coherence_index', 0):.4f}\n")
        
        result_text.insert(tk.END, f"\nChannel Types Used: {', '.join(sorted(self.selected_channel_types))}\n")
        result_text.insert(tk.END, f"Channels Included: {', '.join(sorted(used_channels))}\n")
        
        # Make the text widget read-only
        result_text.config(state=tk.DISABLED)
        
        # Add close button
        ttk.Button(result_dialog, text="Close", command=result_dialog.destroy).pack(pady=10)

    def save_figure(self):
        filepath = filedialog.asksaveasfilename(
            title="Save Figure",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                       ("JPEG files", "*.jpg"),
                       ("PDF files", "*.pdf"),
                       ("SVG files", "*.svg"),
                       ("All files", "*.*")]
        )
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Figure saved to {filepath}")
            except Exception as e:
                logging.error(f"Error saving figure: {e}")
                messagebox.showerror("Error", f"Failed to save figure: {str(e)}")


def main():
    root = tk.Tk()
    root.geometry("1200x800")
    root.minsize(800, 600)
    app = EEGWignerViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
