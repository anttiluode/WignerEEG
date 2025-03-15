#!/usr/bin/env python3
"""
Wigner Transform Visualizer
- Beautiful full-screen Wigner-Ville distribution visualization
- Frame-by-frame playback with variable speed control
- Time seeking capability to visualize any point in the EEG
- High-quality colormap options for stunning visualizations
"""

# ------------------------------------------------------------------
# 1) IMPORTS
# ------------------------------------------------------------------
# Python standard library
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale
import threading
import logging
import os
import io
from typing import Tuple, List, Optional, Dict, Any, Union

# Scientific and data processing
import numpy as np
from scipy import signal

# Visualization
import matplotlib
matplotlib.use("Agg")  # We'll generate Wigner images off-screen
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# Image processing
from PIL import Image, ImageTk

# EEG processing
import mne

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------
# 2) WIGNER TRANSFORM UTILS
# ------------------------------------------------------------------
class WignerTransform:
    @staticmethod
    def compute_wigner(sig1, sig2=None, fs=1000.0):
        """Compute the Wigner-Ville distribution.
        
        Args:
            sig1: First signal
            sig2: Second signal (optional, auto-Wigner if None)
            fs: Sampling frequency
            
        Returns:
            Tuple of (wigner matrix, time array, frequency array)
        """
        sig1 = np.asarray(sig1).flatten()
        if sig2 is None:
            sig2 = sig1
        else:
            sig2 = np.asarray(sig2).flatten()

        min_len = min(len(sig1), len(sig2))
        sig1 = sig1[:min_len]
        sig2 = sig2[:min_len]

        a1 = signal.hilbert(sig1)
        a2 = signal.hilbert(sig2)

        n_freq = 1
        while n_freq < min_len:
            n_freq *= 2

        t = np.arange(min_len) / fs
        f = np.fft.fftshift(np.fft.fftfreq(n_freq, 1/fs))

        wvd = np.zeros((min_len, n_freq), dtype=complex)
        half_len = min_len // 2

        for tau in range(-half_len, half_len):
            idx_t = np.arange(max(0, -tau), min(min_len, min_len - tau))
            idx_shifted = idx_t + tau
            valid = (idx_shifted >= 0) & (idx_shifted < min_len)
            if not np.any(valid):
                continue
            idx_t = idx_t[valid]
            idx_shifted = idx_shifted[valid]
            product = a1[idx_t] * np.conjugate(a2[idx_shifted])

            padded = np.zeros(n_freq, dtype=complex)
            padded[:len(product)] = product
            F = np.fft.fftshift(np.fft.fft(padded))
            tau_idx = tau + half_len
            if 0 <= tau_idx < min_len:
                wvd[tau_idx] = F

        # Real part if auto
        if np.array_equal(sig1, sig2):
            wvd = np.real(wvd)
        return wvd, t, f

    @staticmethod
    def get_colormap(name, alpha=1.0):
        """Get a matplotlib colormap by name with optional alpha adjustment."""
        cmap = plt.get_cmap(name)
        
        if alpha < 1.0:
            # Create a new colormap with alpha
            cmap_rgba = cmap(np.arange(cmap.N))
            cmap_rgba[:, -1] = alpha
            return LinearSegmentedColormap.from_list(f"{name}_alpha", cmap_rgba)
        return cmap
    
    @staticmethod
    def plot_wigner(wigner_matrix, t, f, title="Wigner-Ville Distribution", colormap='magma', 
                    figsize=(8, 6), dpi=100, show_colorbar=True, normalize=True, vmin=None, vmax=None):
        """Generate a Wigner plot as a PIL Image.
        
        Args:
            wigner_matrix: The computed Wigner-Ville distribution
            t: Time array
            f: Frequency array
            title: Title for the plot
            colormap: Matplotlib colormap name
            figsize: Figure size in inches
            dpi: Resolution in dots per inch
            show_colorbar: Whether to display a colorbar
            normalize: Whether to normalize the Wigner matrix
            vmin: Minimum value for color scaling
            vmax: Maximum value for color scaling
            
        Returns:
            PIL Image of the Wigner plot
        """
        if np.iscomplexobj(wigner_matrix):
            wigner_matrix = np.real(wigner_matrix)

        # Handle size adjustments if needed
        if len(t) != wigner_matrix.shape[0]:
            t = np.linspace(0, wigner_matrix.shape[0]/len(t), wigner_matrix.shape[0]) if len(t) > 0 else np.arange(wigner_matrix.shape[0])
        if len(f) != wigner_matrix.shape[1]:
            f = np.linspace(f[0], f[-1], wigner_matrix.shape[1]) if len(f) > 0 else np.arange(wigner_matrix.shape[1])

        # Create figure with minimal white space
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Set up plotting extent
        extent = [t[0], t[-1], f[0], f[-1]] 
        
        # Plot the Wigner distribution
        if normalize and vmin is None and vmax is None:
            # Find a reasonable normalization that avoids extreme values
            perc_low, perc_high = np.percentile(wigner_matrix, [2, 98])
            vmin = perc_low if perc_low < 0 else None
            vmax = perc_high if perc_high > 0 else None
            
        im = ax.imshow(wigner_matrix.T, origin='lower', aspect='auto',
                   extent=extent, cmap=colormap, vmin=vmin, vmax=vmax)
        
        # Style the plot
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Add colorbar if requested
        if show_colorbar:
            plt.colorbar(im, ax=ax, pad=0.01)
        
        # Tighten the layout to maximize visualization area
        plt.tight_layout(pad=0.5)
        
        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

# ------------------------------------------------------------------
# 3) EEG LOADER
# ------------------------------------------------------------------
class EEGLoader:
    def __init__(self):
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        self.channels = []
        self.window_size = 2.0
        self.ch1 = None
        self.ch2 = None

    def load_file(self, path):
        """Load an EEG file and extract metadata."""
        try:
            self.raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            self.channels = self.raw.ch_names
            logging.info(f"Loaded {path} with {len(self.channels)} channels, duration: {self.duration:.2f}s")
            return True
        except Exception as e:
            logging.error(f"Failed to load EDF: {e}")
            return False

    def set_channels(self, c1, c2=None):
        """Set which EEG channels to use."""
        self.ch1 = c1
        self.ch2 = c2

    def get_data_segment(self, start_time):
        """Get a segment of EEG data from specified start time."""
        if self.raw is None:
            return None, None
            
        start_samp = int(start_time * self.sfreq)
        end_samp = min(int(start_samp + self.window_size*self.sfreq), self.raw.n_times)

        d1 = d2 = None
        if self.ch1 in self.channels:
            idx1 = self.raw.ch_names.index(self.ch1)
            d1, _ = self.raw[idx1, start_samp:end_samp]
            d1 = d1.flatten()
        if self.ch2 in self.channels:
            idx2 = self.raw.ch_names.index(self.ch2)
            d2, _ = self.raw[idx2, start_samp:end_samp]
            d2 = d2.flatten()

        return d1, d2

# ------------------------------------------------------------------
# 4) MAIN GUI
# ------------------------------------------------------------------
class WignerVisualizerGUI:
    def __init__(self, root):
        """Initialize the main GUI."""
        self.root = root
        self.root.title("Wigner Transform Visualizer")
        
        # Set a dark theme
        self.root.configure(bg="#2E2E2E")

        # Initialize components
        self.eeg = EEGLoader()
        self.wigner = WignerTransform()

        # Playback state
        self.current_time = 0.0
        self.playing = False
        self.wigner_image = None
        self.max_time = 0.0
        
        # Define available colormaps
        self.colormap_options = {
            'magma': "Magma",
            'viridis': "Viridis",
            'plasma': "Plasma",
            'inferno': "Inferno", 
            'cividis': "Cividis",
            'jet': "Jet",
            'rainbow': "Rainbow",
            'turbo': "Turbo",
            'twilight': "Twilight",
            'twilight_shifted': "Twilight Shifted",
            'hot': "Hot",
            'cool': "Cool"
        }
        self.current_colormap = "magma"
        
        # UI setup
        self.setup_ui()

    def setup_ui(self):
        """Create the main UI layout."""
        # Create a frame for the left panel (controls)
        control_frame = tk.Frame(self.root, bg="#2E2E2E", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Create a frame for the right panel (visualization)
        viz_frame = tk.Frame(self.root, bg="#1A1A1A")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for Wigner visualization
        self.canvas = tk.Canvas(viz_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Setup control sections
        self.setup_eeg_controls(control_frame)
        self.setup_wigner_controls(control_frame)
        self.setup_playback_controls(control_frame)
        
        # Bind keyboard shortcuts
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Left>', lambda e: self.seek_time(-self.eeg.window_size))
        self.root.bind('<Right>', lambda e: self.seek_time(self.eeg.window_size))
    
    def setup_eeg_controls(self, parent):
        """Setup EEG file and channel selection controls."""
        eeg_frame = tk.LabelFrame(parent, text="EEG Input", bg="#2E2E2E", fg="white", font=("Arial", 10, "bold"))
        eeg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File loading button
        load_btn = tk.Button(
            eeg_frame, text="Load EEG File", command=self.load_eeg_file,
            bg="#444", fg="white", height=2
        )
        load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # File path display
        self.file_label = tk.Label(eeg_frame, text="No file loaded", fg="#888", bg="#2E2E2E")
        self.file_label.pack(padx=5, pady=2)
        
        # Channel selection
        channel_frame = tk.Frame(eeg_frame, bg="#2E2E2E")
        channel_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Channel 1
        tk.Label(channel_frame, text="Channel 1:", bg="#2E2E2E", fg="white").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ch1_var = tk.StringVar()
        self.ch1_combo = ttk.Combobox(channel_frame, textvariable=self.ch1_var, state="readonly", width=15)
        self.ch1_combo.grid(row=0, column=1, padx=5, pady=2)
        self.ch1_combo.bind("<<ComboboxSelected>>", lambda e: self.update_channels())
        
        # Channel 2 (optional)
        tk.Label(channel_frame, text="Channel 2:", bg="#2E2E2E", fg="white").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.ch2_var = tk.StringVar()
        self.ch2_combo = ttk.Combobox(channel_frame, textvariable=self.ch2_var, state="readonly", width=15)
        self.ch2_combo.grid(row=1, column=1, padx=5, pady=2)
        self.ch2_combo.bind("<<ComboboxSelected>>", lambda e: self.update_channels())
        
        # Window size control
        window_frame = tk.Frame(eeg_frame, bg="#2E2E2E")
        window_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(window_frame, text="Window Size (s):", bg="#2E2E2E", fg="white").pack(side=tk.LEFT, padx=5)
        self.window_var = tk.DoubleVar(value=2.0)
        
        window_scale = Scale(
            window_frame, variable=self.window_var, from_=0.1, to=5.0, 
            resolution=0.1, orient=tk.HORIZONTAL, bg="#2E2E2E", fg="white",
            highlightthickness=0, troughcolor="#555", activebackground="#777"
        )
        window_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        window_scale.bind("<ButtonRelease-1>", lambda e: self.update_window_size())
        
        self.window_label = tk.Label(window_frame, text="2.0s", bg="#2E2E2E", fg="white", width=4)
        self.window_label.pack(side=tk.RIGHT, padx=5)
    
    def setup_wigner_controls(self, parent):
        """Setup Wigner transform display controls."""
        wigner_frame = tk.LabelFrame(parent, text="Wigner Transform", bg="#2E2E2E", fg="white", font=("Arial", 10, "bold"))
        wigner_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Colormap selection
        cmap_frame = tk.Frame(wigner_frame, bg="#2E2E2E")
        cmap_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(cmap_frame, text="Colormap:", bg="#2E2E2E", fg="white").pack(side=tk.LEFT, padx=5)
        self.cmap_var = tk.StringVar(value="Magma")
        cmap_combo = ttk.Combobox(
            cmap_frame, textvariable=self.cmap_var, 
            values=list(self.colormap_options.values()), 
            state="readonly", width=12
        )
        cmap_combo.current(0)
        cmap_combo.pack(side=tk.RIGHT, padx=5)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_colormap())
        
        # Display options
        options_frame = tk.Frame(wigner_frame, bg="#2E2E2E")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Frequency range option
        freq_frame = tk.Frame(options_frame, bg="#2E2E2E")
        freq_frame.pack(fill=tk.X, padx=0, pady=2)
        
        tk.Label(freq_frame, text="Max Frequency (Hz):", bg="#2E2E2E", fg="white").pack(side=tk.LEFT, padx=5)
        self.freq_var = tk.IntVar(value=100)
        freq_options = [25, 50, 100, 150, 200, "All"]
        freq_combo = ttk.Combobox(
            freq_frame, textvariable=self.freq_var, 
            values=freq_options, state="readonly", width=5
        )
        freq_combo.current(2)  # Default to 100 Hz
        freq_combo.pack(side=tk.RIGHT, padx=5)
        freq_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_wigner())
        
        # Show colorbar option
        colorbar_frame = tk.Frame(options_frame, bg="#2E2E2E")
        colorbar_frame.pack(fill=tk.X, padx=0, pady=2)
        
        self.colorbar_var = tk.BooleanVar(value=True)
        colorbar_check = tk.Checkbutton(
            colorbar_frame, text="Show Colorbar", variable=self.colorbar_var,
            bg="#2E2E2E", fg="white", selectcolor="#555", 
            activebackground="#3E3E3E", activeforeground="white"
        )
        colorbar_check.pack(side=tk.LEFT, padx=5)
        colorbar_check.bind("<ButtonRelease-1>", lambda e: self.refresh_wigner())
        
        # Normalize option
        self.normalize_var = tk.BooleanVar(value=True)
        normalize_check = tk.Checkbutton(
            colorbar_frame, text="Auto Normalize", variable=self.normalize_var,
            bg="#2E2E2E", fg="white", selectcolor="#555", 
            activebackground="#3E3E3E", activeforeground="white"
        )
        normalize_check.pack(side=tk.RIGHT, padx=5)
        normalize_check.bind("<ButtonRelease-1>", lambda e: self.refresh_wigner())
        
        # Refresh button
        refresh_btn = tk.Button(
            wigner_frame, text="Refresh Wigner Plot", command=self.refresh_wigner,
            bg="#006699", fg="white", height=2
        )
        refresh_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Screenshot button
        screenshot_btn = tk.Button(
            wigner_frame, text="Save Screenshot", command=self.take_screenshot,
            bg="#444", fg="white"
        )
        screenshot_btn.pack(fill=tk.X, padx=5, pady=5)
    
    def setup_playback_controls(self, parent):
        """Setup playback and time navigation controls."""
        play_frame = tk.LabelFrame(parent, text="Playback", bg="#2E2E2E", fg="white", font=("Arial", 10, "bold"))
        play_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time slider
        time_frame = tk.Frame(play_frame, bg="#2E2E2E")
        time_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.time_scale = Scale(
            time_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            bg="#2E2E2E", fg="white", highlightthickness=0,
            troughcolor="#555", activebackground="#777"
        )
        self.time_scale.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.time_scale.bind("<ButtonRelease-1>", self.on_time_scale_change)
        
        # Current time display
        self.time_label = tk.Label(time_frame, text="Time: 0.0s / 0.0s", bg="#2E2E2E", fg="white")
        self.time_label.pack(padx=5, pady=2)
        
        # Playback controls
        controls_frame = tk.Frame(play_frame, bg="#2E2E2E")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons in a row
        btn_frame = tk.Frame(controls_frame, bg="#2E2E2E")
        btn_frame.pack(fill=tk.X, padx=0, pady=2)
        
        # Step backward button
        self.back_btn = tk.Button(
            btn_frame, text="⏮", command=lambda: self.seek_time(-self.eeg.window_size),
            bg="#444", fg="white", width=3, font=("Arial", 12)
        )
        self.back_btn.pack(side=tk.LEFT, padx=2, pady=5, fill=tk.X, expand=True)
        
        # Play/pause button
        self.play_btn = tk.Button(
            btn_frame, text="▶", command=self.toggle_play,
            bg="#008800", fg="white", width=3, font=("Arial", 12)
        )
        self.play_btn.pack(side=tk.LEFT, padx=2, pady=5, fill=tk.X, expand=True)
        
        # Step forward button
        self.fwd_btn = tk.Button(
            btn_frame, text="⏭", command=lambda: self.seek_time(self.eeg.window_size),
            bg="#444", fg="white", width=3, font=("Arial", 12)
        )
        self.fwd_btn.pack(side=tk.LEFT, padx=2, pady=5, fill=tk.X, expand=True)
        
        # Speed control
        speed_frame = tk.Frame(controls_frame, bg="#2E2E2E")
        speed_frame.pack(fill=tk.X, padx=0, pady=2)
        
        tk.Label(speed_frame, text="Speed:", bg="#2E2E2E", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = Scale(
            speed_frame, variable=self.speed_var, from_=0.25, to=3.0, 
            resolution=0.25, orient=tk.HORIZONTAL, 
            bg="#2E2E2E", fg="white", highlightthickness=0,
            troughcolor="#555", activebackground="#777"
        )
        speed_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        self.speed_label = tk.Label(speed_frame, text="1.0x", bg="#2E2E2E", fg="white", width=3)
        self.speed_label.pack(side=tk.RIGHT, padx=5)
        speed_scale.bind("<ButtonRelease-1>", lambda e: self.update_speed())

    #-----------------------
    # Callback functions
    #-----------------------
    def update_window_size(self):
        """Update window size for EEG segments."""
        self.eeg.window_size = self.window_var.get()
        self.window_label.config(text=f"{self.eeg.window_size:.1f}s")
        # Refresh the Wigner plot
        self.refresh_wigner()
    
    def update_channels(self):
        """Handle channel selection updates."""
        # Update the selected channels and refresh
        if self.ch1_var.get():
            self.refresh_wigner()
    
    def update_colormap(self):
        """Update the Wigner plot colormap."""
        colormap_name = self.cmap_var.get()
        # Find the key (actual colormap name) from the displayed value
        for key, value in self.colormap_options.items():
            if value == colormap_name:
                self.current_colormap = key
                break
        
        # Refresh the Wigner plot
        self.refresh_wigner()
    
    def update_speed(self):
        """Update playback speed."""
        speed = self.speed_var.get()
        self.speed_label.config(text=f"{speed:.1f}x")
    
    def on_time_scale_change(self, event):
        """Handle time slider changes."""
        if not self.eeg.raw:
            return
            
        # Calculate time from slider position
        pos = self.time_scale.get() / 100.0
        self.current_time = pos * (self.eeg.duration - self.eeg.window_size)
        
        # Update time display
        self.update_time_display()
        
        # Refresh the Wigner plot
        self.refresh_wigner()

    #-----------------------
    # Action functions
    #-----------------------
    def load_eeg_file(self):
        """Load an EEG file and set up channels."""
        file_path = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        success = self.eeg.load_file(file_path)
        if success:
            # Update UI to show loaded file
            file_name = os.path.basename(file_path)
            self.file_label.config(text=file_name, fg="green")
            
            # Update channel dropdowns
            channels = self.eeg.channels
            self.ch1_combo["values"] = channels
            self.ch2_combo["values"] = ["None"] + channels
            
            # Set default channels
            if len(channels) > 0:
                self.ch1_combo.current(0)
            if len(channels) > 1:
                self.ch2_combo.current(2)  # Set to second channel
            else:
                self.ch2_combo.current(0)  # Set to None
                
            # Reset current time and update slider range
            self.current_time = 0.0
            self.max_time = self.eeg.duration
            
            # Update time display
            self.update_time_display()
            
            # Generate initial Wigner plot
            self.refresh_wigner()
        else:
            self.file_label.config(text="Failed to load file", fg="red")
            messagebox.showerror("Error", "Failed to load EEG file")
    
    def refresh_wigner(self):
        """Generate or refresh the Wigner plot for the current time."""
        if not self.eeg.raw:
            messagebox.showinfo("Info", "Please load an EEG file first")
            return
            
        # Get selected channels
        ch1 = self.ch1_var.get()
        ch2 = self.ch2_var.get()
        if ch2 == "None":
            ch2 = None
            
        # Set channels and get data
        self.eeg.set_channels(ch1, ch2)
        d1, d2 = self.eeg.get_data_segment(self.current_time)
        
        if d1 is None:
            messagebox.showerror("Error", "Could not get EEG data")
            return
            
        try:
            # Compute Wigner transform
            wvd, t, f = self.wigner.compute_wigner(d1, d2, fs=self.eeg.sfreq)
            
            # Filter frequency range if selected
            if self.freq_var.get() != "All":
                max_freq = float(self.freq_var.get())
                # Find index of maximum frequency
                max_idx = np.searchsorted(f, max_freq)
                if max_idx < len(f):
                    # Crop the frequency range
                    min_idx = np.searchsorted(f, -max_freq)
                    wvd = wvd[:, min_idx:max_idx]
                    f = f[min_idx:max_idx]
            
            # Generate plot with current settings
            title = f"Wigner-Ville Distribution - Time: {self.current_time:.2f}s"
            
            # Get canvas size for optimal figure dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Set default if canvas not yet sized
            if canvas_width < 100 or canvas_height < 100:
                canvas_width, canvas_height = 800, 600
                
            # Calculate DPI and figure size to maximize quality
            dpi = 100
            figsize = (canvas_width / dpi, canvas_height / dpi)
            
            # Generate the Wigner plot
            self.wigner_image = self.wigner.plot_wigner(
                wigner_matrix=wvd, 
                t=t, 
                f=f, 
                title=title,
                colormap=self.current_colormap,
                figsize=figsize,
                dpi=dpi,
                show_colorbar=self.colorbar_var.get(),
                normalize=self.normalize_var.get()
            )
            
            # Update the display
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error computing Wigner transform: {str(e)}")
            logging.error(f"Wigner error: {e}")
    
    def take_screenshot(self):
        """Save the current Wigner plot as an image."""
        if self.wigner_image is None:
            messagebox.showinfo("Info", "No Wigner plot to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Screenshot",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Save the current Wigner image
            self.wigner_image.save(file_path)
            messagebox.showinfo("Success", f"Screenshot saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save screenshot: {str(e)}")
            logging.error(f"Screenshot error: {e}")
    
    def toggle_play(self):
        """Start or stop playback animation."""
        if not self.eeg.raw:
            messagebox.showinfo("Info", "Please load an EEG file first")
            return
            
        # Toggle playback state
        self.playing = not self.playing
        
        if self.playing:
            # Update button to show pause icon
            self.play_btn.config(text="⏸", bg="#880000")
            # Start playback loop
            self.playback_loop()
        else:
            # Update button to show play icon
            self.play_btn.config(text="▶", bg="#008800")
    
    def seek_time(self, time_delta):
        """Jump forward or backward in time."""
        if not self.eeg.raw:
            return
            
        # Calculate new time
        new_time = self.current_time + time_delta
        
        # Ensure time stays within bounds
        if new_time < 0:
            new_time = 0
        elif new_time > self.eeg.duration - self.eeg.window_size:
            new_time = self.eeg.duration - self.eeg.window_size
            
        # Update current time
        self.current_time = new_time
        
        # Update time display and slider
        self.update_time_display()
        
        # Update Wigner plot
        self.refresh_wigner()
    
    def playback_loop(self):
        """Main playback loop for animating through EEG data."""
        if not self.playing:
            return
            
        # Get data for current time and generate Wigner plot
        self.refresh_wigner()
        
        # Advance time based on speed
        self.current_time += self.eeg.window_size * 0.25 * self.speed_var.get()
        
        # Check if we've reached the end
        if self.current_time >= self.eeg.duration - self.eeg.window_size:
            # Loop back to beginning
            self.current_time = 0
            
        # Update time display and slider
        self.update_time_display()
        
        # Schedule next frame
        delay = int(250 / self.speed_var.get())  # Adjust delay based on speed (milliseconds)
        self.root.after(delay, self.playback_loop)
    
    def update_time_display(self):
        """Update the time display and slider position."""
        # Update time label
        self.time_label.config(text=f"Time: {self.current_time:.2f}s / {self.eeg.duration:.2f}s")
        
        # Update slider position (avoid triggering events)
        if self.eeg.duration > 0:
            pos = (self.current_time / (self.eeg.duration - self.eeg.window_size)) * 100
            self.time_scale.set(pos)
    
    def update_display(self):
        """Update the canvas with the current Wigner image."""
        if self.wigner_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Default dimensions if canvas not yet sized
        if canvas_width < 100 or canvas_height < 100:
            canvas_width, canvas_height = 800, 600
            
        # Calculate scaling to fit image to canvas while preserving aspect ratio
        img_width, img_height = self.wigner_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image to fit canvas
        resized_img = self.wigner_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        tk_image = ImageTk.PhotoImage(resized_img)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        
        # Position image in center of canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Create image on canvas
        self.canvas.create_image(x, y, image=tk_image, anchor=tk.NW)
        
        # Keep reference to prevent garbage collection
        self.canvas.image = tk_image

def main():
    """Main entry point for the application."""
    root = tk.Tk()
    root.geometry("1280x800")  # Set initial window size
    root.configure(bg="#2E2E2E")
    
    # Set up styles for ttk widgets
    style = ttk.Style()
    style.theme_use("default")
    style.configure("TCombobox", 
                    fieldbackground="#444",
                    background="#2E2E2E", 
                    foreground="white", 
                    arrowcolor="white")
    style.map("TCombobox", 
              fieldbackground=[("readonly", "#444")],
              selectbackground=[("readonly", "#666")],
              selectforeground=[("readonly", "white")])
    
    app = WignerVisualizerGUI(root)
    
    # Make the window responsive to resizing
    def on_resize(event):
        # Only trigger on substantial size changes to avoid too many redraws
        if event.widget == root and app.wigner_image is not None:
            # Delay refresh to avoid flickering during resize
            root.after(100, app.update_display)
    
    root.bind("<Configure>", on_resize)
    
    root.mainloop()

if __name__ == "__main__":
    main()