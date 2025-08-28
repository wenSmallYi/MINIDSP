#!/usr/bin/env python3
"""
MINIDSP UMIK-1 Recording and Analysis Program (Command Line Version)

This program provides a complete solution for recording audio using the 
MINIDSP UMIK-1 microphone with official calibration file support, 
parameter configuration, and comprehensive visualization capabilities.

Features:
1. Microphone calibration using official calibration files
2. Basic parameter settings (sample rate, recording duration, etc.)
3. Audio recording functionality
4. Time domain and frequency domain visualization
5. Both log and linear scale frequency plots

Author: Audio Analysis Program
Date: 2025-01-28
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal
from scipy.io import wavfile
import pandas as pd
import os
import json
from datetime import datetime
import argparse
import time


class UMIK1RecorderCLI:
    """
    MINIDSP UMIK-1 microphone recording and analysis class (Command Line Interface)
    """
    
    def __init__(self):
        """Initialize the recorder with default parameters"""
        # Default recording parameters
        self.sample_rate = 48000  # Hz
        self.duration = 10.0      # seconds
        self.channels = 1         # mono recording
        self.dtype = np.float32
        
        # Calibration data
        self.calibration_data = None
        self.calibration_file = None
        
        # Recording data
        self.recorded_audio = None
        self.time_vector = None
        
        # Device settings
        self.device_name = None
        self.device_id = None
        
    def list_devices(self):
        """List all available audio input devices"""
        print("\nAvailable Audio Input Devices:")
        print("=" * 50)
        
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"{i:2d}: {device['name']} ({device['max_input_channels']} channels)")
                    input_devices.append((i, device['name']))
                    
                    # Auto-detect UMIK-1
                    if 'umik' in device['name'].lower() or 'minidsp' in device['name'].lower():
                        print(f"     ^^ UMIK-1 DETECTED ^^")
                        self.device_id = i
                        self.device_name = device['name']
            
            print(f"\nFound {len(input_devices)} input devices")
            return input_devices
            
        except Exception as e:
            print(f"Error listing devices: {e}")
            return []
    
    def select_device(self, device_id=None):
        """Select audio input device"""
        if device_id is None:
            devices = self.list_devices()
            
            while True:
                try:
                    if self.device_id is not None:
                        choice = input(f"\nSelect device ID (default: {self.device_id} - auto-detected): ").strip()
                        device_id = int(choice) if choice else self.device_id
                    else:
                        device_id = int(input("\nEnter device ID: "))
                    break
                except ValueError:
                    print("Please enter a valid device ID number")
        
        try:
            device_info = sd.query_devices(device_id)
            if device_info['max_input_channels'] > 0:
                self.device_id = device_id
                self.device_name = device_info['name']
                print(f"Selected device: {self.device_name}")
                return True
            else:
                print("Selected device has no input channels")
                return False
        except Exception as e:
            print(f"Error selecting device: {e}")
            return False
    
    def load_calibration_file(self, file_path=None):
        """Load microphone calibration file"""
        if file_path is None:
            file_path = input("Enter calibration file path (or press Enter to skip): ").strip()
            
        if not file_path:
            print("Skipping calibration file")
            return True
            
        if not os.path.exists(file_path):
            print(f"Calibration file not found: {file_path}")
            return False
        
        try:
            self.calibration_data = self.parse_calibration_file(file_path)
            self.calibration_file = file_path
            filename = os.path.basename(file_path)
            print(f"Loaded calibration file: {filename}")
            print(f"Frequency range: {self.calibration_data['frequencies'][0]:.1f} - {self.calibration_data['frequencies'][-1]:.1f} Hz")
            return True
            
        except Exception as e:
            print(f"Failed to load calibration file: {e}")
            return False
    
    def parse_calibration_file(self, file_path):
        """Parse the UMIK-1 calibration file"""
        try:
            # Try to read as CSV first
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                # Try to read as space or tab delimited
                data = pd.read_csv(file_path, delimiter=None, engine='python')
            
            # Expected format: Frequency (Hz), Magnitude (dB)
            if len(data.columns) >= 2:
                frequencies = data.iloc[:, 0].values
                magnitudes = data.iloc[:, 1].values
                
                # Create calibration dictionary
                calibration = {
                    'frequencies': frequencies,
                    'magnitudes': magnitudes
                }
                
                return calibration
            else:
                raise ValueError("Calibration file must have at least 2 columns (frequency, magnitude)")
                
        except Exception as e:
            raise Exception(f"Error parsing calibration file: {str(e)}")
    
    def configure_recording(self, sample_rate=None, duration=None):
        """Configure recording parameters"""
        print("\nRecording Configuration:")
        print("=" * 30)
        
        # Sample rate
        if sample_rate is None:
            print("Available sample rates: 44100, 48000, 96000 Hz")
            while True:
                try:
                    rate_input = input(f"Sample rate (default: {self.sample_rate}): ").strip()
                    if rate_input:
                        self.sample_rate = int(rate_input)
                        if self.sample_rate not in [44100, 48000, 96000]:
                            print("Warning: Non-standard sample rate selected")
                    break
                except ValueError:
                    print("Please enter a valid sample rate")
        else:
            self.sample_rate = sample_rate
        
        # Duration
        if duration is None:
            while True:
                try:
                    dur_input = input(f"Recording duration in seconds (default: {self.duration}): ").strip()
                    if dur_input:
                        self.duration = float(dur_input)
                        if self.duration <= 0:
                            print("Duration must be positive")
                            continue
                    break
                except ValueError:
                    print("Please enter a valid duration")
        else:
            self.duration = duration
        
        print(f"Configuration: {self.sample_rate} Hz, {self.duration} seconds")
    
    def record_audio(self):
        """Record audio from the selected device"""
        if self.device_id is None:
            print("No device selected. Please select a device first.")
            return False
        
        try:
            print(f"\nRecording from: {self.device_name}")
            print(f"Parameters: {self.sample_rate} Hz, {self.duration} seconds")
            print("Press Ctrl+C to stop early")
            
            # Calculate number of samples
            num_samples = int(self.sample_rate * self.duration)
            
            # Start recording
            print("\nRecording started...")
            start_time = time.time()
            
            self.recorded_audio = sd.rec(
                frames=num_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device_id
            )
            
            # Show progress
            try:
                for i in range(int(self.duration)):
                    time.sleep(1)
                    elapsed = time.time() - start_time
                    remaining = max(0, self.duration - elapsed)
                    print(f"\rRecording... {elapsed:.1f}s / {self.duration:.1f}s (remaining: {remaining:.1f}s)", end="", flush=True)
                
                # Wait for recording to complete
                sd.wait()
                
            except KeyboardInterrupt:
                print("\nRecording stopped by user")
                sd.stop()
            
            print(f"\nRecording completed! Duration: {len(self.recorded_audio) / self.sample_rate:.2f} seconds")
            
            # Create time vector
            self.time_vector = np.linspace(0, len(self.recorded_audio) / self.sample_rate, len(self.recorded_audio))
            
            return True
            
        except Exception as e:
            print(f"Recording failed: {e}")
            return False
    
    def save_recording(self, output_file=None):
        """Save the recorded audio to a WAV file"""
        if self.recorded_audio is None:
            print("No recording to save")
            return False
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"umik1_recording_{timestamp}.wav"
        
        try:
            # Convert to 16-bit integer for WAV file
            audio_int16 = (self.recorded_audio.flatten() * 32767).astype(np.int16)
            wavfile.write(output_file, self.sample_rate, audio_int16)
            
            print(f"Recording saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"Failed to save recording: {e}")
            return False
    
    def apply_calibration(self, audio_data):
        """Apply microphone calibration to the audio data"""
        if self.calibration_data is None:
            return audio_data
            
        try:
            print("Applying calibration...")
            
            # Compute FFT
            fft_data = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            
            # Get positive frequencies only
            positive_freq_mask = frequencies >= 0
            frequencies = frequencies[positive_freq_mask]
            fft_data = fft_data[positive_freq_mask]
            
            # Interpolate calibration data to match FFT frequencies
            cal_freqs = self.calibration_data['frequencies']
            cal_mags = self.calibration_data['magnitudes']
            
            # Interpolate calibration magnitudes
            cal_interp = np.interp(frequencies, cal_freqs, cal_mags)
            
            # Apply calibration (subtract in dB domain = divide in linear domain)
            cal_linear = 10**(cal_interp / 20)
            fft_calibrated = fft_data / cal_linear
            
            # Convert back to time domain
            # Reconstruct full FFT (including negative frequencies)
            fft_full = np.zeros(len(audio_data), dtype=complex)
            fft_full[:len(fft_calibrated)] = fft_calibrated
            fft_full[len(fft_calibrated):] = np.conj(fft_calibrated[1:len(audio_data)-len(fft_calibrated)+1][::-1])
            
            calibrated_audio = np.real(np.fft.ifft(fft_full))
            
            print("Calibration applied successfully")
            return calibrated_audio
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return audio_data  # Return uncalibrated data if calibration fails
    
    def analyze_recording(self, save_plots=True):
        """Analyze and visualize the recorded audio"""
        if self.recorded_audio is None:
            print("No recording to analyze")
            return False
        
        try:
            print("\nAnalyzing recording...")
            
            # Flatten audio if it's 2D
            audio_data = self.recorded_audio.flatten()
            
            # Apply calibration if available
            original_audio = audio_data.copy()
            if self.calibration_data is not None:
                audio_data = self.apply_calibration(audio_data)
            
            # Create analysis plots
            self.create_analysis_plots(audio_data, save_plots)
            
            # Create spectral analysis
            self.create_spectral_analysis(audio_data, save_plots)
            
            # Print basic statistics
            self.print_statistics(audio_data)
            
            print("Analysis complete!")
            return True
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            return False
    
    def create_analysis_plots(self, audio_data, save_plots=True):
        """Create time domain and frequency domain analysis plots"""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('UMIK-1 Audio Analysis', fontsize=16, fontweight='bold')
        
        # Time domain plot
        time_vector = np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data))
        ax1.plot(time_vector, audio_data, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time Domain Signal')
        ax1.grid(True, alpha=0.3)
        
        # RMS over time (windowed)
        window_size = int(0.1 * self.sample_rate)  # 100ms windows
        rms_values = []
        rms_times = []
        
        for i in range(0, len(audio_data) - window_size, window_size//2):
            window = audio_data[i:i+window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(20 * np.log10(rms + 1e-12))  # Convert to dB
            rms_times.append(i / self.sample_rate)
        
        ax2.plot(rms_times, rms_values, 'r-', linewidth=2)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('RMS Level (dB)')
        ax2.set_title('RMS Level Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Frequency domain - Linear scale
        fft_data = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Get positive frequencies only
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        magnitude_spectrum = np.abs(fft_data[positive_freq_mask])
        
        # Linear scale plot - use raw magnitude values
        ax3.plot(frequencies, magnitude_spectrum, 'g-', linewidth=1)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude (Linear)')
        ax3.set_title('Frequency Spectrum (Linear Scale)')
        ax3.set_xlim(0, self.sample_rate/2)
        ax3.grid(True, alpha=0.3)
        
        # Convert to dB for log scale plot
        magnitude_db = 20 * np.log10(magnitude_spectrum + 1e-12)
        
        # Frequency domain - Log scale
        ax4.semilogx(frequencies[1:], magnitude_db[1:], 'purple', linewidth=1)  # Skip DC component
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude (dB)')
        ax4.set_title('Frequency Spectrum (Log Scale)')
        ax4.set_xlim(20, self.sample_rate/2)
        ax4.grid(True, alpha=0.3)
        
        # Add calibration status to the plot
        cal_status = "Calibrated" if self.calibration_data is not None else "Uncalibrated"
        fig.text(0.02, 0.02, f"Status: {cal_status} | Sample Rate: {self.sample_rate} Hz | Duration: {len(audio_data)/self.sample_rate:.1f}s", 
                fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"umik1_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved to: {plot_file}")
        
        plt.show()
    
    def create_spectral_analysis(self, audio_data, save_plots=True):
        """Create additional spectral analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Spectral Analysis', fontsize=16, fontweight='bold')
        
        # Spectrogram
        f, t, Sxx = scipy.signal.spectrogram(audio_data, self.sample_rate, nperseg=1024)
        im1 = ax1.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('Spectrogram')
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # Power Spectral Density
        f_psd, psd = scipy.signal.welch(audio_data, self.sample_rate, nperseg=1024)
        ax2.semilogy(f_psd, psd, 'b-', linewidth=1)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density (V²/Hz)')
        ax2.set_title('Power Spectral Density')
        ax2.grid(True, alpha=0.3)
        
        # 1/3 Octave band analysis
        octave_freqs, octave_levels = self.calculate_octave_bands(audio_data)
        ax3.bar(range(len(octave_freqs)), octave_levels, alpha=0.7)
        ax3.set_xlabel('1/3 Octave Band')
        ax3.set_ylabel('Level (dB)')
        ax3.set_title('1/3 Octave Band Analysis')
        ax3.set_xticks(range(0, len(octave_freqs), 3))
        ax3.set_xticklabels([f'{f:.0f}' for f in octave_freqs[::3]], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Statistical analysis
        ax4.hist(audio_data, bins=100, alpha=0.7, density=True, color='orange')
        ax4.set_xlabel('Amplitude')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Amplitude Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
RMS: {np.sqrt(np.mean(audio_data**2)):.4f}
Peak: {np.max(np.abs(audio_data)):.4f}
Crest Factor: {np.max(np.abs(audio_data)) / np.sqrt(np.mean(audio_data**2)):.2f}
THD+N: {self.calculate_thd(audio_data):.3f}%"""
        
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"umik1_spectral_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Spectral analysis plot saved to: {plot_file}")
        
        plt.show()
    
    def calculate_octave_bands(self, audio_data):
        """Calculate 1/3 octave band levels"""
        # Standard 1/3 octave band center frequencies
        octave_freqs = np.array([
            25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
            500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000, 12500, 16000, 20000
        ])
        
        # Filter audio into octave bands and calculate levels
        octave_levels = []
        
        for fc in octave_freqs:
            if fc > self.sample_rate / 2:
                break
                
            # Calculate filter parameters
            flow = fc / (2**(1/6))
            fhigh = fc * (2**(1/6))
            
            # Create bandpass filter
            sos = scipy.signal.butter(4, [flow, fhigh], btype='band', 
                                    fs=self.sample_rate, output='sos')
            
            # Apply filter
            filtered = scipy.signal.sosfilt(sos, audio_data)
            
            # Calculate RMS level in dB
            rms = np.sqrt(np.mean(filtered**2))
            level_db = 20 * np.log10(rms + 1e-12)
            octave_levels.append(level_db)
        
        return octave_freqs[:len(octave_levels)], octave_levels
    
    def calculate_thd(self, audio_data):
        """Calculate Total Harmonic Distortion + Noise"""
        # Simple THD+N calculation
        # Find fundamental frequency (assume it's the strongest component)
        fft_data = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
        
        positive_freq_mask = frequencies > 0
        magnitude_spectrum = np.abs(fft_data[positive_freq_mask])
        positive_frequencies = frequencies[positive_freq_mask]
        
        # Find fundamental (strongest component between 20Hz and 2kHz)
        valid_range = (positive_frequencies >= 20) & (positive_frequencies <= 2000)
        if np.any(valid_range):
            fund_idx = np.argmax(magnitude_spectrum[valid_range])
            fund_freq = positive_frequencies[valid_range][fund_idx]
            
            # Calculate total power and fundamental power
            total_power = np.sum(magnitude_spectrum**2)
            
            # Find fundamental bin
            fund_bin = np.argmin(np.abs(positive_frequencies - fund_freq))
            fund_power = magnitude_spectrum[fund_bin]**2
            
            # THD+N = sqrt((Total - Fundamental) / Fundamental) * 100%
            thd_plus_n = np.sqrt((total_power - fund_power) / fund_power) * 100
            
            return min(thd_plus_n, 100)  # Cap at 100%
        else:
            return 0
    
    def print_statistics(self, audio_data):
        """Print detailed statistics about the recording"""
        print("\n" + "="*50)
        print("RECORDING STATISTICS")
        print("="*50)
        
        # Basic statistics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        crest_factor = peak / rms if rms > 0 else 0
        
        print(f"Duration:        {len(audio_data)/self.sample_rate:.2f} seconds")
        print(f"Sample Rate:     {self.sample_rate} Hz")
        print(f"Samples:         {len(audio_data):,}")
        print(f"RMS Level:       {rms:.6f} ({20*np.log10(rms + 1e-12):.1f} dB)")
        print(f"Peak Level:      {peak:.6f} ({20*np.log10(peak + 1e-12):.1f} dB)")
        print(f"Crest Factor:    {crest_factor:.2f}")
        print(f"THD+N:           {self.calculate_thd(audio_data):.3f}%")
        print(f"Calibration:     {'Applied' if self.calibration_data else 'Not applied'}")
        
        # Frequency domain statistics
        fft_data = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
        positive_freq_mask = frequencies > 0
        magnitude_spectrum = np.abs(fft_data[positive_freq_mask])
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitude_spectrum)
        dominant_freq = frequencies[positive_freq_mask][dominant_freq_idx]
        
        print(f"Dominant Freq:   {dominant_freq:.1f} Hz")
        print("="*50)
    
    def interactive_mode(self):
        """Run the recorder in interactive mode"""
        print("MINIDSP UMIK-1 Recording and Analysis Program")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. List audio devices")
            print("2. Select device")
            print("3. Load calibration file")
            print("4. Configure recording")
            print("5. Record audio")
            print("6. Save recording")
            print("7. Analyze recording")
            print("8. Full workflow (device + record + analyze)")
            print("9. Exit")
            
            try:
                choice = input("\nEnter choice (1-9): ").strip()
                
                if choice == '1':
                    self.list_devices()
                
                elif choice == '2':
                    self.select_device()
                
                elif choice == '3':
                    self.load_calibration_file()
                
                elif choice == '4':
                    self.configure_recording()
                
                elif choice == '5':
                    self.record_audio()
                
                elif choice == '6':
                    self.save_recording()
                
                elif choice == '7':
                    self.analyze_recording()
                
                elif choice == '8':
                    print("\n--- Full Workflow ---")
                    if self.select_device():
                        self.load_calibration_file()
                        self.configure_recording()
                        if self.record_audio():
                            self.save_recording()
                            self.analyze_recording()
                
                elif choice == '9':
                    print("Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please enter 1-9.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='MINIDSP UMIK-1 Recording and Analysis Program')
    parser.add_argument('--device', '-d', type=int, help='Audio device ID')
    parser.add_argument('--calibration', '-c', type=str, help='Calibration file path')
    parser.add_argument('--sample-rate', '-s', type=int, choices=[44100, 48000, 96000], 
                       default=48000, help='Sample rate (Hz)')
    parser.add_argument('--duration', '-t', type=float, default=10.0, help='Recording duration (seconds)')
    parser.add_argument('--output', '-o', type=str, help='Output WAV file name')
    parser.add_argument('--auto', '-a', action='store_true', help='Auto mode: detect UMIK-1 and run full workflow')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Create recorder instance
    recorder = UMIK1RecorderCLI()
    
    if args.interactive:
        recorder.interactive_mode()
        return
    
    if args.auto:
        print("Auto mode: Detecting UMIK-1 and running full workflow...")
        
        # List devices and auto-detect UMIK-1
        devices = recorder.list_devices()
        
        if recorder.device_id is None:
            print("UMIK-1 not detected automatically. Please specify device manually.")
            return
        
        # Load calibration if specified
        if args.calibration:
            if not recorder.load_calibration_file(args.calibration):
                print("Continuing without calibration...")
        
        # Configure recording
        recorder.configure_recording(args.sample_rate, args.duration)
        
        # Record and analyze
        if recorder.record_audio():
            recorder.save_recording(args.output)
            recorder.analyze_recording()
        
    else:
        # Manual command line mode
        if args.device is not None:
            if not recorder.select_device(args.device):
                return
        else:
            recorder.list_devices()
            print("Please specify a device with --device or use --interactive mode")
            return
        
        if args.calibration:
            recorder.load_calibration_file(args.calibration)
        
        recorder.configure_recording(args.sample_rate, args.duration)
        
        if recorder.record_audio():
            recorder.save_recording(args.output)
            recorder.analyze_recording()


if __name__ == "__main__":
    # Check for required dependencies
    required_packages = ['numpy', 'matplotlib', 'sounddevice', 'scipy', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        main()
