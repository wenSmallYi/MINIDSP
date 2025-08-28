#!/usr/bin/env python3
"""
MINIDSP UMIK-1 Recording Server

A FastAPI-based server for controlling UMIK-1 recordings via REST API.
Calibration is loaded at startup and applied to all recordings.

Features:
- REST API for start/stop recording control
- Pre-loaded calibration file
- Fixed recording parameters (48kHz sample rate)
- Automatic saving of raw audio and analysis results
- Background recording management

Author: Audio Recording Server
Date: 2025-01-28
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal
from scipy.io import wavfile
import pandas as pd
import os
import json
from datetime import datetime
import threading
import time
import asyncio
from typing import Optional, Dict, Any
import uuid
import base64
from io import BytesIO

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn


class RecordingRequest(BaseModel):
    """Request model for starting recording"""
    duration: Optional[float] = None  # If None, record until stop command
    session_id: Optional[str] = None


class RecordingResponse(BaseModel):
    """Response model for recording operations"""
    success: bool
    message: str
    session_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class UMIK1Server:
    """
    MINIDSP UMIK-1 recording server with API control
    """
    
    def __init__(self, calibration_file: str = None, device_id: int = None):
        """Initialize the server with fixed parameters"""
        
        # Fixed recording parameters
        self.sample_rate = 48000  # Hz - fixed as requested
        self.channels = 1         # mono recording
        self.dtype = np.float32
        
        # Device settings
        self.device_id = device_id
        self.device_name = None
        
        # Calibration data (loaded at startup)
        self.calibration_data = None
        self.calibration_file = calibration_file
        
        # Recording management
        self.active_recordings = {}  # session_id -> recording_info
        self.completed_recordings = {}  # session_id -> analysis_results
        
        # Server state
        self.is_initialized = False
        
        # Create output directories (relative to project root)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.recordings_dir = os.path.join(self.project_root, "recordings")
        self.analysis_dir = os.path.join(self.project_root, "analysis")
        self.plots_dir = os.path.join(self.project_root, "plots")
        self.calibration_dir = os.path.join(self.project_root, "calibration")
        
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.calibration_dir, exist_ok=True)
        
        print("UMIK-1 Recording Server initializing...")
        self._initialize_server()
    
    def _initialize_server(self):
        """Initialize server components"""
        try:
            # Auto-detect and select UMIK-1 device
            if not self._setup_audio_device():
                raise Exception("Failed to setup audio device")
            
            # Load calibration file if provided
            if self.calibration_file:
                if not self._load_calibration():
                    print("Warning: Failed to load calibration file, continuing without calibration")
            
            self.is_initialized = True
            print(f"Server initialized successfully!")
            print(f"Audio Device: {self.device_name}")
            print(f"Sample Rate: {self.sample_rate} Hz")
            print(f"Calibration: {'Loaded' if self.calibration_data else 'Not loaded'}")
            
        except Exception as e:
            print(f"Server initialization failed: {e}")
            raise
    
    def _setup_audio_device(self):
        """Setup audio device (auto-detect UMIK-1 or use specified device)"""
        try:
            devices = sd.query_devices()
            
            if self.device_id is not None:
                # Use specified device
                device_info = sd.query_devices(self.device_id)
                if device_info['max_input_channels'] > 0:
                    self.device_name = device_info['name']
                    print(f"Using specified device {self.device_id}: {self.device_name}")
                    return True
                else:
                    print(f"Specified device {self.device_id} has no input channels")
                    return False
            
            # Auto-detect UMIK-1
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    if 'umik' in device['name'].lower() or 'minidsp' in device['name'].lower():
                        self.device_id = i
                        self.device_name = device['name']
                        print(f"Auto-detected UMIK-1: {self.device_name}")
                        return True
            
            # If no UMIK-1 found, use default input device
            default_device = sd.query_devices(kind='input')
            if default_device['max_input_channels'] > 0:
                self.device_id = sd.default.device[0]  # Default input device
                self.device_name = default_device['name']
                print(f"Using default input device: {self.device_name}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error setting up audio device: {e}")
            return False
    
    def _load_calibration(self):
        """Load calibration file at startup"""
        try:
            if not os.path.exists(self.calibration_file):
                print(f"Calibration file not found: {self.calibration_file}")
                return False
            
            # Handle UMIK-1 calibration files with header line
            # These files typically have a quoted header and tab-separated data
            try:
                # Try loading with skiprows=1 and tab separator
                data = pd.read_csv(self.calibration_file, skiprows=1, sep='\t', header=None)
                
                if len(data.columns) >= 2:
                    frequencies = data.iloc[:, 0].values
                    magnitudes = data.iloc[:, 1].values
                    
                    self.calibration_data = {
                        'frequencies': frequencies,
                        'magnitudes': magnitudes
                    }
                    
                    print(f"Calibration loaded: {len(frequencies)} points, "
                          f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz")
                    return True
                else:
                    print("Calibration file must have at least 2 columns")
                    return False
                    
            except Exception as inner_e:
                print(f"Error parsing calibration file: {inner_e}")
                
                # Fallback: manual parsing
                with open(self.calibration_file, 'r') as f:
                    lines = f.readlines()
                
                frequencies = []
                magnitudes = []
                
                for line in lines[1:]:  # Skip first line
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                freq = float(parts[0])
                                mag = float(parts[1])
                                frequencies.append(freq)
                                magnitudes.append(mag)
                            except ValueError:
                                continue
                
                if len(frequencies) > 0:
                    self.calibration_data = {
                        'frequencies': np.array(frequencies),
                        'magnitudes': np.array(magnitudes)
                    }
                    print(f"Calibration loaded (manual): {len(frequencies)} points, "
                          f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz")
                    return True
                else:
                    print("No valid calibration data found")
                    return False
                
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def start_recording(self, duration: float = None, session_id: str = None) -> Dict[str, Any]:
        """Start a new recording session"""
        if not self.is_initialized:
            return {"success": False, "message": "Server not initialized"}
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Check if session already exists
        if session_id in self.active_recordings:
            return {"success": False, "message": f"Session {session_id} already active"}
        
        try:
            # Create recording info
            recording_info = {
                'session_id': session_id,
                'start_time': datetime.now(),
                'duration': duration,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'status': 'starting',
                'thread': None,
                'audio_data': None,
                'stop_flag': False
            }
            
            self.active_recordings[session_id] = recording_info
            
            # Start recording in background thread
            recording_thread = threading.Thread(
                target=self._record_audio_background,
                args=(session_id,)
            )
            recording_info['thread'] = recording_thread
            recording_thread.start()
            
            return {
                "success": True,
                "message": f"Recording started",
                "session_id": session_id,
                "data": {
                    "sample_rate": self.sample_rate,
                    "duration": duration,
                    "device": self.device_name
                }
            }
            
        except Exception as e:
            if session_id in self.active_recordings:
                del self.active_recordings[session_id]
            return {"success": False, "message": f"Failed to start recording: {str(e)}"}
    
    def stop_recording(self, session_id: str) -> Dict[str, Any]:
        """Stop an active recording session"""
        if session_id not in self.active_recordings:
            return {"success": False, "message": f"Session {session_id} not found"}
        
        try:
            recording_info = self.active_recordings[session_id]
            recording_info['stop_flag'] = True
            recording_info['status'] = 'stopping'
            
            # Wait for recording thread to complete (with timeout)
            if recording_info['thread'] and recording_info['thread'].is_alive():
                recording_info['thread'].join(timeout=5.0)
            
            return {
                "success": True,
                "message": f"Recording stopped",
                "session_id": session_id
            }
            
        except Exception as e:
            return {"success": False, "message": f"Failed to stop recording: {str(e)}"}
    
    def _record_audio_background(self, session_id: str):
        """Background recording function"""
        recording_info = self.active_recordings[session_id]
        
        try:
            recording_info['status'] = 'recording'
            
            if recording_info['duration']:
                # Fixed duration recording
                num_samples = int(self.sample_rate * recording_info['duration'])
                
                audio_data = sd.rec(
                    frames=num_samples,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    device=self.device_id
                )
                
                # Wait for recording to complete or stop flag
                start_time = time.time()
                while time.time() - start_time < recording_info['duration']:
                    if recording_info['stop_flag']:
                        sd.stop()
                        break
                    time.sleep(0.1)
                
                sd.wait()
                
            else:
                # Continuous recording until stop
                chunk_duration = 1.0  # 1 second chunks
                chunk_samples = int(self.sample_rate * chunk_duration)
                audio_chunks = []
                
                while not recording_info['stop_flag']:
                    chunk = sd.rec(
                        frames=chunk_samples,
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype=self.dtype,
                        device=self.device_id
                    )
                    sd.wait()
                    audio_chunks.append(chunk.flatten())
                
                # Concatenate all chunks
                if audio_chunks:
                    audio_data = np.concatenate(audio_chunks).reshape(-1, 1)
                else:
                    audio_data = np.array([]).reshape(-1, 1)
            
            # Store audio data
            recording_info['audio_data'] = audio_data.flatten()
            recording_info['end_time'] = datetime.now()
            recording_info['status'] = 'completed'
            
            # Process and save results
            self._process_recording(session_id)
            
        except Exception as e:
            recording_info['status'] = 'error'
            recording_info['error'] = str(e)
            print(f"Recording error for session {session_id}: {e}")
        
        finally:
            # Move from active to completed
            if session_id in self.active_recordings:
                completed_info = self.active_recordings.pop(session_id)
                self.completed_recordings[session_id] = completed_info
    
    def _process_recording(self, session_id: str):
        """Process recording: save raw data and perform analysis"""
        recording_info = self.completed_recordings.get(session_id) or self.active_recordings.get(session_id)
        if not recording_info or recording_info['audio_data'] is None:
            return
        
        try:
            timestamp = recording_info['start_time'].strftime("%Y%m%d_%H%M%S")
            base_filename = f"umik1_{session_id}_{timestamp}"
            
            # Save raw audio data
            raw_filename = os.path.join(self.recordings_dir, f"{base_filename}.wav")
            audio_int16 = (recording_info['audio_data'] * 32767).astype(np.int16)
            wavfile.write(raw_filename, self.sample_rate, audio_int16)
            
            # Apply calibration if available
            calibrated_audio = recording_info['audio_data'].copy()
            if self.calibration_data:
                calibrated_audio = self._apply_calibration(calibrated_audio)
            
            # Perform analysis
            analysis_results = self._analyze_audio(calibrated_audio, session_id, timestamp)
            
            # Save analysis results
            analysis_filename = os.path.join(self.analysis_dir, f"{base_filename}_analysis.json")
            with open(analysis_filename, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            # Create and save plots
            self._create_analysis_plots(calibrated_audio, base_filename)
            
            # Update recording info
            recording_info['files'] = {
                'raw_audio': raw_filename,
                'analysis': analysis_filename,
                'plots': os.path.join(self.plots_dir, f"{base_filename}_analysis.png")
            }
            recording_info['analysis_results'] = analysis_results
            
            print(f"Processing completed for session {session_id}")
            
        except Exception as e:
            print(f"Error processing recording {session_id}: {e}")
            recording_info['processing_error'] = str(e)
    
    def _apply_calibration(self, audio_data):
        """Apply microphone calibration to audio data"""
        if not self.calibration_data:
            return audio_data
        
        try:
            # Compute FFT
            fft_data = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            
            # Interpolate calibration data to all frequency bins
            cal_freqs = self.calibration_data['frequencies']
            cal_mags = self.calibration_data['magnitudes']
            
            # Create calibration correction for all frequencies
            cal_correction = np.ones(len(frequencies))
            
            # Only apply calibration to positive frequencies within calibration range
            for i, freq in enumerate(frequencies):
                abs_freq = abs(freq)  # Use absolute frequency for negative frequencies
                if abs_freq >= cal_freqs[0] and abs_freq <= cal_freqs[-1] and abs_freq > 0:
                    cal_db = np.interp(abs_freq, cal_freqs, cal_mags)
                    cal_correction[i] = 10**(cal_db / 20)
            
            # Apply calibration
            fft_calibrated = fft_data / cal_correction
            
            # Convert back to time domain
            calibrated_audio = np.real(np.fft.ifft(fft_calibrated))
            
            return calibrated_audio
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return audio_data
    
    def _analyze_audio(self, audio_data, session_id, timestamp):
        """Perform comprehensive audio analysis"""
        try:
            # Basic statistics
            duration = len(audio_data) / self.sample_rate
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            crest_factor = peak / rms if rms > 0 else 0
            
            # Frequency domain analysis
            fft_data = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            positive_freq_mask = frequencies > 0
            magnitude_spectrum = np.abs(fft_data[positive_freq_mask])
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(magnitude_spectrum)
            dominant_freq = frequencies[positive_freq_mask][dominant_freq_idx]
            
            # THD+N calculation
            thd_plus_n = self._calculate_thd(audio_data)
            
            # 1/3 Octave band analysis
            octave_freqs, octave_levels = self._calculate_octave_bands(audio_data)
            
            # Compile results
            analysis_results = {
                'session_id': session_id,
                'timestamp': timestamp,
                'duration': duration,
                'sample_rate': self.sample_rate,
                'calibrated': self.calibration_data is not None,
                'statistics': {
                    'rms_linear': float(rms),
                    'rms_db': float(20 * np.log10(rms + 1e-12)),
                    'peak_linear': float(peak),
                    'peak_db': float(20 * np.log10(peak + 1e-12)),
                    'crest_factor': float(crest_factor),
                    'thd_plus_n_percent': float(thd_plus_n),
                    'dominant_frequency_hz': float(dominant_freq)
                },
                'octave_bands': {
                    'frequencies': octave_freqs.tolist(),
                    'levels_db': octave_levels
                }
            }
            
            return analysis_results
            
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def _calculate_thd(self, audio_data):
        """Calculate Total Harmonic Distortion + Noise"""
        try:
            fft_data = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            
            positive_freq_mask = frequencies > 0
            magnitude_spectrum = np.abs(fft_data[positive_freq_mask])
            positive_frequencies = frequencies[positive_freq_mask]
            
            # Find fundamental frequency
            valid_range = (positive_frequencies >= 20) & (positive_frequencies <= 2000)
            if np.any(valid_range):
                fund_idx = np.argmax(magnitude_spectrum[valid_range])
                fund_freq = positive_frequencies[valid_range][fund_idx]
                
                total_power = np.sum(magnitude_spectrum**2)
                fund_bin = np.argmin(np.abs(positive_frequencies - fund_freq))
                fund_power = magnitude_spectrum[fund_bin]**2
                
                thd_plus_n = np.sqrt((total_power - fund_power) / fund_power) * 100
                return min(thd_plus_n, 100)
            else:
                return 0
        except:
            return 0
    
    def _calculate_octave_bands(self, audio_data):
        """Calculate 1/3 octave band levels"""
        octave_freqs = np.array([
            25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
            500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000, 12500, 16000, 20000
        ])
        
        octave_levels = []
        
        for fc in octave_freqs:
            if fc > self.sample_rate / 2:
                break
                
            flow = fc / (2**(1/6))
            fhigh = fc * (2**(1/6))
            
            try:
                sos = scipy.signal.butter(4, [flow, fhigh], btype='band', 
                                        fs=self.sample_rate, output='sos')
                filtered = scipy.signal.sosfilt(sos, audio_data)
                rms = np.sqrt(np.mean(filtered**2))
                level_db = 20 * np.log10(rms + 1e-12)
                octave_levels.append(level_db)
            except:
                octave_levels.append(-120.0)  # Very low level for failed bands
        
        return octave_freqs[:len(octave_levels)], octave_levels
    
    def _create_analysis_plots(self, audio_data, base_filename):
        """Create and save analysis plots"""
        try:
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'UMIK-1 Analysis - {base_filename}', fontsize=16, fontweight='bold')
            
            # Time domain
            time_vector = np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data))
            ax1.plot(time_vector, audio_data, 'b-', linewidth=0.5)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Time Domain')
            ax1.grid(True, alpha=0.3)
            
            # RMS over time
            window_size = int(0.1 * self.sample_rate)
            rms_values = []
            rms_times = []
            
            for i in range(0, len(audio_data) - window_size, window_size//2):
                window = audio_data[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(20 * np.log10(rms + 1e-12))
                rms_times.append(i / self.sample_rate)
            
            ax2.plot(rms_times, rms_values, 'r-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('RMS Level (dB)')
            ax2.set_title('RMS Level Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Frequency domain - Linear scale
            fft_data = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            positive_freq_mask = frequencies >= 0
            frequencies = frequencies[positive_freq_mask]
            magnitude_spectrum = np.abs(fft_data[positive_freq_mask])
            
            ax3.plot(frequencies, magnitude_spectrum, 'g-', linewidth=1)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Magnitude (Linear)')
            ax3.set_title('Frequency Spectrum (Linear)')
            ax3.set_xlim(0, self.sample_rate/2)
            ax3.grid(True, alpha=0.3)
            
            # Frequency domain - Log scale
            magnitude_db = 20 * np.log10(magnitude_spectrum + 1e-12)
            ax4.semilogx(frequencies[1:], magnitude_db[1:], 'purple', linewidth=1)
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Magnitude (dB)')
            ax4.set_title('Frequency Spectrum (Log)')
            ax4.set_xlim(20, self.sample_rate/2)
            ax4.grid(True, alpha=0.3)
            
            # Add info
            cal_status = "Calibrated" if self.calibration_data else "Uncalibrated"
            fig.text(0.02, 0.02, f"Status: {cal_status} | Sample Rate: {self.sample_rate} Hz", 
                    fontsize=10, style='italic')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = os.path.join(self.plots_dir, f"{base_filename}_analysis.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a recording session"""
        # Check active recordings
        if session_id in self.active_recordings:
            info = self.active_recordings[session_id]
            return {
                "success": True,
                "session_id": session_id,
                "status": info['status'],
                "start_time": info['start_time'].isoformat(),
                "duration": info.get('duration'),
                "sample_rate": info['sample_rate']
            }
        
        # Check completed recordings
        if session_id in self.completed_recordings:
            info = self.completed_recordings[session_id]
            return {
                "success": True,
                "session_id": session_id,
                "status": info['status'],
                "start_time": info['start_time'].isoformat(),
                "end_time": info.get('end_time', '').isoformat() if info.get('end_time') else None,
                "duration": len(info['audio_data']) / self.sample_rate if info.get('audio_data') is not None else None,
                "files": info.get('files', {}),
                "analysis_results": info.get('analysis_results', {})
            }
        
        return {"success": False, "message": f"Session {session_id} not found"}
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all recording sessions"""
        active = list(self.active_recordings.keys())
        completed = list(self.completed_recordings.keys())
        
        return {
            "success": True,
            "active_sessions": active,
            "completed_sessions": completed,
            "total_sessions": len(active) + len(completed)
        }


# Initialize global server instance
server_instance = None


# FastAPI app
app = FastAPI(title="UMIK-1 Recording Server", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    global server_instance
    
    # Fixed calibration file path - relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calibration_file = os.path.join(project_root, "calibration", "7163752.txt")
    device_id = os.getenv("UMIK1_DEVICE_ID")  # Optional
    
    if device_id:
        device_id = int(device_id)
    
    server_instance = UMIK1Server(calibration_file=calibration_file, device_id=device_id)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "UMIK-1 Recording Server", "status": "running"}


@app.get("/status")
async def server_status():
    """Get server status"""
    if not server_instance or not server_instance.is_initialized:
        return {"status": "not_initialized", "error": "Server not properly initialized"}
    
    return {
        "status": "ready",
        "device": server_instance.device_name,
        "sample_rate": server_instance.sample_rate,
        "calibration_loaded": server_instance.calibration_data is not None,
        "active_sessions": len(server_instance.active_recordings),
        "completed_sessions": len(server_instance.completed_recordings)
    }


@app.post("/start_recording")
async def start_recording(request: RecordingRequest):
    """Start a new recording"""
    if not server_instance or not server_instance.is_initialized:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    result = server_instance.start_recording(
        duration=request.duration,
        session_id=request.session_id
    )
    
    if result["success"]:
        return RecordingResponse(**result)
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@app.post("/stop_recording/{session_id}")
async def stop_recording(session_id: str):
    """Stop an active recording"""
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    result = server_instance.stop_recording(session_id)
    
    if result["success"]:
        return RecordingResponse(**result)
    else:
        raise HTTPException(status_code=404, detail=result["message"])


@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get status of a specific session"""
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    result = server_instance.get_session_status(session_id)
    
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=404, detail=result["message"])


@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    return server_instance.list_sessions()


@app.get("/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    """Download files (raw, analysis, plots)"""
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    if session_id not in server_instance.completed_recordings:
        raise HTTPException(status_code=404, detail="Session not found or not completed")
    
    session_info = server_instance.completed_recordings[session_id]
    files = session_info.get('files', {})
    
    if file_type not in files:
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not found")
    
    file_path = files[file_type]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path)
    )


if __name__ == "__main__":
    # Configuration
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = 8000       # Default port
    
    print("Starting UMIK-1 Recording Server...")
    print(f"Server will be available at: http://{HOST}:{PORT}")
    print("\nAPI Endpoints:")
    print("  GET  /              - Server info")
    print("  GET  /status        - Server status")
    print("  POST /start_recording - Start recording")
    print("  POST /stop_recording/{session_id} - Stop recording")
    print("  GET  /session/{session_id} - Session status")
    print("  GET  /sessions      - List all sessions")
    print("  GET  /download/{session_id}/{file_type} - Download files")
    print("\nExample usage:")
    print("  curl -X POST http://localhost:8000/start_recording -H 'Content-Type: application/json' -d '{\"duration\": 10}'")
    print("  curl -X POST http://localhost:8000/stop_recording/SESSION_ID")
    
    # Run server
    uvicorn.run(app, host=HOST, port=PORT)
