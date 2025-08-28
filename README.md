# UMIK-1 Recording Project

A complete MINIDSP UMIK-1 recording and analysis system with multiple interfaces and professional audio measurement capabilities.

## Project Structure

```
UMIK1_Project/
├── src/                          # Source code
│   ├── umik1_recorder_cli.py     # Command line interface
│   └── umik1_server.py           # Web API server
├── recordings/                   # Audio recordings (WAV files)
├── analysis/                     # Analysis results (JSON files)
├── plots/                        # Generated plots (PNG files)
├── calibration/                  # Calibration files
│   └── 7163752.txt              # UMIK-1 calibration file
├── run.py                        # Project launcher script
└── README.md                     # This file
```

## Features

### Core Functionality
- **Professional Audio Recording**: 48kHz sample rate recording using MINIDSP UMIK-1
- **Microphone Calibration**: Automatic loading and application of official calibration files
- **Comprehensive Analysis**: Time domain, frequency domain, RMS analysis
- **Multi-format Visualization**: Linear and logarithmic frequency plots
- **Data Export**: WAV audio files, JSON analysis results, PNG plots

### Two Operation Modes

#### 1. Command Line Interface (CLI)
Interactive command-line program for direct control and immediate feedback.

**Features:**
- Interactive device selection
- Real-time parameter configuration
- Live recording feedback
- Immediate analysis and plotting
- Manual calibration file loading

#### 2. Web API Server
FastAPI-based server for remote control and integration with other systems.

**Features:**
- REST API endpoints for all operations
- Background recording management
- Session-based recording control
- File download endpoints
- Automatic processing pipeline

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install numpy matplotlib sounddevice scipy pandas fastapi uvicorn pydantic
```

### Quick Start

1. **Clone/Download the project to your workspace**

2. **Run CLI Interface:**
```bash
cd UMIK1_Project
python run.py cli
```

3. **Run Web Server:**
```bash
cd UMIK1_Project
python run.py server
```

## Usage Examples

### CLI Mode
```bash
# Interactive mode with device auto-detection
python run.py cli

# Direct recording with parameters
python src/umik1_recorder_cli.py --duration 30 --sample-rate 48000 --calibration calibration/7163752.txt
```

### Server Mode API Examples
```bash
# Start server
python run.py server

# Check server status
curl http://localhost:8000/status

# Start recording
curl -X POST http://localhost:8000/start_recording \
  -H 'Content-Type: application/json' \
  -d '{"duration": 10}'

# Stop recording
curl -X POST http://localhost:8000/stop_recording/SESSION_ID

# Download results
curl http://localhost:8000/download/SESSION_ID/raw_audio > recording.wav
curl http://localhost:8000/download/SESSION_ID/analysis > analysis.json
curl http://localhost:8000/download/SESSION_ID/plots > plots.png
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server information |
| GET | `/status` | Server status and capabilities |
| POST | `/start_recording` | Start new recording session |
| POST | `/stop_recording/{session_id}` | Stop active recording |
| GET | `/session/{session_id}` | Get session status |
| GET | `/sessions` | List all sessions |
| GET | `/download/{session_id}/{file_type}` | Download files |

## Configuration

### Calibration Setup
1. Place your UMIK-1 calibration file in the `calibration/` directory
2. Update the filename in `src/umik1_server.py` if different from `7163752.txt`
3. CLI mode allows runtime calibration file selection

### Recording Parameters
- **Sample Rate**: 48kHz (optimized for UMIK-1)
- **Channels**: Mono (1 channel)
- **Format**: 32-bit float (internal), 16-bit WAV (export)
- **Duration**: Configurable (default 10 seconds, or continuous until stop)

## Output Files

### Audio Files (`recordings/`)
- Format: WAV, 16-bit, 48kHz
- Naming: `umik1_[session_id]_[timestamp].wav`

### Analysis Files (`analysis/`)
- Format: JSON with comprehensive measurements
- Contents: RMS levels, frequency analysis, peak detection, statistics

### Plot Files (`plots/`)
- Format: PNG, 300 DPI
- Contents: Time domain, RMS over time, frequency spectrum (linear & log)

## Troubleshooting

### Common Issues
1. **Device Not Found**: Ensure UMIK-1 is connected and recognized by the system
2. **Permission Errors**: Run with appropriate audio device permissions
3. **Calibration Loading**: Check file path and format (tab-separated values)
4. **Server Port Conflicts**: Default port 8000, change if needed

### macOS Specific
- Use Homebrew Python if system Python lacks audio support
- Grant microphone permissions in System Preferences > Security & Privacy

## Technical Details

### Architecture
- **Audio Backend**: sounddevice (PortAudio)
- **Signal Processing**: SciPy
- **Visualization**: Matplotlib (non-GUI backend for server)
- **Web Framework**: FastAPI with Uvicorn
- **Data Format**: NumPy arrays, Pandas DataFrames

### Performance
- Real-time recording capability
- Non-blocking server operations
- Efficient FFT processing
- Automatic memory management

## License & Credits

This project is designed for professional audio measurement and analysis using the MINIDSP UMIK-1 microphone system.

**Created**: January 2025  
**Author**: Audio Recording System  
**Version**: 1.0.0
