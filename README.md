# ChArUco Camera Calibration

A Python tool for camera calibration using ChArUco boards with export to MicMac and Meshroom formats.

## Features

- **ChArUco Detection**: Robust corner detection using ArUco markers
- **Camera Calibration**: High-precision calibration with OpenCV
- **MicMac Export**: Fraser model compatible with MicMac photogrammetry software
- **Meshroom Export**: JSON format for AliceVision/Meshroom
- **Configurable Parameters**: Adjustable corner detection thresholds and board configurations

## Installation

```bash
pip install opencv-python numpy
```

## Usage

### Basic Calibration

```bash
python camera_calibrator.py images_folder --square-size 2.0 --min-corners-percent 30
```

### Export to MicMac

```bash
python camera_calibrator.py images_folder --square-size 2.0 --export-formats micmac
```

### Export to Both Formats

```bash
python camera_calibrator.py images_folder --square-size 2.0 --export-formats both
```

## Parameters

- `--square-size`: Size of squares in cm (default: 2.0)
- `--squares-x`: Number of squares in X direction (default: 11)
- `--squares-y`: Number of squares in Y direction (default: 8)
- `--marker-ratio`: Ratio of marker to square size (default: 0.7)
- `--min-corners-percent`: Minimum percentage of corners detected (default: 30)
- `--export-formats`: Export formats: `micmac`, `meshroom`, or `both`

## Output Files

- **JSON**: Complete calibration results
- **XML**: MicMac Fraser model format
- **JSON**: Meshroom sensor format

## MicMac Integration

The exported XML file is compatible with MicMac's Fraser model and can be used directly with:

```bash
mm3d Tapas Figee ".*jpg" InCal=calibration.xml
```

## Requirements

- Python 3.7+
- OpenCV 4.0+
- NumPy

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.