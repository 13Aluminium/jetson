# X Landing — Flight Scripts (MAVProxy Architecture)

## Architecture

```
┌─────────────┐     USB-C      ┌──────────┐    UDP 14551    ┌──────────────┐
│  Pixhawk 6C │ ◄────────────► │ MAVProxy │ ──────────────► │ Your Script  │
│  (ArduPilot)│                │ Terminal 1│                 │  Terminal 2  │
└─────────────┘                └──────────┘                 └──────────────┘
                                    │
                              You can type
                              "mode RTL" here
                              as emergency!
```

**Why this design:** MAVProxy is battle-tested, handles heartbeats and data streams,
and you can type emergency commands directly. Same scripts work for SITL testing
and real flight — just change what MAVProxy connects to.

## Setup

### On Jetson (real flight)
```bash
# Install dependencies
pip3 install pymavlink MAVProxy

# Copy scripts
scp flight_utils.py 1_*.py 2_*.py 3_*.py 4_*.py 5_*.py jetson@<ip>:~/

# Make sure best_22.pt is in ~/
```

### SITL Testing (Mac/Linux) — DO THIS FIRST!
```bash
# One-time: Install ArduPilot SITL
# https://ardupilot.org/dev/docs/SITL-setup-landingpage.html

# For Mac:
brew install gcc-arm-none-eabi
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-mac.sh
. ~/.profile

# Build + run SITL:
cd ArduCopter
sim_vehicle.py --console --map --out=udp:127.0.0.1:14551
```

## Running

### SITL Testing (safe, no hardware)
```bash
# Terminal 1 — SITL + MAVProxy
cd ~/ardupilot/ArduCopter
sim_vehicle.py --console --map --out=udp:127.0.0.1:14551

# Terminal 2 — Your script
python3 4_fly_and_move.py --sitl    # test movement commands
python3 1_fly_and_record.py --sitl  # test takeoff/RTL
```

### Real Flight (on Jetson via SSH)
```bash
# SSH Terminal 1 — MAVProxy
mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \
            --out=udp:127.0.0.1:14551

# SSH Terminal 2 — Script
python3 1_fly_and_record.py
```

If the port isn't /dev/ttyACM0, check with `ls /dev/ttyACM* /dev/ttyUSB*`

## Script Order (DO NOT SKIP)

| # | Script | Tests | Camera? |
|---|--------|-------|---------|
| 1 | `1_fly_and_record.py` | Takeoff, hover, RTL, video | Yes |
| 2 | `2_fly_and_detect.py` | YOLO X detection in flight | Yes |
| 3 | `3_fly_and_guide.py` | Offset→meters math (NO movement) | Yes |
| 4 | `4_fly_and_move.py` | GUIDED velocity commands | No |
| 5 | `5_land_on_x.py` | Full autonomous X landing | Yes |

**Test each in SITL first, then with --dry-run on real hardware, then for real.**

## Emergency

At ANY time, type in the MAVProxy terminal:
```
mode RTL        ← return to launch
mode LAND       ← land immediately
disarm           ← kill motors (USE WITH CAUTION — drone will fall!)
```

Or Ctrl+C in the script terminal → triggers RTL automatically.

## Failsafes Built In

- Ctrl+C → RTL
- Any Python exception → RTL
- Lost X for 10s during descent → RTL
- Search timeout (60s) → RTL
- Pre-flight check: GPS, battery, position must pass before arming

## Key Parameters

- Default altitude: **5 meters** (~16ft)
- Override: `--alt 3` for 3 meters
- YOLO model: `--weights best_22.pt` (default)
- Confidence: `--conf 0.50` (default)
