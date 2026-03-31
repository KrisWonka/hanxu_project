# Hardware Setup Guide - Orange Pi 5 + SIM7600CE

## Required Hardware

1. **Orange Pi 5** (RK3588S, Ubuntu 22.04.5 LTS)
2. **SIM7600CE 4G Module** (USB interface, ~150 RMB)
3. **USB Microphone** (any USB mic or USB sound card + 3.5mm mic)
4. **Speaker** (3.5mm or USB)
5. **Physical Button** (normally-open momentary switch, connected to GPIO)
6. **LED** (optional, for status indication)
7. **SIM Card** (standard China Mobile/Unicom/Telecom SIM with voice plan)

## Wiring

### Button (GPIO)
- Button pin 1 → GPIO pin 7 (gpiochip0, configurable in settings.yaml)
- Button pin 2 → GND

### LED (GPIO)
- LED anode (+) → GPIO pin 11 via 330Ω resistor
- LED cathode (-) → GND

### SIM7600CE
- Connect via USB cable to Orange Pi
- Insert SIM card into the module's SIM slot
- After connecting, should appear as `/dev/ttyUSB0`, `/dev/ttyUSB1`, `/dev/ttyUSB2`
- Voice AT commands typically use `/dev/ttyUSB2`

## First Boot Setup

```bash
# 1. Clone the project
git clone https://github.com/KrisWonka/hanxu_project.git
cd hanxu_project

# 2. Install system dependencies
sudo apt update && sudo apt install -y python3-pip ffmpeg libportaudio2

# 3. Install Python dependencies
pip3 install --user -r requirements.txt

# 4. Create .env with API key
cp .env.example .env
nano .env  # fill in DEEPSEEK_API_KEY

# 5. Edit config for production
nano config/settings.yaml
# Change: mode: prod
# Change: telephony.modem_mode: real
# Verify: telephony.serial_port matches your ttyUSB device

# 6. Test the 4G module
python3 -c "
from src.telephony.modem import Modem
m = Modem(mode='real', port='/dev/ttyUSB2')
print('Signal:', m.check_signal())
print('Registration:', m.check_registration())
m.close()
"

# 7. Test microphone
python3 -m sounddevice  # should list USB mic
python3 -c "
from src.audio.recorder import Recorder
r = Recorder()
print('Recording 3 seconds...')
import sounddevice as sd
sd.sleep(3000)
print('Mic works!')
"

# 8. Run the agent
python3 src/main.py

# 9. Install as system service (auto-start on boot)
sudo cp deploy/voice-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable voice-agent
sudo systemctl start voice-agent
```

## Troubleshooting

### SIM7600CE not detected
```bash
ls /dev/ttyUSB*          # Should show ttyUSB0/1/2
dmesg | grep -i usb      # Check kernel messages
sudo usermod -a -G dialout $USER  # Add user to dialout group
```

### No audio devices
```bash
arecord -l               # List recording devices
aplay -l                 # List playback devices
python3 -m sounddevice   # List sounddevice devices
```

### GPIO permission denied
```bash
sudo usermod -a -G gpio $USER
# Or use sudo for testing
```

## LED Status Indicators

| LED State | Meaning |
|-----------|---------|
| Off | Idle, waiting for button press |
| Solid On | Recording / Speaking |
| Blinking | Processing (STT / LLM) |
