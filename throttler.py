#!/usr/bin/env python3
"""
Ultimate Thermal-Aware Storage I/O Throttler
Monitors storage device temperatures and dynamically throttles I/O throughput
to prevent overheating, prolong device lifespan, and reduce failure rates.

Combines best practices from multiple implementations for maximum robustness.
"""

import os
import sys
import json
import time
import math
import signal
import logging
import argparse
import subprocess
import threading
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

# --- Configuration ---
DEFAULT_CONFIG = {
    "cache_dir": "/var/cache/storage_thermals",
    "log_file": "/var/log/storage_thermals.log",
    "cgroup_path": "/sys/fs/cgroup/storage_throttle",
    "monitor_interval": 30,
    "target_temp": 45,      # Start throttling here
    "critical_temp": 55,    # Maximum throttling here  
    "hysteresis": 3,        # Stop throttling at target_temp - hysteresis
    "min_speed_percent": 10, # Minimum speed when critically hot
    "sigmoid_steepness": 8.0,
    "benchmark_size_mb": 512,
    "benchmark_expiry_days": 90,
}

# --- Data Classes ---
@dataclass
class DeviceInfo:
    """Complete device information and state"""
    path: str
    serial: str
    model: str
    type: str  # HDD, SSD, NVMe
    major: int
    minor: int
    max_read_mbps: float = 100.0
    max_write_mbps: float = 100.0
    base_speed_mbps: float = 100.0  # min(read, write)
    current_temp: Optional[float] = None
    is_throttled: bool = False
    throttle_percent: float = 0.0
    last_benchmark: Optional[datetime] = None
    temp_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def needs_benchmark(self, expiry_days: int) -> bool:
        """Check if device needs benchmarking"""
        if not self.last_benchmark:
            return True
        age = datetime.now() - self.last_benchmark
        return age.days > expiry_days

@dataclass  
class ThrottleState:
    """Point-in-time throttle state for logging"""
    device: str
    temp: float
    throttle_pct: float
    speed_mbps: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self):
        emoji = "🔥" if self.temp > 50 else "❄️" if self.temp < 35 else "✅"
        return (f"[{self.timestamp.strftime('%H:%M:%S')}] [{self.device}] "
                f"Temp: {self.temp:.1f}°C {emoji} | "
                f"Throttle: {self.throttle_pct:.0f}% | "
                f"Speed: {self.speed_mbps:.1f} MB/s")

# --- Core Components ---

class TemperatureMonitor:
    """Robust temperature monitoring across device types"""
    
    TEMP_PATTERNS = [
        r"Temperature_Celsius.*?(\d+)",
        r"Airflow_Temperature_Cel.*?(\d+)",
        r"Current Temperature.*?(\d+)",
        r"Temperature:.*?(\d+)",
        r"drive temperature.*?(\d+)",
    ]
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.smartctl = self._find_smartctl()
        
    def _find_smartctl(self) -> str:
        """Locate smartctl binary"""
        paths = ["/usr/sbin/smartctl", "/usr/bin/smartctl", 
                "/sbin/smartctl", "/bin/smartctl"]
        for path in paths:
            if Path(path).exists():
                return path
        
        # Try which
        try:
            result = subprocess.run(["which", "smartctl"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
            
        raise RuntimeError("smartctl not found. Install smartmontools.")
    
    def get_temperature(self, device_path: str) -> Optional[float]:
        """Get device temperature with multiple fallback methods"""
        # Try JSON output first (most reliable)
        temp = self._get_temp_json(device_path)
        if temp is not None:
            return temp
            
        # Fallback to text parsing
        return self._get_temp_text(device_path)
    
    def _get_temp_json(self, device_path: str) -> Optional[float]:
        """Get temperature using JSON output"""
        try:
            cmd = [self.smartctl, "-A", "-j", device_path]
            result = subprocess.run(cmd, capture_output=True, 
                                  text=True, timeout=10)
            if result.returncode not in [0, 4]:  # 4 = SMART unavailable
                return None
                
            data = json.loads(result.stdout)
            
            # NVMe
            if "nvme_smart_health_information_log" in data:
                return float(data["nvme_smart_health_information_log"].get("temperature", 0))
            
            # SCSI
            if "temperature" in data:
                return float(data["temperature"].get("current", 0))
                
            # ATA
            if "ata_smart_attributes" in data:
                for attr in data["ata_smart_attributes"].get("table", []):
                    if "temperature" in attr["name"].lower():
                        return float(attr["raw"]["value"])
                        
        except Exception as e:
            self.logger.debug(f"JSON temp read failed for {device_path}: {e}")
            
        return None
    
    def _get_temp_text(self, device_path: str) -> Optional[float]:
        """Fallback text parsing for temperature"""
        try:
            cmd = [self.smartctl, "-A", device_path]
            result = subprocess.run(cmd, capture_output=True, 
                                  text=True, timeout=10)
            if result.returncode not in [0, 4]:
                return None
                
            import re
            for pattern in self.TEMP_PATTERNS:
                match = re.search(pattern, result.stdout, re.IGNORECASE)
                if match:
                    return float(match.group(1))
                    
        except Exception as e:
            self.logger.debug(f"Text temp read failed for {device_path}: {e}")
            
        return None
    
    def get_device_info(self, device_path: str) -> DeviceInfo:
        """Get comprehensive device information"""
        try:
            # Get device numbers
            stat = os.stat(device_path)
            major = os.major(stat.st_rdev)
            minor = os.minor(stat.st_rdev)
            
            # Get SMART info
            cmd = [self.smartctl, "-i", "-j", device_path]
            result = subprocess.run(cmd, capture_output=True, 
                                  text=True, timeout=10)
            
            if result.returncode in [0, 4]:
                data = json.loads(result.stdout)
                serial = data.get("serial_number", "")
                model = data.get("model_name", "Unknown")
                
                # Determine type
                if "nvme" in device_path.lower():
                    dev_type = "NVMe"
                elif data.get("rotation_rate", 1) == 0:
                    dev_type = "SSD"
                else:
                    dev_type = "HDD"
            else:
                # Fallback
                serial = ""
                model = "Unknown"
                dev_type = "HDD"
            
            # Generate serial if missing
            if not serial:
                import hashlib
                serial = f"gen_{hashlib.md5(device_path.encode()).hexdigest()[:8]}"
                
            return DeviceInfo(
                path=device_path,
                serial=serial,
                model=model,
                type=dev_type,
                major=major,
                minor=minor
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get device info: {e}")
            raise

class SafeBenchmark:
    """Safe device benchmarking with caching"""
    
    def __init__(self, cache_dir: Path, logger: logging.Logger):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
    def benchmark(self, device: DeviceInfo, size_mb: int = 512) -> bool:
        """Run safe benchmark and update device speeds"""
        self.logger.info(f"🚀 Benchmarking {device.path} ({device.model})...")
        
        # Find mount point
        mount = self._find_mount_point(device.path)
        if not mount:
            self.logger.warning(f"No mount point for {device.path}, using defaults")
            return False
            
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(dir=mount, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                
            size_bytes = size_mb * 1024 * 1024
            
            # Write benchmark
            self.logger.debug(f"Writing {size_mb}MB test file...")
            start = time.perf_counter()
            subprocess.run([
                "dd", "if=/dev/zero", f"of={tmp_path}",
                f"bs=1M", f"count={size_mb}", 
                "oflag=direct", "conv=fdatasync", "status=none"
            ], check=True, timeout=120)
            write_time = time.perf_counter() - start
            write_mbps = (size_bytes / write_time) / (1024 * 1024)
            
            # Read benchmark  
            self.logger.debug(f"Reading {size_mb}MB test file...")
            subprocess.run(["sync"], check=True)  # Clear caches
            subprocess.run(["echo", "3", ">", "/proc/sys/vm/drop_caches"], 
                         shell=True, check=False)  # Best effort
            
            start = time.perf_counter()
            subprocess.run([
                "dd", f"if={tmp_path}", "of=/dev/null",
                f"bs=1M", "iflag=direct", "status=none"
            ], check=True, timeout=120)
            read_time = time.perf_counter() - start
            read_mbps = (size_bytes / read_time) / (1024 * 1024)
            
            # Update device
            device.max_read_mbps = read_mbps
            device.max_write_mbps = write_mbps
            device.base_speed_mbps = min(read_mbps, write_mbps)
            device.last_benchmark = datetime.now()
            
            # Save to cache
            self._save_cache(device)
            
            self.logger.info(
                f"✅ Benchmark complete: R={read_mbps:.1f} W={write_mbps:.1f} MB/s"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return False
        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
    
    def load_cache(self, device: DeviceInfo) -> bool:
        """Load cached benchmark if valid"""
        cache_file = self.cache_dir / f"{device.serial}.json"
        
        if not cache_file.exists():
            return False
            
        try:
            with open(cache_file) as f:
                data = json.load(f)
                
            device.max_read_mbps = data["read_mbps"]
            device.max_write_mbps = data["write_mbps"]
            device.base_speed_mbps = data["base_mbps"]
            device.last_benchmark = datetime.fromisoformat(data["timestamp"])
            
            age = datetime.now() - device.last_benchmark
            self.logger.info(
                f"Loaded {age.days}-day old cache for {device.path}: "
                f"{device.base_speed_mbps:.1f} MB/s"
            )
            return True
            
        except Exception as e:
            self.logger.debug(f"Cache load failed: {e}")
            return False
    
    def _save_cache(self, device: DeviceInfo):
        """Save benchmark to cache"""
        cache_file = self.cache_dir / f"{device.serial}.json"
        
        data = {
            "model": device.model,
            "type": device.type,
            "read_mbps": device.max_read_mbps,
            "write_mbps": device.max_write_mbps,
            "base_mbps": device.base_speed_mbps,
            "timestamp": device.last_benchmark.isoformat()
        }
        
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _find_mount_point(self, device_path: str) -> Optional[Path]:
        """Find mount point for device"""
        try:
            result = subprocess.run(
                ["lsblk", "-o", "MOUNTPOINT", "-nr", device_path],
                capture_output=True, text=True, timeout=5
            )
            
            for line in result.stdout.strip().split("\n"):
                if line and line.strip():
                    return Path(line.strip())
                    
        except Exception:
            pass
            
        return None

class CgroupThrottler:
    """Cgroups v2 based I/O throttling"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.io_max_file = Path("/sys/fs/cgroup/io.max")
        self._check_cgroups()
        
    def _check_cgroups(self):
        """Verify cgroups v2 is available"""
        if not self.io_max_file.parent.exists():
            raise RuntimeError("cgroups v2 not found at /sys/fs/cgroup")
            
        # Check if io controller is enabled
        subtree = self.io_max_file.parent / "cgroup.subtree_control"
        if subtree.exists():
            controllers = subtree.read_text()
            if "io" not in controllers:
                self.logger.warning("io controller not enabled in cgroups")
    
    def set_throttle(self, device: DeviceInfo, speed_mbps: float):
        """Apply throttle to device"""
        try:
            speed_bps = int(speed_mbps * 1024 * 1024)
            rule = f"{device.major}:{device.minor} rbps={speed_bps} wbps={speed_bps}\n"
            
            # Read existing rules
            existing = {}
            if self.io_max_file.exists():
                for line in self.io_max_file.read_text().strip().split("\n"):
                    if line:
                        parts = line.split(maxsplit=1)
                        if parts:
                            existing[parts[0]] = line
            
            # Update rule
            existing[f"{device.major}:{device.minor}"] = rule.strip()
            
            # Write all rules
            with open(self.io_max_file, "w") as f:
                for rule in existing.values():
                    f.write(rule + "\n")
                    
            device.is_throttled = True
            self.logger.debug(f"Throttled {device.path} to {speed_mbps:.1f} MB/s")
            
        except Exception as e:
            self.logger.error(f"Failed to set throttle: {e}")
    
    def remove_throttle(self, device: DeviceInfo):
        """Remove throttle from device"""
        try:
            rule = f"{device.major}:{device.minor} rbps=max wbps=max\n"
            
            # Read existing rules
            existing = {}
            if self.io_max_file.exists():
                for line in self.io_max_file.read_text().strip().split("\n"):
                    if line:
                        parts = line.split(maxsplit=1)
                        if parts:
                            existing[parts[0]] = line
            
            # Update rule
            existing[f"{device.major}:{device.minor}"] = rule.strip()
            
            # Write all rules
            with open(self.io_max_file, "w") as f:
                for rule in existing.values():
                    f.write(rule + "\n")
                    
            device.is_throttled = False
            self.logger.debug(f"Removed throttle from {device.path}")
            
        except Exception as e:
            self.logger.error(f"Failed to remove throttle: {e}")

class ThermalManager:
    """Main thermal management orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.monitor = TemperatureMonitor(self.logger)
        self.benchmark = SafeBenchmark(Path(config["cache_dir"]), self.logger)
        self.throttler = CgroupThrottler(self.logger)
        self.devices: Dict[str, DeviceInfo] = {}
        self.running = False
        self.monitor_thread = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger("thermal_throttler")
        logger.setLevel(logging.DEBUG)
        
        # Console handler - INFO and above
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            "%(message)s"  # Simple format for console
        ))
        
        # File handler - Everything
        log_file = Path(self.config["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        
        logger.addHandler(console)
        logger.addHandler(file_handler)
        
        return logger
    
    def add_device(self, device_path: str) -> bool:
        """Add device to monitoring"""
        try:
            # Get device info
            device = self.monitor.get_device_info(device_path)
            
            # Load cache or benchmark
            if not self.benchmark.load_cache(device):
                if not self.benchmark.benchmark(device, self.config["benchmark_size_mb"]):
                    self.logger.error(f"Failed to benchmark {device_path}")
                    return False
            
            # Check if needs fresh benchmark
            if device.needs_benchmark(self.config["benchmark_expiry_days"]):
                self.logger.info(f"Cache expired for {device_path}, re-benchmarking...")
                self.benchmark.benchmark(device, self.config["benchmark_size_mb"])
            
            self.devices[device_path] = device
            self.logger.info(f"Added {device_path}: {device.model} ({device.type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add device {device_path}: {e}")
            return False
    
    def calculate_throttle(self, device: DeviceInfo, temp: float) -> float:
        """Calculate throttle amount using sigmoid curve"""
        target = self.config["target_temp"]
        critical = self.config["critical_temp"]
        hysteresis = self.config["hysteresis"]
        min_speed_pct = self.config["min_speed_percent"]
        steepness = self.config["sigmoid_steepness"]
        
        # Check hysteresis
        if device.is_throttled and temp <= (target - hysteresis):
            return 0.0  # Remove throttle
        elif not device.is_throttled and temp <= target:
            return 0.0  # Don't start throttling
        
        # Calculate sigmoid-based throttle
        if temp <= target:
            return 0.0
        elif temp >= critical:
            return 100.0 - min_speed_pct
        else:
            # Map temperature to [0, 1] range
            temp_range = critical - target
            temp_normalized = (temp - target) / temp_range
            
            # Apply sigmoid function
            sigmoid = 1 / (1 + math.exp(-steepness * (temp_normalized - 0.5)))
            
            # Map to throttle percentage
            max_throttle = 100 - min_speed_pct
            return sigmoid * max_throttle
    
    def update_device(self, device: DeviceInfo):
        """Update single device state"""
        # Get temperature
        temp = self.monitor.get_temperature(device.path)
        if temp is None:
            return
            
        device.current_temp = temp
        device.temp_history.append(temp)
        
        # Calculate throttle
        throttle_pct = self.calculate_throttle(device, temp)
        device.throttle_percent = throttle_pct
        
        # Apply or remove throttle
        if throttle_pct > 0:
            speed_pct = 100 - throttle_pct
            speed_mbps = device.base_speed_mbps * (speed_pct / 100)
            speed_mbps = max(speed_mbps, device.base_speed_mbps * 
                           (self.config["min_speed_percent"] / 100))
            
            self.throttler.set_throttle(device, speed_mbps)
            
            state = ThrottleState(
                device=device.path,
                temp=temp,
                throttle_pct=throttle_pct,
                speed_mbps=speed_mbps
            )
            self.logger.info(str(state))
            
        elif device.is_throttled:
            # Remove throttle
            self.throttler.remove_throttle(device)
            
            state = ThrottleState(
                device=device.path,
                temp=temp,
                throttle_pct=0,
                speed_mbps=device.base_speed_mbps
            )
            self.logger.info(f"{state} | Throttle removed")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("🌡️  Storage Thermal Monitor Started")
        self.logger.info("-" * 50)
        
        for device in self.devices.values():
            self.logger.info(
                f"  Monitoring: {device.path} ({device.model}) "
                f"@ {device.base_speed_mbps:.0f} MB/s"
            )
        self.logger.info("-" * 50)
        
        while self.running:
            try:
                for device in self.devices.values():
                    self.update_device(device)
                
                time.sleep(self.config["monitor_interval"])
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(10)
    
    def start(self, daemon: bool = False):
        """Start monitoring"""
        if not self.devices:
            self.logger.error("No devices to monitor")
            return False
            
        self.running = True
        
        if daemon:
            self.monitor_thread = threading.Thread(
                target=self.monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            return True
        else:
            self.monitor_loop()
            return True
    
    def stop(self):
        """Stop monitoring and cleanup"""
        self.logger.info("Stopping thermal monitor...")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Remove all throttles
        for device in self.devices.values():
            if device.is_throttled:
                self.logger.info(f"Removing throttle from {device.path}")
                self.throttler.remove_throttle(device)
        
        self.logger.info("👋 Thermal monitor stopped")

# --- CLI Interface ---

def check_root():
    """Verify root permissions"""
    if os.geteuid() != 0:
        print("❌ This tool requires root privileges. Run with 'sudo'.")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Ultimate Thermal-Aware Storage I/O Throttler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor devices with defaults
  sudo %(prog)s monitor /dev/sda /dev/sdb
  
  # Custom temperature thresholds  
  sudo %(prog)s monitor /dev/sda --target-temp 50 --critical-temp 65
  
  # Run benchmark only
  sudo %(prog)s benchmark /dev/sda /dev/sdb
  
  # Run as daemon with custom interval
  sudo %(prog)s monitor /dev/sda --daemon --interval 60
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Benchmark command
    bench = subparsers.add_parser(
        "benchmark",
        help="Benchmark device speeds"
    )
    bench.add_argument(
        "devices",
        nargs="+",
        help="Device paths (e.g., /dev/sda)"
    )
    bench.add_argument(
        "--size",
        type=int,
        default=512,
        help="Benchmark file size in MB (default: 512)"
    )
    
    # Monitor command
    mon = subparsers.add_parser(
        "monitor",
        help="Monitor and throttle devices"
    )
    mon.add_argument(
        "devices",
        nargs="+",
        help="Device paths to monitor"
    )
    mon.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as background daemon"
    )
    mon.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)"
    )
    mon.add_argument(
        "--target-temp", "-t",
        type=int,
        default=45,
        help="Target temperature in °C (default: 45)"
    )
    mon.add_argument(
        "--critical-temp", "-c",
        type=int,
        default=55,
        help="Critical temperature in °C (default: 55)"
    )
    mon.add_argument(
        "--min-speed",
        type=int,
        default=10,
        help="Minimum speed %% at critical temp (default: 10)"
    )
    mon.add_argument(
        "--hysteresis", "-H",
        type=int,
        default=3,
        help="Temperature hysteresis in °C (default: 3)"
    )
    mon.add_argument(
        "--steepness", "-k",
        type=float,
        default=8.0,
        help="Sigmoid curve steepness (default: 8.0)"
    )
    mon.add_argument(
        "--expire",
        type=int,
        default=90,
        help="Benchmark expiry in days (default: 90)"
    )
    
    args = parser.parse_args()
    
    # Check permissions
    check_root()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        try:
            with open(args.config) as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return 1
    
    # Apply command line overrides
    if hasattr(args, "interval"):
        config["monitor_interval"] = args.interval
    if hasattr(args, "target_temp"):
        config["target_temp"] = args.target_temp
    if hasattr(args, "critical_temp"):
        config["critical_temp"] = args.critical_temp
    if hasattr(args, "min_speed"):
        config["min_speed_percent"] = args.min_speed
    if hasattr(args, "hysteresis"):
        config["hysteresis"] = args.hysteresis
    if hasattr(args, "steepness"):
        config["sigmoid_steepness"] = args.steepness
    if hasattr(args, "expire"):
        config["benchmark_expiry_days"] = args.expire
    if hasattr(args, "size"):
        config["benchmark_size_mb"] = args.size
    
    # Create manager
    manager = ThermalManager(config)
    
    # Set verbose mode
    if args.verbose:
        manager.logger.setLevel(logging.DEBUG)
        for handler in manager.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    # Execute command
    if args.command == "benchmark":
        # Just benchmark
        for device_path in args.devices:
            if not Path(device_path).exists():
                print(f"❌ Device not found: {device_path}")
                continue
                
            device = manager.monitor.get_device_info(device_path)
            manager.benchmark.benchmark(device, config["benchmark_size_mb"])
            
        return 0
        
    elif args.command == "monitor":
        # Add devices
        valid_devices = 0
        for device_path in args.devices:
            if not Path(device_path).exists():
                print(f"❌ Device not found: {device_path}")
                continue
            
            if manager.add_device(device_path):
                valid_devices += 1
            else:
                print(f"⚠️ Failed to add {device_path}")
        
        if valid_devices == 0:
            print("❌ No valid devices to monitor")
            return 1
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            manager.logger.info(f"Received signal {signum}")
            manager.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start monitoring
        try:
            if args.daemon:
                print(f"Starting daemon monitoring {valid_devices} device(s)...")
                manager.start(daemon=True)
                
                # Keep running
                while manager.running:
                    time.sleep(1)
            else:
                # Run in foreground
                manager.start(daemon=False)
                
        except KeyboardInterrupt:
            manager.logger.info("Interrupted by user")
        finally:
            manager.stop()
        
        return 0

if __name__ == "__main__":
    sys.exit(main())