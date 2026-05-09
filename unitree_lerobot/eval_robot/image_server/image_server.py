# Copyright 2025 YuShu TECHNOLOGY CO.,LTD ("Unitree Robotics")
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)
import os
import argparse
import glob
import cv2
import numpy as np
# uvc will be imported when needed
import yaml
import time
import threading
import signal
import functools
import subprocess
import platform
from .image_client import TripleRingBuffer, ZMQ_PublisherManager, ZMQ_Responser
# webrtc dependencies
import asyncio
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.contrib.media import MediaRelay
from aiortc.codecs import h264
import av
import ssl
from pathlib import Path
import queue
import fractions
from typing import Dict, Optional, Tuple, Any

# ========================================================
# cam_config_server.yaml path
# ========================================================
from pathlib import Path
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "cam_config_server.yaml"
)
CONFIG_PATH = os.path.normpath(CONFIG_PATH)

# ========================================================
# certificate and key paths
# ========================================================
module_dir = Path(__file__).resolve().parent.parent.parent
default_cert = module_dir / "cert.pem"
default_key = module_dir / "key.pem"
env_cert = os.getenv("XR_TELEOP_CERT")
env_key = os.getenv("XR_TELEOP_KEY")
user_config_dir = Path.home() / ".config" / "xr_teleoperate"
user_cert = user_config_dir / "cert.pem"
user_key = user_config_dir / "key.pem"
CERT_PEM_PATH = Path(env_cert or (user_cert if user_cert.exists() else default_cert))
KEY_PEM_PATH = Path(env_key or (user_key if user_key.exists() else default_key))
CERT_PEM_PATH = CERT_PEM_PATH.resolve()
KEY_PEM_PATH = KEY_PEM_PATH.resolve()

# ========================================================
# libx264 for Jetson (Patch h264 Encoder)
# ========================================================
def jetson_software_encode_frame(self, frame: av.VideoFrame, force_keyframe: bool):
    if self.codec and (frame.width != self.codec.width or frame.height != self.codec.height):
        self.codec = None

    if self.codec is None:
        try:
            self.codec = av.CodecContext.create("libx264", "w")
            self.codec.width = frame.width
            self.codec.height = frame.height
            self.codec.bit_rate = self.target_bitrate
            self.codec.pix_fmt = "yuv420p"
            self.codec.framerate = fractions.Fraction(30, 1)
            self.codec.time_base = fractions.Fraction(1, 30)
        
            self.codec.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "threads": "1",
                "g": "60",
            }
            self.frame_count = 0
            force_keyframe = True
        except Exception as e:
            logger_mp.error(f"[H264 Patch] Initialization failed: {e}")
            return

    if not force_keyframe and hasattr(self, "frame_count") and self.frame_count % 60 == 0:
        force_keyframe = True
    
    self.frame_count = self.frame_count + 1 if hasattr(self, "frame_count") else 1
    frame.pict_type = av.video.frame.PictureType.I if force_keyframe else av.video.frame.PictureType.NONE

    try:
        for packet in self.codec.encode(frame):
            data = bytes(packet)
            if data:
                yield from self._split_bitstream(data)
    except Exception as e:
        logger_mp.warning(f"[H264 Patch] Encode error: {e}")

h264.H264Encoder._encode_frame = jetson_software_encode_frame

# ========================================================
# Embed HTML and JS directly
# ========================================================
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>WebRTC Stream</title>
    <style>
    body { 
        font-family: sans-serif; 
        background: #fff; 
        color: #000; 
        text-align: center; 
    }
    button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
    video { width: 100%; max-width: 1280px; background: #000; margin-top: 10px; }
    
    /* Title link style */
    h1 a {
        text-decoration: none;
        color: #000;
    }
    h1 a:hover {
        color: #555;
    }
    </style>
</head>
<body>
    <h1>
        <a href="https://github.com/unitreerobotics/teleimager" target="_blank">
            XR Teleoperation WebRTC Camera Stream
        </a>
    </h1>

    <div style="margin-bottom: 20px;">
        <a href="https://www.unitree.com/" target="_blank">
            <img src="https://www.unitree.com/images/0079f8938336436e955ea3a98c4e1e59.svg" alt="Unitree LOGO" width="10%">
        </a>
    </div>

    <button id="start" onclick="start()">Start</button>
    <button id="stop" style="display: none" onclick="stop()">Stop</button>
    
    <div id="media">
        <video id="video" autoplay playsinline muted></video>
        <audio id="audio" autoplay></audio>
    </div>
    
    <script src="client.js"></script>
</body>
</html>
"""

CLIENT_JS = """
var pc = null;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        alert(e);
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    // Removed STUN server check logic completely

    pc = new RTCPeerConnection(config);

    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';
    document.getElementById('start').style.display = 'inline-block';
    if (pc) {
        pc.close();
        pc = null;
    }
}
"""

# ========================================================
# WebRTC publish
# ========================================================
class BGRArrayVideoStreamTrack(MediaStreamTrack):
    """MediaStreamTrack exposing BGR ndarrays as av.VideoFrame (latest-frame semantics)."""
    kind = "video"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._start_time = None
        self._pts = 0

    async def recv(self) -> av.VideoFrame:
        # This will suspend execution until a frame is available
        # preventing CPU busy-waiting
        frame = await self._queue.get()
        return frame

    def push_frame(self, bgr_numpy: np.ndarray, loop: Optional[asyncio.AbstractEventLoop] = None):
        if bgr_numpy is None:
            return

        # 1. Convert and calculate PTS immediately
        # MediaRelay requires consistent PTS to function correctly
        try:
            video_frame = av.VideoFrame.from_ndarray(bgr_numpy, format="bgr24")
            
            if self._start_time is None:
                self._start_time = time.time()
                self._pts = 0
            else:
                # 90000 is the standard RTP clock rate for video
                # This ensures smooth playback
                self._pts = int((time.time() - self._start_time) * 90000)
            
            video_frame.pts = self._pts
            video_frame.time_base = fractions.Fraction(1, 90000)
            
        except Exception as e:
            logger_mp.debug(f"Conversion failed: {e}")
            return

        # 2. Push to queue thread-safely
        target_loop = loop or asyncio.get_event_loop()
        if target_loop.is_closed():
            return
            
        def _put():
            try:
                # Drop old frame if queue is full (Low Latency strategy)
                if self._queue.full():
                    self._queue.get_nowait()
                self._queue.put_nowait(video_frame)
            except Exception:
                pass

        target_loop.call_soon_threadsafe(_put)


class WebRTC_PublisherThread(threading.Thread):
    """
    Runs aiohttp + aiortc in a separate THREAD (not Process).
    This enables shared memory and removes Pickling overhead.
    """
    def __init__(self, port: int, host: str = "0.0.0.0", codec_pref: str = None):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._codec_pref = codec_pref
        self._app = web.Application()
        self._runner: Optional[web.AppRunner] = None
        self._pcs = set()
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=1)

        self._bgr_track: Optional[BGRArrayVideoStreamTrack] = None
        self._relay: Optional[MediaRelay] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # register routes
        self._app.router.add_get("/", self._index)
        self._app.router.add_get("/client.js", self._javascript)
        self._app.router.add_post("/offer", self._offer)

        self._app.router.add_options("/", self._options)
        self._app.router.add_options("/client.js", self._options)
        self._app.router.add_options("/offer", self._options)

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(content_type="text/html", text=INDEX_HTML)
    
    async def _javascript(self, request: web.Request) -> web.Response:
        return web.Response(content_type="application/javascript", text=CLIENT_JS)

    async def _options(self, request):
        return web.Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

    async def _offer(self, request: web.Request) -> web.Response:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self._pcs.add(pc)

        # CORE LOGIC: Use MediaRelay to subscribe
        # This ensures encoding happens only once globally
        if self._bgr_track and self._relay:
            try:
                relayed_track = self._relay.subscribe(self._bgr_track)
                transceiver = pc.addTransceiver(relayed_track, direction="sendonly")
                capabilities = RTCRtpSender.getCapabilities("video")
                pref = (self._codec_pref or "h264").lower()

                if pref == "h264":
                    h264_codecs = [c for c in capabilities.codecs if c.mimeType == "video/H264"]
                    if h264_codecs:
                        transceiver.setCodecPreferences(h264_codecs)
                        logger_mp.info(f"[WebRTC] Preferred H264 for port:{self._port}")
                    else:
                        logger_mp.warning(f"[WebRTC] H264 preferred but not found, using auto-negotiation for port:{self._port}")
                        
                elif pref == "vp8":
                    vp8_codecs = [c for c in capabilities.codecs if c.mimeType == "video/VP8"]
                    if vp8_codecs:
                        transceiver.setCodecPreferences(vp8_codecs)
                        logger_mp.info(f"[WebRTC] Preferred VP8 for port:{self._port}")
                    else:
                        logger_mp.warning(f"[WebRTC] VP8 preferred but not found, using auto-negotiation for port:{self._port}")
                
                else:
                    h264_codecs = [c for c in capabilities.codecs if c.mimeType == "video/H264"]
                    if h264_codecs:
                        transceiver.setCodecPreferences(h264_codecs)
                        logger_mp.info(f"[WebRTC] Preferred codec '{pref}' not found, falling back to H264 for port:{self._port}")
                    else:
                        logger_mp.warning(f"[WebRTC] Preferred codec '{pref}' not found, using auto-negotiation for port:{self._port}")
                    
            except Exception as e:
                logger_mp.error(f"Relay subscription failed: {e}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                await self._cleanup_pc(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

    async def _cleanup_pc(self, pc):
        self._pcs.discard(pc)
        try:
            await pc.close()
        except: pass

    def wait_for_start(self, timeout=1.0):
        return self._start_event.wait(timeout=timeout)

    def run(self):
        # Create a new Event Loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        async def _main():
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            
            # Init Track and Relay inside the loop
            self._bgr_track = BGRArrayVideoStreamTrack()
            self._relay = MediaRelay()

            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(CERT_PEM_PATH, KEY_PEM_PATH)
            site = web.TCPSite(self._runner, self._host, self._port, ssl_context=ssl_context)
            await site.start()
            self._start_event.set()
            
            # Frame Pushing Loop
            while not self._stop_event.is_set():
                try:
                    # Non-blocking check for new frames
                    if not self._frame_queue.empty():
                        # Get frame (no pickling overhead in Threads!)
                        frame = self._frame_queue.get_nowait()
                        self._bgr_track.push_frame(frame, loop=self._loop)
                    
                    # CRITICAL: Yield control to asyncio loop to handle WebRTC packets
                    await asyncio.sleep(0.005)
                except Exception:
                    await asyncio.sleep(0.005)

        try:
            self._loop.run_until_complete(_main())
        except Exception as e:
            logger_mp.error(f"WebRTC Thread Error: {e}")
        finally:
            if self._loop: self._loop.close()

    def send(self, data: np.ndarray):
        """Send data to the processing thread."""
        # Simple drop-frame logic if queue is full
        if not self._frame_queue.full():
            self._frame_queue.put(data)
        else:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put(data)
            except: pass

    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)


# ========================================================
# WebRTC Manager
# ========================================================
class WebRTC_PublisherManager:
    """Manages WebRTC_PublisherThreads."""
    _instance: Optional["WebRTC_PublisherManager"] = None
    _publisher_threads: Dict[Tuple[str, int], WebRTC_PublisherThread] = {}
    _lock = threading.Lock()
    _running = True

    def __init__(self):
        pass

    @classmethod
    def get_instance(cls) -> "WebRTC_PublisherManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _create_publisher(self, port: int, host: str, codec_pref: str):
        t = WebRTC_PublisherThread(port, host, codec_pref)
        t.start()
        if not t.wait_for_start(timeout=10.0):  # Increase timeout to 10 seconds
             raise ConnectionError("Publisher failed to start (Timeout)")
        return t

    def _get_publisher(self, port, host, codec_pref):
        key = (host, port)
        with self._lock:
            if key not in self._publisher_threads:
                self._publisher_threads[key] = self._create_publisher(port, host, codec_pref)
            return self._publisher_threads[key]

    def publish(self, data: Any, port: int, host: str = "0.0.0.0", codec_pref: str = None) -> None:
        if not self._running: return
        try:
            pub = self._get_publisher(port, host, codec_pref)
            pub.send(data)
        except Exception as e:
            logger_mp.error(f"Unexpected error in publish: {e}")
            pass

    def close(self) -> None:
        self._running = False
        with self._lock:
            for key, pub in list(self._publisher_threads.items()):
                try:
                    pub.stop()
                except Exception: pass
            self._publisher_threads.clear()

# ========================================================
# UVC driver reload
# ========================================================
def reload_uvc_driver():
    try:
        subprocess.run("sudo modprobe -r uvcvideo", shell=True, check=True)
        time.sleep(1)
        subprocess.run("sudo modprobe uvcvideo debug=0", shell=True, check=True)
        time.sleep(1)
        logger_mp.info("UVC driver reloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger_mp.error(f"Failed to reload driver: {e}")

# ========================================================
# camera finder and cameras
# ========================================================
class CameraFinder:
    """
    Discover connected cameras and their properties.
    vpath: /dev/videoX
    ppath: physical path in /sys/class/video4linux, e.g. /sys/devices/pci0000:00/0000:00:14.0/usb1/1-11/1-11.2/1-11.2:1.0
    uid: USB unique ID, e.g. "001:002"
    dev_info: extra info from uvc
    sn: serial number of the camera
    """
    def __init__(self, realsense_enable=False, verbose=False):
        self.verbose = verbose
        # uvc
        reload_uvc_driver()
        import uvc
        self.uvc_devices = uvc.device_list()
        self.uid_map = {dev["uid"]: dev for dev in self.uvc_devices}
        # all video devices
        self.video_paths = self._list_video_paths()
        # realsense
        if realsense_enable:
            self.rs_serial_numbers = self._list_realsense_serial_numbers()
            self.rs_video_paths = self._list_realsense_video_paths()
            self.rs_rgb_video_paths = [p for p in self.rs_video_paths if self._is_like_rgb(p)]
        else:
            self.rs_serial_numbers = []
            self.rs_video_paths = []
            self.rs_rgb_video_paths = []
        # rgb & uvc
        self.uvc_rgb_video_paths = self._list_uvc_rgb_video_paths()
        self.uvc_rgb_video_ids = [int(v.replace("/dev/video", "")) for v in self.uvc_rgb_video_paths]
        self.uvc_rgb_physical_paths = [self._get_ppath_from_vpath(v) for v in self.uvc_rgb_video_paths]
        self.uvc_rgb_uids = [self._get_uid_from_ppath(p) for p in self.uvc_rgb_physical_paths]
        self.uvc_rgb_dev_info = [self.uid_map.get(uid) for uid in self.uvc_rgb_uids]
        self.uvc_rgb_serial_numbers = [dev_info.get("serialNumber") if dev_info else None for dev_info in self.uvc_rgb_dev_info]
        # all uvc cameras
        self.uvc_rgb_cameras = {}
        for vpath, vid, ppath, uid, dev_info, sn in zip(
            self.uvc_rgb_video_paths,
            self.uvc_rgb_video_ids,
            self.uvc_rgb_physical_paths,
            self.uvc_rgb_uids,
            self.uvc_rgb_dev_info,
            self.uvc_rgb_serial_numbers,
        ):
            self.uvc_rgb_cameras[vpath] = {
                "video_id": vid,
                "physical_path": ppath,
                "uid": uid,
                "dev_info": dev_info,
                "serial_number": sn
            }
        if self.verbose:
            self.info()

    # utils
    def _list_video_paths(self):
        base = "/sys/class/video4linux/"
        if not os.path.exists(base):
            return []
        return [f"/dev/{x}" for x in sorted(os.listdir(base)) if x.startswith("video")]

    def _list_uvc_rgb_video_paths(self):
        return [p for p in self.video_paths if self._is_like_rgb(p) and p not in self.rs_video_paths]

    def _list_realsense_video_paths(self):
        def _read_text(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
            except Exception:
                return None

        def _parent_usb_device_sysdir(video_sysdir):
            d = os.path.realpath(os.path.join(video_sysdir, "device"))
            for _ in range(10):
                if d is None or d == "/" or not os.path.isdir(d):
                    break
                id_vendor = _read_text(os.path.join(d, "idVendor"))
                id_product = _read_text(os.path.join(d, "idProduct"))
                if id_vendor and id_product:
                    return d
                d_next = os.path.dirname(d)
                if d_next == d:
                    break
                d = d_next
            return None

        ports = []
        for devnode in sorted(glob.glob("/dev/video*")):
            sysdir = f"/sys/class/video4linux/{os.path.basename(devnode)}"
            name = _read_text(os.path.join(sysdir, "name"))
            usb_dir = _parent_usb_device_sysdir(sysdir)
            vendor_id = _read_text(os.path.join(usb_dir, "idVendor")) if usb_dir else None

            # Match RealSense by name and Intel vendor ID
            if name and "realsense" in name.lower() and (vendor_id or "").lower() in ("8086", "32902"):
                ports.append(devnode)

        return ports
    
    def get_realsense_module(self) -> object:
        try:
            import pyrealsense2 as rs
            return rs
        except ImportError:
            arch = platform.machine()
            system = platform.system()
            print(f"[RealSense] Platform: {system} / {arch}")

            if system == "Linux" and arch.startswith("aarch64"):
                # Jetson NX / arm64
                msg = (
                    "[RealSense] pyrealsense2 not installed. please build from source:\n"
                    "    cd ~\n"
                    "    git clone https://github.com/IntelRealSense/librealsense.git\n"
                    "    cd librealsense\n"
                    "    git checkout v2.50.0\n"
                    "    mkdir build && cd build\n"
                    "    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3)\n"
                    "    make -j$(nproc)\n"
                    "    sudo make install\n"
                )
            else:
                # x86/x64
                msg = (
                    "[RealSense] pyrealsense2 not installed. You can try:\n"
                    "    pip install pyrealsense2\n"
                )
            raise RuntimeError(msg)

    def _list_realsense_serial_numbers(self):
        rs = self.get_realsense_module()
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = []
        for dev in devices:
            try:
                serials.append(dev.get_info(rs.camera_info.serial_number))
            except Exception:
                continue
        return serials

    def _get_ppath_from_vpath(self, video_path):
        sysfs_path = f"/sys/class/video4linux/{os.path.basename(video_path)}/device"
        return os.path.realpath(sysfs_path)

    def _get_uid_from_ppath(self, physical_path):
        def read_file(path):
            return open(path).read().strip() if os.path.exists(path) else None

        busnum_file = os.path.join(physical_path, "busnum")
        devnum_file = os.path.join(physical_path, "devnum")

        if not (os.path.exists(busnum_file) and os.path.exists(devnum_file)):
            parent = os.path.dirname(physical_path)
            busnum_file = os.path.join(parent, "busnum")
            devnum_file = os.path.join(parent, "devnum")

        if os.path.exists(busnum_file) and os.path.exists(devnum_file):
            bus = read_file(busnum_file)
            dev = read_file(devnum_file)
            return f"{bus}:{dev}"
        return None

    def _is_like_rgb(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None and frame.ndim == 3 and frame.shape[2] == 3

    # --------------------------------------------------------
    # public api
    # --------------------------------------------------------
    def is_rs_serial_exist(self, serial_number):
        return str(serial_number) in self.rs_serial_numbers

    def is_vpath_exist(self, vpath):
        return vpath in self.video_paths
    
    def is_ppath_exist(self, physical_path):
        for cam in self.uvc_rgb_cameras.values():
            if cam.get("physical_path") == physical_path:
                return True
        return False
    
    def get_uid_by_sn(self, serial_number):
        matches = [
            cam for cam in self.uvc_rgb_cameras.values()
            if cam.get("serial_number") == str(serial_number)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"Multiple cameras found with serial number {serial_number}")
        return matches[0].get("uid")

    def get_uid_by_ppath(self, physical_path):
        for cam in self.uvc_rgb_cameras.values():
            if cam.get("physical_path") == physical_path:
                return cam.get("uid")
        return None
    
    def get_uid_by_vpath(self, video_path):
        cam = self.uvc_rgb_cameras.get(video_path)
        if cam:
            return cam.get("uid")
        return None
    
    def get_vpath_by_sn(self, serial_number):
        matches = []
        for cam in self.uvc_rgb_cameras.values():
            if cam.get("serial_number") == str(serial_number):
                vpath = f"/dev/video{cam.get('video_id')}"
                matches.append(vpath)
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"Multiple video devices found for serial number {serial_number}: {matches}. ")
        return matches[0]

    def get_vpath_by_ppath(self, physical_path):
        base = "/sys/class/video4linux/"
        matches = []
        for v in os.listdir(base):
            sys_path = os.path.realpath(os.path.join(base, v, "device"))
            if sys_path == physical_path:
                vpath = f"/dev/{v}"
                if self._is_like_rgb(vpath):
                    matches.append(vpath)
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"Multiple video devices found for physical path {physical_path}: {matches}. ")
        return matches[0]
    

    def info(self):
        logger_mp.info("======================= Camera Discovery Start ==================================")
        logger_mp.info("Found video devices: %s", self.video_paths)
        logger_mp.info("Found RGB video devices: %s", self.uvc_rgb_video_paths)

        if self.rs_serial_numbers:
            logger_mp.info("----------------------- Realsense Cameras ----------------------------------")
            logger_mp.info(f"RealSense serial numbers: {self.rs_serial_numbers}")
            logger_mp.info(f"RealSense video paths: {self.rs_video_paths}")
            logger_mp.info(f"RealSense RGB-like video paths: {self.rs_rgb_video_paths}")

        for idx, (vpath, cam) in enumerate(self.uvc_rgb_cameras.items(), start=1):
            logger_mp.info("----------------------- OpenCV / UVC Camera %d -----------------------------", idx)
            logger_mp.info("video_path    : %s", vpath)
            logger_mp.info("video_id      : %s", cam.get("video_id"))
            logger_mp.info("serial_number : %s", cam.get("serial_number") or "unknown")
            logger_mp.info("physical_path : %s", cam.get("physical_path"))
            logger_mp.info("extra_info:")

            dev_info = cam.get("dev_info")
            uid = cam.get("uid")

            if dev_info:
                for k, v in dev_info.items():
                    logger_mp.info("    %s: %s", k, v)
                try:
                    import uvc
                    cap = uvc.Capture(uid)
                    for fmt in cap.available_modes:
                        logger_mp.info("    format: %dx%d@%d %s", fmt.height, fmt.width, fmt.fps, fmt.format_name)
                    cap.close()
                    cap = None
                except Exception as e:
                    logger_mp.warning("    failed to get formats: %s", e)
            else:
                logger_mp.info("    no uvc extra info available")

        logger_mp.info("=========================== Camera Discovery End ================================")

class BaseCamera:
    def __init__(self, cam_topic, img_shape, fps, 
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666, webrtc_codec=None):
        self._ready = threading.Event()
        self._cam_topic = cam_topic
        self._img_shape = img_shape # (H, W)
        self._fps = fps
        self._enable_zmq = enable_zmq
        self._zmq_port = zmq_port
        if self._enable_zmq:
            self._zmq_buffer = TripleRingBuffer()
        else:
            self._zmq_buffer = None

        self._enable_webrtc = enable_webrtc
        self._webrtc_port = webrtc_port
        self._webrtc_codec = webrtc_codec
        if self._enable_webrtc:
            self._webrtc_buffer = TripleRingBuffer()
        else:
            self._webrtc_buffer = None

    def __str__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return self.__str__()

    def _update_frame(self):
        """Return a jepg frame as bytes, and a bgr frame as numpy array"""
        raise NotImplementedError
    
    def wait_until_ready(self, timeout=None):
        """Block until the camera is ready (first frame is available) or timeout occurs."""
        return self._ready.wait(timeout=timeout)

    def enable_webrtc(self):
        return self._enable_webrtc
    
    def enable_zmq(self):
        return self._enable_zmq

    def get_jpeg_bytes(self):
        jpeg_bytes = self._zmq_buffer.read() if self._enable_zmq and self._zmq_buffer else None
        return jpeg_bytes

    def get_bgr_frame(self):
        bgr_numpy = self._webrtc_buffer.read() if self._enable_webrtc and self._webrtc_buffer else None
        return bgr_numpy

    def get_depth_frame(self):
        """Return a depth frame as bytes, or None if not supported. 
           Before call this function, must first call get_frame() to update the latest depth data."""
        return None

    def get_zmq_port(self):
        """Return the zmq port number the camera is serving on."""
        return self._zmq_port
    
    def get_webrtc_port(self):
        """Return the webrtc port number the camera is serving on."""
        return self._webrtc_port
    
    def get_webrtc_codec(self):
        """Return the webrtc codec setting."""
        return self._webrtc_codec

    def get_fps(self):
        """Return the camera FPS setting."""
        return self._fps

    def release(self):
        """Release camera resources."""
        raise NotImplementedError

class RealSenseCamera(BaseCamera):
    def __init__(self, cam_topic, serial_number, img_shape, fps, 
                 enable_zmq=True, zmq_port = 55555, enable_webrtc=False, webrtc_port=66666, webrtc_codec=None, enable_depth=False):
        rs = self.check_pyrealsense2_install()
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
        self._serial_number = serial_number
        self._enable_depth = enable_depth
        self._latest_depth = None
        try:
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self._serial_number)

            config.enable_stream(rs.stream.color, self._img_shape[1], self._img_shape[0], rs.format.bgr8, self._fps)
            if self._enable_depth:
                config.enable_stream(rs.stream.depth, self._img_shape[1], self._img_shape[0], rs.format.z16, self._fps)

            profile = self.pipeline.start(config)
            self._device = profile.get_device()
            if self._device is None:
                logger_mp.error('[RealSenseCamera] pipe_profile.get_device() is None .')
            if self._enable_depth:
                assert self._device is not None
                depth_sensor = self._device.first_depth_sensor()
                self.g_depth_scale = depth_sensor.get_depth_scale()

            self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            logger_mp.info(str(self))
        except Exception as e:
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except:
                    pass
            raise RuntimeError(f"[RealSenseCamera] Failed to initialize RealSense camera {self._serial_number}: {e}")

    def __str__(self):
        return (
            f"[RealSenseCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS.\n"
            f"ZMQ: {'enabled, zmq_port=' + str(self._zmq_port) if self._enable_zmq else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc_port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}"
        )

    def check_pyrealsense2_install(self):
        try:
            import pyrealsense2 as rs
            return rs
        except Exception as e:
            raise ImportError(
                "pyrealsense2 not installed. Install Intel RealSense SDK and pyrealsense2 Python bindings."
            ) from e
    
    def _update_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None

        if self._enable_depth:   
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame:
                self._latest_depth = np.asanyarray(depth_frame.get_data())
            else:
                self._latest_depth = None

        bgr_numpy = np.asanyarray(color_frame.get_data())

        if self._enable_webrtc:
            self._webrtc_buffer.write(bgr_numpy)

        if self._enable_zmq:
            ok, buf = cv2.imencode(".jpg", bgr_numpy)
            if ok:
                self._zmq_buffer.write(buf.tobytes())
        
        if not self._ready.is_set():
            self._ready.set()
    
    def get_depth_frame(self):
        if self._latest_depth is None:
            return None
        return self._latest_depth.tobytes()

    def release(self):
        try:
            if hasattr(self.pipeline, "stop") and getattr(self.pipeline, "_running", False):
                try:
                    self.pipeline.stop()
                except Exception as e:
                    logger_mp.warning(f"[RealSenseCamera] pipeline.stop() failed: {e}")
        except Exception:
            pass
        self.pipeline = None
        logger_mp.info(f"[RealSenseCamera] Released {self._cam_topic}")

class UVCCamera(BaseCamera):
    def __init__(self, cam_topic, uid, img_shape, fps, 
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666, webrtc_codec=None):
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
        import uvc
        self.uid = uid
        self.cap = None
        try:
            self.cap = uvc.Capture(self.uid)
        except Exception as e:
            self.cap = None
            raise RuntimeError(f"[UVCCamera] Failed to open camera {self._cam_topic}: {e}")

        try:
            self.cap.frame_mode = self._choose_mode(self.cap, width=self._img_shape[1], height=self._img_shape[0], fps=self._fps)
            logger_mp.info(str(self))
        except Exception as e:
            self.cap = None
            raise RuntimeError(f"[UVCCamera] Failed to set mode for {self._cam_topic}: {e}")

    def __str__(self):
        return (
            f"[UVCCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS, MJPG.\n"
            f"ZMQ: {'enabled, zmq port=' + str(self._zmq_port) if self._enable_zmq else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}"
        )

    def _choose_mode(self, cap, width=None, height=None, fps=None):
        for m in cap.available_modes:
            if m.width == width and m.height == height and m.fps == fps and m.format_name == "MJPG":
                return m
        raise ValueError("[UVCCamera] No matching uvc mode found")

    def _update_frame(self):
        if self.cap is not None:
            frame = self.cap.get_frame_robust() # get_frame(timeout=500)
            if frame is not None:
                if self._enable_zmq:
                    if frame.jpeg_buffer is not None:
                        self._zmq_buffer.write(bytes(frame.jpeg_buffer))

                if self._enable_webrtc:
                    if frame.bgr is not None:
                        self._webrtc_buffer.write(frame.bgr)

                if not self._ready.is_set():
                    self._ready.set()
            else:
                raise RuntimeError

    def release(self):
        # if usbhub is plugged out, calling stop_streaming and close may hang forever.
        # try:
        #     self.cap.stop_streaming()
        # except Exception:
        #     pass
        # try:
        #     self.cap.close()
        # except Exception:
        #     pass
        # self.cap = None
        logger_mp.info(f"[UVCCamera] Released {self._cam_topic}")

class OpenCVCamera(BaseCamera):
    def __init__(self, cam_topic, video_path, img_shape, fps, 
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666, webrtc_codec=None):
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
        self._video_path = video_path

        self.cap = cv2.VideoCapture(self._video_path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Test if the camera can read frames
        if not self._can_read_frame():
            self.release()
            raise RuntimeError(f"[OpenCVCamera] Camera {self._cam_topic} failed to initialize or read frames.")
        else:
            logger_mp.info(str(self))

    def __str__(self):
        return (
            f"[OpenCVCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS.\n"
            f"ZMQ: {'enabled, zmq port=' + str(self._zmq_port) if self._enable_zmq else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}"
        )
        
    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success
    
    def _update_frame(self):
        if self.cap is not None:
            ret, bgr_numpy = self.cap.read()
            if ret:
                if self._enable_webrtc:
                    self._webrtc_buffer.write(bgr_numpy)

                if self._enable_zmq:
                    ok, buf = cv2.imencode(".jpg", bgr_numpy)
                    if ok:
                        self._zmq_buffer.write(buf.tobytes())
                
                if not self._ready.is_set():
                    self._ready.set()
            else:
                raise RuntimeError

    def release(self):
        self.cap.release()
        self.cap = None
        logger_mp.info(f"[OpenCVCamera] Released {self._cam_topic}")

class IsaacSimCamera(BaseCamera):
    def __init__(self, cam_topic, img_shape, fps,
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666, webrtc_codec=None,
                 image_source="head", binocular=False):
        """
        IsaacSim camera that reads from shared memory.

        Args:
            cam_topic: camera topic name
            img_shape: image shape [height, width]
            fps: frames per second
            enable_zmq: enable ZMQ publishing
            zmq_port: ZMQ port
            enable_webrtc: enable WebRTC publishing
            webrtc_port: WebRTC port
            webrtc_codec: WebRTC codec preference
            image_source: which image to read from shared memory ("head", "left", "right")
            binocular: if True and image_source=="head", concatenate left+right for binocular vision
        """
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
        from tools.shared_memory_utils import MultiImageReader # https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/main/tools
        self.multi_image_reader = MultiImageReader()
        self._image_source = image_source  # "head", "left", or "right"
        self._binocular = binocular
        # For IsaacSim cameras, set ready immediately since the camera object is initialized
        # and will wait for shared memory data in _update_frame
        self._ready.set()
        logger_mp.info(str(self))

    def __str__(self):
        mode = "binocular" if self._binocular else "monocular"
        return (
            f"[IsaacSimCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS, source='{self._image_source}', mode='{mode}'.\n"
            f"ZMQ: {'enabled, zmq port=' + str(self._zmq_port) if self._enable_zmq else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}"
        )

    def _update_frame(self):
        # Get the image data based on source and binocular settings
        frame_data = None
        if self._binocular:
            # For binocular cameras: concatenate left + right images
            left_img = self.multi_image_reader.read_single_image('left')
            right_img = self.multi_image_reader.read_single_image('right')
            logger_mp.debug(f"[IsaacSimCamera] {self._cam_topic} - left: {left_img is not None}, right: {right_img is not None}")

            if left_img is not None and right_img is not None:
                frame_data = cv2.hconcat([left_img, right_img])
                logger_mp.debug(f"[IsaacSimCamera] {self._cam_topic} - concatenated binocular frame: {frame_data.shape}")
        else:
            # For monocular cameras: use the specified source directly
            frame_data = self.multi_image_reader.read_single_image(self._image_source)
            if frame_data is None:
                logger_mp.debug(f"[IsaacSimCamera] {self._cam_topic} - no data for source '{self._image_source}'")

        # Publish the frame data only if we have valid data
        if frame_data is not None:
            # For ZMQ: encode to JPEG bytes
            if self._enable_zmq:
                ok, buf = cv2.imencode(".jpg", frame_data)
                if ok:
                    self._zmq_buffer.write(buf.tobytes())
                else:
                    logger_mp.warning(f"[IsaacSimCamera] Failed to encode to JPEG for {self._cam_topic}")

            # For WebRTC: use BGR frames directly
            if self._enable_webrtc:
                self._webrtc_buffer.write(frame_data)
            else:
                logger_mp.warning(f"[IsaacSimCamera] Failed to encode to WebRTC for {self._cam_topic}")
            if not self._ready.is_set():
                self._ready.set()
        else:
            logger_mp.debug(f"[IsaacSimCamera] No data available for {self._cam_topic}, frame_data is None")
        # If no data is available, just return silently and wait for next frame

    def release(self):
        if hasattr(self, 'multi_image_reader') and self.multi_image_reader is not None:
            self.multi_image_reader.close()
        self.multi_image_reader = None
        logger_mp.info(f"[IsaacSimCamera] Released {self._cam_topic}")
# ========================================================
# image server
# ========================================================
class ImageServer:
    def __init__(self, cam_config, realsense_enable=False, camera_finder_verbose=False, isaacsim_enable=False):
        self._cam_config = cam_config
        self._realsense_enable = realsense_enable
        self._isaacsim_enable = isaacsim_enable
        self._stop_event = threading.Event()
        self._cameras: dict[str, BaseCamera] = {}
        if not self._isaacsim_enable:
            self._cam_finder = CameraFinder(realsense_enable, camera_finder_verbose)
        self._responser = ZMQ_Responser(self._cam_config)
        self._zmq_publisher_manager = ZMQ_PublisherManager.get_instance()
        self._webrtc_publisher_manager = WebRTC_PublisherManager.get_instance()
        self._publisher_threads = []  # keep references for graceful join

        try:
            # Load cameras from self.cam_config
            for cam_topic, cam_cfg in self._cam_config.items():
                if not cam_cfg.get("enable_zmq", False) and not cam_cfg.get("enable_webrtc", False):
                    continue

                enable_zmq = cam_cfg.get("enable_zmq", False)
                zmq_port = cam_cfg.get("zmq_port", None)
                enable_webrtc = cam_cfg.get("enable_webrtc", False)
                webrtc_port = cam_cfg.get("webrtc_port", None)
                webrtc_codec = cam_cfg.get("webrtc_codec", None)
                cam_type = cam_cfg.get("type", "uvc").lower()
                if self._isaacsim_enable and cam_type!="isaacsim":
                    cam_type = "isaacsim"
                img_shape = cam_cfg.get("image_shape", None)
                fps = cam_cfg.get("fps", 30)
                video_id = cam_cfg.get("video_id", "0")
                video_path = f"/dev/video{video_id}" if video_id else None
                physical_path = str(cam_cfg.get("physical_path")) if cam_cfg.get("physical_path") else None
                serial_number = str(cam_cfg.get("serial_number")) if cam_cfg.get("serial_number") else None

                if cam_type == "opencv":
                    if physical_path is not None:
                        vpath = self._cam_finder.get_vpath_by_ppath(physical_path)
                        if vpath is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find OpenCVCamera for {cam_topic} with physical path {physical_path}")
                        else:
                            self._cameras[cam_topic] = OpenCVCamera(cam_topic, vpath, img_shape, fps, 
                                                                    enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
                            continue

                    if serial_number is not None:
                        vpath = self._cam_finder.get_vpath_by_sn(serial_number)
                        if vpath is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find OpenCVCamera for {cam_topic} with serial number {serial_number}")
                        else:
                            self._cameras[cam_topic] = OpenCVCamera(cam_topic, vpath, img_shape, fps, 
                                                                    enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
                        # once you specify either `physical_path` or `serial_number`, the system will no longer fall back to searching by `video_id`.
                        # ——— even if no camera matches the given path/serial.
                        continue
                    
                    if not self._cam_finder.is_vpath_exist(video_path):
                        self._cameras[cam_topic] = None
                        logger_mp.error(f"[Image Server] Cannot find OpenCVCamera for {cam_topic} with video_id {video_id}")
                    else:
                        self._cameras[cam_topic] = OpenCVCamera(cam_topic, video_path, img_shape, fps,
                                                                enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
                        

                elif cam_type == "realsense":
                    if not self._realsense_enable:
                        self._cameras[cam_topic] = None
                        logger_mp.error(f"[Image Server] Please start image server with the '--rs' flag to support Realsense {cam_topic}.")
                    elif not self._cam_finder.is_rs_serial_exist(serial_number):
                        self._cameras[cam_topic] = None
                        logger_mp.error(f"[Image Server] Cannot find RealSenseCamera for {cam_topic}")
                    else:
                        self._cameras[cam_topic] = RealSenseCamera(cam_topic, serial_number, img_shape, fps,
                                                                   enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)

                elif cam_type == "uvc":
                    uid = None
                    if physical_path is not None:
                        uid = self._cam_finder.get_uid_by_ppath(physical_path)
                        if uid is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find UVCCamera for {cam_topic} with physical path {physical_path}")
                        else:
                            self._cameras[cam_topic] = UVCCamera(cam_topic, uid, img_shape, fps, 
                                                                 enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
                            continue

                    if serial_number is not None:
                        uid = self._cam_finder.get_uid_by_sn(serial_number)
                        if uid is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find UVCCamera for {cam_topic} with serial number {serial_number}")
                        else:
                            self._cameras[cam_topic] = UVCCamera(cam_topic, uid, img_shape, fps, 
                                                                 enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec)
                        # once you specify either `physical_path` or `serial_number`, the system will no longer fall back to searching by `video_id`.
                        # ——— even if no camera matches the given path/serial.
                        continue
                elif cam_type == "isaacsim":
                    # Check if binocular mode is enabled
                    binocular = cam_cfg.get("binocular", False)

                    # For IsaacSim cameras, determine image source based on camera topic and binocular setting
                    if binocular:
                        # Binocular cameras (like head) need to read left+right and concatenate
                        image_source = "head"  # Special marker for binocular
                    else:
                        # Monocular cameras read their specific source
                        if "left" in cam_topic.lower():
                            image_source = "left"
                        elif "right" in cam_topic.lower():
                            image_source = "right"
                        else:
                            image_source = "head"  # fallback

                    self._cameras[cam_topic] = IsaacSimCamera(cam_topic, img_shape, fps,
                                                                enable_zmq, zmq_port, enable_webrtc, webrtc_port, webrtc_codec,
                                                                image_source=image_source, binocular=binocular)
                else:
                    logger_mp.error(f"[Image Server] Unknown camera type {cam_type} for {cam_topic}, skipping...")
                    continue
        except Exception as e:
            logger_mp.error(f"[Image Server] Initialization failed: {e}")
            self._clean_up()
            raise

        logger_mp.info("[Image Server] Image server has started, waiting for client connections...")

    def _update_frames(self, cam_topic: str, camera: BaseCamera):
        try:
            interval = 1.0 / camera.get_fps()
            next_frame_time = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    camera._update_frame()
                except Exception as e:
                    logger_mp.error(f"[Image Server] Error updating frame for {cam_topic} camera")
                    self._stop_event.set()
                    break
                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] Failed to update frames for {cam_topic} camera: {e}")
            self._stop_event.set()

    def _zmq_pub(self, cam_topic: str, camera: BaseCamera):
        try:
            interval = 1.0 / camera.get_fps()
            next_frame_time = time.monotonic()

            while not self._stop_event.is_set():
                jpeg_bytes = camera.get_jpeg_bytes()
                if jpeg_bytes is not None:
                    self._zmq_publisher_manager.publish(jpeg_bytes, camera.get_zmq_port())
                else:
                    logger_mp.warning(f"[Image Server] {cam_topic} returned no frame.")
                    self._stop_event.set()
                    break

                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] Failed to publish zmq frame from {cam_topic} camera.")
            self._stop_event.set()
    
    def _webrtc_pub(self, cam_topic: str, camera: BaseCamera):
        try:
            interval = 1.0 / camera.get_fps()
            webrtc_codec = camera.get_webrtc_codec()
            next_frame_time = time.monotonic()
            while not self._stop_event.is_set():
                bgr_frame = camera.get_bgr_frame()

                if bgr_frame is not None:
                    self._webrtc_publisher_manager.publish(bgr_frame, camera.get_webrtc_port(), codec_pref=webrtc_codec)
                else:
                    logger_mp.info(f"[Image Server] {cam_topic} returned no frame.")
                    self._stop_event.set()
                    break

                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] Failed to publish rtc frame from {cam_topic} camera.")
            self._stop_event.set()

    def _clean_up(self):
        self._responser.stop()
        for t in self._publisher_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._publisher_threads.clear()
        
        try:
            self._zmq_publisher_manager.close()
        except Exception:
            pass
        try:
            self._webrtc_publisher_manager.close()
        except Exception:
            pass

        for cam in self._cameras.values():
            if cam:
                try:
                    cam.release()
                except Exception as e:
                    logger_mp.error(f"[Image Server] Error releasing camera {cam._cam_topic}: {e}")
        logger_mp.info("[Image Server] Clean up completed. Server stopped.")

    # --------------------------------------------------------
    # public api
    # --------------------------------------------------------
    def start(self):
        for camera_topic, camera in self._cameras.items():
            if camera is None:
                logger_mp.error(f"[Image Server] Camera {camera_topic} failed to initialize previously, cannot start.")
                self._stop_event.set()
                self._clean_up()
                return
            t = threading.Thread(target=self._update_frames, args=(camera_topic, camera), daemon=True)
            t.start()
            self._publisher_threads.append(t)
        if self._isaacsim_enable:
            time.sleep(2.0)  # wait a bit for IsaacSim shared memory to be ready

        for camera_topic, camera in self._cameras.items():
            # Use longer timeout for IsaacSim cameras since they need to wait for shared memory data
            if self._isaacsim_enable:
                timeout = 15.0
            else:
                timeout = 5.0
            ready = camera.wait_until_ready(timeout=timeout)
            if not ready:
                logger_mp.error(f"[Image Server] {camera_topic} ready timeout after {timeout}s.")
                self._stop_event.set()
                self._clean_up()
            logger_mp.info(f"[Image Server] {camera_topic} is ready.")
        
        for camera_topic, camera in self._cameras.items():
            if camera.enable_webrtc():
                t = threading.Thread(target=self._webrtc_pub, args=(camera_topic, camera), daemon=True)
                t.start()
                self._publisher_threads.append(t)

            if camera.enable_zmq():
                t = threading.Thread(target=self._zmq_pub, args=(camera_topic, camera), daemon=True)
                t.start()
                self._publisher_threads.append(t)

    def wait(self):
        self._stop_event.wait()
        self._clean_up()

    def stop(self):
        self._stop_event.set()

# ========================================================
# utility functions
# ========================================================
def signal_handler(server, signum, frame):
    logger_mp.info(f"[Image Server] Received signal {signum}, initiating graceful shutdown...")
    server.stop()

def set_performance_mode(cores=[0, 1, 2]):
    import psutil
    try:
        p = psutil.Process(os.getpid())
        
        # Set CPU affinity for the process and all its threads
        p.cpu_affinity(cores)
        logger_mp.info(f"[Performance] CPU Affinity locked to: {cores}")

    except psutil.AccessDenied:
        logger_mp.warning("[Performance] Access Denied: Run as sudo for full optimization")
    except Exception as e:
        logger_mp.error(f"[Performance] Error: {e}")

def run_isaacsim_server():
    # Load config file, start image server
    try:
        with open(CONFIG_PATH, "r") as f:
            cam_config = yaml.safe_load(f)
    except Exception as e:
        logger_mp.error(f"Failed to load configuration file at {CONFIG_PATH}: {e}")
        exit(1)
    # start image server
    server = ImageServer(cam_config, realsense_enable=False, camera_finder_verbose=False, isaacsim_enable=True)
    server.start()
    return server

def main():
    logger_mp.info(
        "\n====================== Image Server Startup Guide ======================\n"
        "Please first read this repo's README.md to learn how to configure and use the teleimager.\n"
        "To discover connected cameras, run the following command:\n"
        "\n"
        "    teleimager-server --cf\n"
        "\n"
        "The '--cf' flag means 'camera find'.\n"
        "This will list all detected cameras and their details (video paths, serial numbers and physical path etc.).\n"
        "Use that information to fill in your 'cam_config_server.yaml' file.\n"
        "Once configured, you can start the image server with:\n"
        "\n"
        "    teleimager-server\n"
        "\n"
        "Note:\n"
        " - If you have RealSense cameras, add the '--rs' flag to enable RealSense support.\n"
        " - Make sure you have proper permissions to access the camera devices (e.g., run with sudo or set udev rules).\n"
        "=========================================================================="
    )

    # command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', action = 'store_true', help = 'Enable camera found mode, print all connected cameras info')
    parser.add_argument('--rs', action = 'store_true', help = 'Enable RealSense camera mode. Otherwise only find UVC/OpenCV cameras.')
    parser.add_argument('--no-affinity', action='store_false', dest='affinity', help='Disable CPU affinity setting for performance optimization.')
    args = parser.parse_args()

    if args.affinity:
        set_performance_mode(cores=[0, 1, 2])

    # if enable camera finder mode, just print cameras info and exit
    if args.cf:
        CameraFinder(realsense_enable=args.rs, verbose=True)
        exit(0)

    # Load config file, start image server
    try:
        with open(CONFIG_PATH, "r") as f:
            cam_config = yaml.safe_load(f)
    except Exception as e:
        logger_mp.error(f"Failed to load configuration file at {CONFIG_PATH}: {e}")
        exit(1)

    # start image server
    server = ImageServer(cam_config, realsense_enable=args.rs, camera_finder_verbose=False)
    server.start()

    # graceful shutdown handling
    signal.signal(signal.SIGINT, functools.partial(signal_handler, server))
    signal.signal(signal.SIGTERM, functools.partial(signal_handler, server))

    logger_mp.info("[Image Server] Running... Press Ctrl+C to exit.")
    server.wait()

    # usbhub plugout may cause block process exit, no better solution for now
    time.sleep(0.5)
    os.killpg(os.getpgrp(), 9)

if __name__ == "__main__":
    main()
