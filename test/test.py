# typos:ignore-file
import time
import numpy as np
import threading
import torch
from sshkeyboard import listen_keyboard, stop_listening
import os
import zmq

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


# state transition
START = False  # Enable to start robot following VR user motion
STOP = False  # Enable to begin system exit procedure
READY = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RESET = False  # Enable to reset robot to initial pose


CAMERA_STATUS = False  # camera status
ARM_STATUS = False  # arm status
EE_STATUS = False  # end effector status

"""
# Client → Server (Request)
1) launch

2) exit
    {
        "reqid": unique id,
        "cmd": "CMD_STOP"
    }
3) start inference
    {
        "reqid": unique id,
        "cmd": "CMD_START"
    }
4) reset
    {
        "reqid": unique id,
        "cmd": "CMD_RESET"
    }
"""


class IPC_Server:
    """
    Inter - Process Communication Server:
    - Handle data via REP
    - Publish heartbeat via PUB, Heartbeat state is provided by external callback get_state()
    """

    # Mapping table for on_press keys
    cmd_map = {
        "CMD_START": "s",  # start inference
        "CMD_RESET": "r",  # reset
        "CMD_STOP": "q",  # exit
    }

    def __init__(self, on_press=None, get_state=None, hb_fps=10.0):
        """
        Args:
            on_press  : callback(cmd:str), called for every command
            get_state : callback() -> dict, provides current heartbeat state
            hb_fps    : heartbeat publish frequency
        """
        if callable(on_press):
            self.on_press = on_press
        else:
            raise ValueError("[IPC_Server] on_press callback function must be provided")

        if callable(get_state):
            self.get_state = get_state
        else:
            raise ValueError("[IPC_Server] get_state callback function must be provided")
        self._hb_interval = 1.0 / float(hb_fps)
        self._running = True
        self._data_loop_thread = None
        self._hb_loop_thread = None

        self.ctx = zmq.Context.instance()
        self.rep_socket = self.ctx.socket(zmq.REP)
        self.rep_socket.bind("ipc://@xr_teleoperate_data.ipc")
        logger_mp.info("[IPC_Server] Listening to Data at ipc://@xr_teleoperate_data.ipc")

        # heartbeat IPC (PUB/SUB)
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind("ipc://@xr_teleoperate_hb.ipc")
        logger_mp.info("[IPC_Server] Publishing HeartBeat at ipc://@xr_teleoperate_hb.ipc")

    def _data_loop(self):
        """
        Listen for REQ/REP commands and optional info.
        """
        poller = zmq.Poller()
        poller.register(self.rep_socket, zmq.POLLIN)
        while self._running:
            try:
                socks = dict(poller.poll(20))
                if self.rep_socket in socks:
                    msg = self.rep_socket.recv_json()
                    reply = self._handle_message(msg)
                    try:
                        self.rep_socket.send_json(reply)
                    except Exception as e:
                        logger_mp.error(f"[IPC_Server] Failed to send reply: {e}")
                    finally:
                        logger_mp.debug(f"[IPC_Server] DATA recv: {msg} -> rep: {reply}")
            except zmq.error.ContextTerminated:
                break
            except Exception as e:
                logger_mp.error(f"[IPC_Server] Data loop exception: {e}")

    def _hb_loop(self):
        """Publish heartbeat periodically"""
        while self._running:
            start_time = time.monotonic()
            try:
                state = dict(self.get_state() or {})
                self.pub_socket.send_json(state)
                logger_mp.debug(f"[IPC_Server] HB pub: {state}")
            except Exception as e:
                logger_mp.error(f"[IPC_Server] HeartBeat loop exception: {e}")
            elapsed = time.monotonic() - start_time
            if elapsed < self._hb_interval:
                time.sleep(self._hb_interval - elapsed)

    def _handle_message(self, msg: dict) -> dict:
        """Process message and return reply"""
        try:
            # validate reqid
            reqid = msg.get("reqid", None)
            if not reqid:
                return {"repid": 0, "status": "error", "msg": "reqid not provided"}  # typos:ignore

            # validate cmd
            cmd = msg.get("cmd", None)
            if not cmd:
                return {"repid": reqid, "status": "error", "msg": "cmd not provided"}

            # unsupported cmd
            if cmd not in self.cmd_map:
                return {"repid": reqid, "status": "error", "msg": f"cmd not supported: {cmd}"}

            # supported cmd path
            self.on_press(self.cmd_map[cmd])

            return {"repid": reqid, "status": "ok", "msg": "ok"}

        except Exception as e:
            return {"repid": 1, "status": "error", "msg": str(e)}

    # ---------------------------
    # Public API
    # ---------------------------
    def start(self):
        """Start both data loop and heartbeat loop"""
        self._data_loop_thread = threading.Thread(target=self._data_loop, daemon=True)
        self._data_loop_thread.start()
        self._hb_loop_thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb_loop_thread.start()

    def stop(self):
        """Stop server"""
        self._running = False
        if self._data_loop_thread:
            self._data_loop_thread.join(timeout=1.0)
        if self._hb_loop_thread:
            self._hb_loop_thread.join(timeout=1.0)
        try:
            self.rep_socket.setsockopt(zmq.LINGER, 0)
            self.rep_socket.close()
        except Exception:
            pass
        try:
            self.pub_socket.setsockopt(zmq.LINGER, 0)
            self.pub_socket.close()
        except Exception:
            pass
        try:
            self.ctx.term()
        except Exception:
            pass


class IPC_Client:
    """
    Inter - Process Communication Client:
    - Send command via REQ
    - Subscribe heartbeat via SUB
    """

    def __init__(self, hb_fps=10.0):
        """hb_fps: heartbeat subscribe frequency, should match server side."""
        rd = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
        self.ctx = zmq.Context.instance()

        # heartbeat IPC (PUB/SUB)
        self._hb_running = True
        self._hb_last_time = 0  # timestamp of last heartbeat received
        self._hb_latest_state = {}  # latest heartbeat state
        self._hb_online = False  # whether heartbeat is online
        self._hb_interval = 1.0 / float(hb_fps)  # expected heartbeat interval
        self._hb_lock = threading.Lock()  # lock for heartbeat state
        self._hb_timeout = 5.0 * self._hb_interval  # timeout to consider offline
        self.hb_ipc = os.path.join(rd, f"xr-teleoperate-hb-{os.getuid()}.ipc")
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVHWM, 1)
        self.sub_socket.connect(f"ipc://{self.hb_ipc}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        logger_mp.info(f"[IPC_Client] Subscribed to HeartBeat at ipc://{self.hb_ipc}")
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb_thread.start()

        # data IPC (REQ/REP)
        self.data_ipc = os.path.join(rd, f"xr-teleoperate-data-{os.getuid()}.ipc")
        self.req_socket = self.ctx.socket(zmq.REQ)
        self.req_socket.connect(f"ipc://{self.data_ipc}")
        logger_mp.info(f"[IPC_Client] Connected to Data at ipc://{self.data_ipc}")

    def _make_reqid(self) -> str:
        import uuid

        return str(uuid.uuid4())

    # ---------------------------
    # Heartbeat handling
    # ---------------------------
    def _hb_loop(self):
        consecutive = 0
        while self._hb_running:
            start_time = time.monotonic()
            try:
                msg = self.sub_socket.recv_json(flags=zmq.NOBLOCK)
                with self._hb_lock:
                    self._hb_latest_state = msg
                    self._hb_last_time = time.monotonic()
                    consecutive += 1
                    if consecutive >= 3:  # require 3 consecutive heartbeats to be considered online
                        self._hb_online = True
            except zmq.Again:
                with self._hb_lock:
                    if self._hb_last_time > 0:
                        if self._hb_online and (time.monotonic() - self._hb_last_time > self._hb_timeout):
                            self._hb_latest_state = {}
                            self._hb_last_time = 0
                            self._hb_online = False
                            consecutive = 0
                            logger_mp.warning("[IPC_Client] HeartBeat timeout -> OFFLINE")
            except Exception as e:
                logger_mp.error(f"[IPC_Client] HB loop exception: {e}")
            elapsed = time.monotonic() - start_time
            if elapsed < self._hb_interval:
                time.sleep(self._hb_interval - elapsed)

    # ---------------------------
    # Public API
    # ---------------------------
    def send_data(self, cmd: str, episode_id: str | None = None) -> dict:
        """Send command to server and wait reply"""
        reqid = self._make_reqid()
        if not self.is_online():
            logger_mp.warning(f"[IPC_Client] Cannot send {cmd}, server offline (no heartbeat)")
            return {"repid": reqid, "status": "error", "msg": "server offline (no heartbeat)"}

        msg = {"reqid": reqid, "cmd": cmd}

        try:
            self.req_socket.send_json(msg)
            # wait up to 1s for reply
            if self.req_socket.poll(1000):
                reply = self.req_socket.recv_json()
            else:
                return {"repid": reqid, "status": "error", "msg": "timeout waiting for server reply"}
        except Exception as e:
            logger_mp.error(f"[IPC_Client] send_data failed: {e}")
            return {"repid": reqid, "status": "error", "msg": str(e)}

        if reply.get("status") != "ok":
            return reply
        if reply.get("repid") != reqid:
            return {
                "repid": reqid,
                "status": "error",
                "msg": f"reply id mismatch: expected {reqid}, got {reply.get('repid')}",
            }
        return reply

    def is_online(self) -> bool:
        with self._hb_lock:
            return self._hb_online

    def latest_state(self) -> dict:
        with self._hb_lock:
            return dict(self._hb_latest_state)

    def stop(self):
        self._hb_running = False
        if self._hb_thread:
            self._hb_thread.join(timeout=1.0)
        try:
            self.req_socket.setsockopt(zmq.LINGER, 0)
            self.req_socket.close()
        except Exception:
            pass
        try:
            self.sub_socket.setsockopt(zmq.LINGER, 0)
            self.sub_socket.close()
        except Exception:
            pass
        try:
            self.ctx.term()
        except Exception:
            pass


# ---------------------------
# Client Example usage
# ---------------------------
if __name__ == "__main__":
    from sshkeyboard import listen_keyboard, stop_listening

    client = None
    es_id = 47

    def on_press(key: str):
        global client, es_id
        if client is None:
            logger_mp.warning("⚠️ Client not initialized, ignoring key press")
            return

        if key == "r":
            logger_mp.info("▶️ Sending launch command...")
            rep = client.send_data("CMD_RESET")
            logger_mp.info("Reply: %s", rep)

        elif key == "s":
            logger_mp.info("⏺️ Sending start inference command...")
            rep = client.send_data("CMD_START")
            logger_mp.info("Reply: %s", rep)

        elif key == "q":
            logger_mp.info("⏹️ Sending exit command...")
            rep = client.send_data("CMD_STOP")
            logger_mp.info("Reply: %s", rep)

        else:
            logger_mp.warning(f"⚠️ Undefined key: {key}")

    # Initialize client
    client = IPC_Client(hb_fps=10.0)

    # Start keyboard listening thread
    listen_keyboard_thread = threading.Thread(
        target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False}, daemon=True
    )
    listen_keyboard_thread.start()

    logger_mp.info(
        "✅ Client started, waiting for keyboard input:\n [r] launch, [s] start/stop record, [b] heartbeat, [q] exit"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger_mp.info("⏹️ User interrupt, preparing to exit...")
    finally:
        stop_listening()
        client.stop()
        logger_mp.info("✅ Client exited")


def on_press(key):
    global START, RESET, STOP
    if key == "s":
        START = not START
        logger_mp.info(f"==> START = {START}")
    elif key == "r":
        RESET = True
        logger_mp.info("==> RESET = True")
    elif key == "q":
        STOP = True
        logger_mp.info("==> STOP = True")


def get_state() -> dict:
    """Return current heartbeat state"""
    global START, RESET, STOP, READY
    return {
        "START": START,
        "RESET": RESET,
        "STOP": STOP,
        "READY": READY,
        "CAMERA_STATUS": CAMERA_STATUS,
        "ARM_STATUS": ARM_STATUS,
        "EE_STATUS": EE_STATUS,
    }


# --- 模拟依赖项 ---
class DummyArmController:
    def ctrl_dual_arm(self, q, tau):
        print(f"[ArmCtrl] Moving to pose (first 3 dims): {q[:3]} ...")

    def ctrl_dual_arm_go_home(self):
        print("[ArmCtrl] Going home...")


class DummyIKSolver:
    def solve_tau(self, q):
        return np.zeros_like(q)


class DummyPolicy:
    def predict_action(self, observation):
        # 模拟推理输出（加一点随机扰动）
        obs = observation["observation.state"].numpy()
        action = obs + np.random.uniform(-0.05, 0.05, size=obs.shape)
        print(f"[Policy] Predicted action (first 3 dims): {action[:3]}")
        return action


# --- 执行动作函数 ---
def execute_action(action_np, arm_dof, ee_dof, arm_ik, arm_ctrl, ee_shared_mem=None):
    tau = arm_ik.solve_tau(action_np[:arm_dof])
    arm_ctrl.ctrl_dual_arm(action_np[:arm_dof], tau)


# --- 模拟观测函数 ---
def process_images_and_observations(*args, **kwargs):
    arm_q = np.random.uniform(-1, 1, 7)  # 模拟机械臂 7 维关节
    obs = {"observation.image": None}
    return obs, arm_q


# --- 主逻辑函数 ---
def eval_main():
    global START, RESET, STOP

    # 模拟配置
    cfg = type("cfg", (), {})()
    cfg.frequency = 5
    cfg.ee = False
    cfg.task = "dummy_task"
    cfg.init_pose = np.zeros(7)
    cfg.policy_url = "http://localhost"
    cfg.force_predict = False

    print("[INIT] Setting up dummy policy and robot...")
    policy = DummyPolicy()
    arm_ctrl = DummyArmController()
    arm_ik = DummyIKSolver()

    init_pose = cfg.init_pose
    arm_dof = len(init_pose)
    ee_dof = 0
    ipc = False
    if ipc:
        ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
        ipc_server.start()

    else:
        listen_keyboard_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={
                "on_press": on_press,
                "until": None,
                "sequential": False,
            },
            daemon=True,
        )
        listen_keyboard_thread.start()

    print("[READY] Use 'r' to toggle START, 't' to RESET, 'q' to STOP\n")

    state = init_pose.copy()
    idx = 0

    while not STOP:
        loop_start = time.perf_counter()
        observation, current_arm_q = process_images_and_observations()
        state = current_arm_q
        observation["observation.state"] = torch.from_numpy(state).float()

        if RESET:
            print("[RESET PHASE] Interpolating back to init pose...")
            interp_poses = np.linspace(state, init_pose, 10)
            for q in interp_poses:
                execute_action(q, arm_dof, ee_dof, arm_ik, arm_ctrl)
                time.sleep(1.0 / cfg.frequency)
            print("[RESET COMPLETE]")
            RESET = False
            START = False
            action_np = init_pose.copy()

        elif START:
            print("[START PHASE] Running policy inference...")
            action_np = policy.predict_action(observation)
        else:
            print("[HOLD PHASE] Holding current position...")
            action_np = state.copy()

        execute_action(action_np, arm_dof, ee_dof, arm_ik, arm_ctrl)
        idx += 1
        time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start)))

    print("\n[EXIT] Stopping program...")
    arm_ctrl.ctrl_dual_arm_go_home()


if __name__ == "__main__":
    eval_main()
