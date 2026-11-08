#!/usr/bin/env python3
"""
pen_viewer.py - live 3D pencil viewer for the two-IMU pen tracker.

Install:   pip install pygame numpy          (add pyserial only for --serial mode)

Run:       python pen_viewer.py              listen for the pen over WiFi (UDP port 4210)
           python pen_viewer.py --ble        connect over Bluetooth instead (pip install bleak)
           python pen_viewer.py --demo       synthetic motion, no hardware needed
           python pen_viewer.py --serial COM5   read the same data over USB instead

First time with real hardware:  press K and follow the two on-screen steps.
The result is saved to pen_calibration.json next to this script and loaded
automatically next time (delete that file to go back to the MOUNT defaults).

Controls
    K           calibrate sensor mounting (2 poses: flat, then standing tip-up)
    G           accelerometer calibration (8 guided poses; needs the new sketch)
    D           start / stop recording raw data to penlog_*.csv (for troubleshooting)
    [ / ]       drift filter shorter / longer (shorter = less drift, letters a bit smaller)
    C           zero the heading (hold the pen roughly level, tip pointing "forward")
    P           toggle position tracking (on by default; the tip moves freely and
                leaves a trail of what you wrote)
    W           "paper" mode: lock the tip to the table plane (kills vertical drift)
    R           clear the drawing and put the tip back at the origin
    E           save the page (true-size Letter .svg, prints 1:1) and a .csv next to this script
    N           page view: what you wrote on a ruled sheet of paper, at true scale
    V           toggle top-down view (to read what you wrote)
    F           toggle camera follow
    L           toggle yaw lock (keeps the two gyros' headings glued together)
    A           show / hide sensor axes
    Left-drag   orbit camera            Mouse wheel   zoom

Writing tips
    Pause briefly (~0.3 s) between strokes / letters. Every pause is used to
    remove the drift that built up during the stroke, so short strokes with
    small pauses track far better than one long continuous scribble.
    Esc         cancel calibration / quit      Q   quit

What you see
    * The pencil is drawn from the fused orientation of both sensors and pivots
      about its tip (the tip is the tracked point).
    * The two small boxes are the sensors, each drawn with its OWN estimate of the
      pen's orientation. After calibration they should sit flush on the pen;
      if they separate, that's flex, a bad calibration, or a sensor problem.
"""
import argparse
import json
import math
import os
import queue
import socket
import sys
import threading
import time
from collections import deque

import numpy as np
import pygame

# =============================================================================
# Settings you may want to change
# =============================================================================
UDP_PORT = 4210
SERIAL_BAUD = 115200

# Positions along the pen axis in metres. 0 = middle of the pen, + = toward the tip.
# Measure these on your pen - the tip-position estimate depends on them.
REAR_SENSOR_X = -0.060
TIP_SENSOR_X = 0.045
TIP_POINT_X = 0.080            # the graphite point

# Default mounting, used until you run the calibration (K). Directions are in the
# SENSOR's own axes (the X/Y/Z arrows printed on the MPU6050 board):
#   tip_axis: which sensor axis points toward the pen tip
#   up_axis : which sensor axis points up out of the pen when it lies flat
MOUNT = {
    "rear": dict(tip_axis=(1, 0, 0), up_axis=(0, 0, 1)),
    "tip": dict(tip_axis=(1, 0, 0), up_axis=(0, 0, 1)),
}
CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pen_calibration.json")

CAL_HOLD_S = 1.5          # how long each calibration pose must be held still
CAL_STILL_DEG = 1.5       # max wobble (deg) that still counts as "held still"
CAL_MOVE_DEG = 45         # pose 2 must differ from pose 1 by at least this much

YAW_LOCK_RATE = 0.5       # 1/s - how fast the two sensors' headings are pulled together
STILL_THRESHOLD = 0.30    # m/s^2 spread over 0.25 s below which the pen may be "still"
STILL_GYRO_DPS = 15.0     # ...and it must also be turning slower than this (deg/s)
VEL_DAMPING = 0.3         # 1/s - base damping
QUIET_ACC = 0.35          # m/s^2 - below this (smoothed) the hand is coasting or stopping...
QUIET_DAMPING = 5.0       # 1/s   ...so leftover speed is bled off quickly (pen stops when you stop)
MAX_SPEED = 0.6           # m/s - handwriting never goes faster; anything above is drift
ACC_HP_TAU = 1.0          # s - acceleration that persists longer than this is treated as
                          #     sensor error (tilt leak) and removed; strokes are much shorter
POSITION_LIMIT = 3.0      # safety clamp on the tip position (metres)
TRAIL_POINTS = 30000      # max points kept in the drawing

# The sheet of paper (US Letter, college ruled). The pencil tip starts at PAGE_START_MM
# (from the sheet's left edge, top edge). "Up the page" = the direction the tip pointed
# when you zeroed the heading (C), so zero it with the pencil pointing away from you.
PAGE_W_MM, PAGE_H_MM = 215.9, 279.4
PAGE_START_MM = (40.0, 45.0)
PAGE_RULE_MM = 7.1
PAGE_TOP_LINE_MM = 32.0
PAGE_MARGIN_MM = 32.0
PENCIL_LEN = 0.178        # tip to eraser (m), for the pencil's shadow in page view

PEN_RADIUS = 0.006
GRID_Z = 0.0              # the "table"; the tip starts on it at the origin
BG = (24, 27, 33)
SENSOR_NAMES = ("rear", "tip")

# =============================================================================
# Math helpers (quaternions are [w, x, y, z], sensor body -> world)
# =============================================================================
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def qmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_q(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return qnorm(np.array(q))


def q_from_euler(yaw, pitch, roll):
    qz_ = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qy_ = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    qx_ = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
    return qmul(qz_, qmul(qy_, qx_))


def qz(a):
    """Quaternion for a rotation of a radians about world Z."""
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(D):
    """Angle of the rotation about world Z that best matches rotation matrix D."""
    return math.atan2(D[1, 0] - D[0, 1], D[0, 0] + D[1, 1])


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def angle_deg(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(u), unit(v)))))))


def world_to_page_mm(p):
    """World point (m) -> position on the sheet (mm from left edge, mm from top edge)."""
    return (PAGE_START_MM[0] - p[1] * 1000.0, PAGE_START_MM[1] - p[0] * 1000.0)


def page_mm_to_world(x_mm, y_mm):
    return np.array([(PAGE_START_MM[1] - y_mm) / 1000.0, (PAGE_START_MM[0] - x_mm) / 1000.0, GRID_Z])


def mount_matrix(tip_axis, up_axis):
    """Columns = pen X (toward tip), Y, Z (up) expressed in the sensor's axes."""
    x = unit(tip_axis)
    z = np.array(up_axis, float)
    z = unit(z - x * np.dot(z, x))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# =============================================================================
# Calibration file
# =============================================================================
def load_mounts():
    mounts = {k: dict(v) for k, v in MOUNT.items()}
    try:
        with open(CAL_FILE) as f:
            data = json.load(f)
        found = False
        for name in SENSOR_NAMES:
            if name in data:
                mounts[name] = dict(tip_axis=tuple(data[name]["tip_axis"]),
                                    up_axis=tuple(data[name]["up_axis"]))
                found = True
        return mounts, found
    except (OSError, ValueError, KeyError, TypeError):
        return {k: dict(v) for k, v in MOUNT.items()}, False


def save_mounts(mounts):
    data = {name: {"tip_axis": [round(float(v), 5) for v in m["tip_axis"]],
                   "up_axis": [round(float(v), 5) for v in m["up_axis"]]}
            for name, m in mounts.items()}
    try:
        with open(CAL_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


# =============================================================================
# Data sources: each pushes parsed sample dicts onto a queue
# =============================================================================
def parse_line(line):
    """$PEN,seq,us,flags, rear(qw,qx,qy,qz,ax,ay,az), tip(qw,qx,qy,qz,ax,ay,az)"""
    line = line.strip()
    if not line.startswith("$PEN,"):
        return None
    p = line.split(",")
    if len(p) != 18:
        return None
    try:
        seq, us, flags = int(p[1]), int(p[2]), int(p[3])
        v = [float(x) for x in p[4:]]
    except ValueError:
        return None
    return {
        "seq": seq, "us": us, "flags": flags,
        "q": [np.array(v[0:4]), np.array(v[7:11])],
        "a": [np.array(v[4:7]), np.array(v[11:14])],
    }


def _push(out_q, sample):
    try:
        out_q.put_nowait(sample)
    except queue.Full:
        pass


def udp_thread(port, out_q, stop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break
        for line in data.decode("ascii", "ignore").splitlines():
            s = parse_line(line)
            if s:
                s["src"] = addr[0]
                _push(out_q, s)
    sock.close()


NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # Bluetooth LE "UART" notify characteristic
BLE_STATUS = {"state": ""}                               # shown on screen while waiting


class LineAssembler:
    """Bluetooth delivers the text in packets that need not match line boundaries."""

    def __init__(self, out_q, src):
        self.buf = b""
        self.out_q, self.src = out_q, src

    def feed(self, data):
        self.buf += bytes(data)
        if len(self.buf) > 4096:                          # garbage guard
            self.buf = self.buf[-512:]
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            s = parse_line(line.decode("ascii", "ignore"))
            if s:
                s["src"] = self.src
                _push(self.out_q, s)


def ble_thread(target, out_q, stop):
    """Finds the pen by name (or address), subscribes to its data, reconnects if it drops."""
    import asyncio
    from bleak import BleakClient, BleakScanner

    def is_address(t):
        return t.count(":") == 5 or (len(t) == 36 and t.count("-") == 4)

    async def run():
        while not stop.is_set():
            try:
                BLE_STATUS["state"] = f"searching for '{target}'..."
                if is_address(target):
                    dev = await BleakScanner.find_device_by_address(target, timeout=6.0)
                else:
                    dev = await BleakScanner.find_device_by_filter(
                        lambda d, ad: (d.name or ad.local_name or "") == target, timeout=6.0)
                if dev is None:
                    BLE_STATUS["state"] = f"'{target}' not found - is the pen on, with LINK_BLE in the sketch?"
                    await asyncio.sleep(1.0)
                    continue
                BLE_STATUS["state"] = "connecting..."
                lines = LineAssembler(out_q, "BLE " + (dev.name or dev.address))
                async with BleakClient(dev) as client:
                    await client.start_notify(NUS_TX_UUID, lambda _c, data: lines.feed(data))
                    BLE_STATUS["state"] = f"connected to {dev.name or dev.address}"
                    while client.is_connected and not stop.is_set():
                        await asyncio.sleep(0.2)
                BLE_STATUS["state"] = "connection lost - reconnecting..."
            except Exception as exc:                          # adapter off, out of range, ...
                BLE_STATUS["state"] = f"Bluetooth error: {exc} - retrying"
                await asyncio.sleep(2.0)

    asyncio.run(run())


def check_ble_available():
    try:
        import bleak  # noqa: F401
    except ImportError:
        sys.exit("--ble needs the bleak package:  pip install bleak")


def serial_thread(port, baud, out_q, stop):
    import serial  # pip install pyserial
    ser = serial.Serial(port, baud, timeout=0.5)
    while not stop.is_set():
        try:
            line = ser.readline().decode("ascii", "ignore")
        except Exception:
            time.sleep(0.2)
            continue
        s = parse_line(line)
        if s:
            s["src"] = port
            _push(out_q, s)
    ser.close()


def demo_thread(out_q, stop):
    """Synthetic data: the pen waves around while moving in a small Lissajous path."""
    t0 = time.time()
    seq = 0
    while not stop.is_set():
        t = time.time() - t0
        yaw = 0.8 * math.sin(t * 0.7)
        pitch = 0.35 + 0.4 * math.sin(t * 1.1)
        roll = 1.0 * math.sin(t * 0.5)
        q = q_from_euler(yaw, pitch, roll)
        wobble = q_from_euler(0.01 * math.sin(t * 2.0), 0.0, 0.0)   # tiny sensor disagreement
        acc = np.array([                      # second derivative of a +/-3 cm wiggle
            -0.03 * 5.0 ** 2 * math.sin(5.0 * t),
            -0.03 * 4.0 ** 2 * math.sin(4.0 * t),
            -0.02 * 6.0 ** 2 * math.sin(6.0 * t),
        ])
        _push(out_q, {
            "seq": seq, "us": int(t * 1e6) & 0xFFFFFFFF, "flags": 3,
            "q": [q, qnorm(qmul(wobble, q))], "a": [acc, acc], "src": "demo",
        })
        seq += 1
        time.sleep(0.01)


# =============================================================================
# Tracking: per-sensor mounting + heading, fused orientation, tip position
# =============================================================================
class Tracker:
    def __init__(self):
        self.valid = [False, False]
        self.q_raw = [None, None]          # sensor -> world, as sent by the pen
        self.q_pen_raw = [None, None]      # pen -> (that sensor's) world
        self.a_raw = [None, None]
        self.up_hist = [deque(maxlen=25), deque(maxlen=25)]   # world-up seen in sensor axes

        self.mounts, self.cal_loaded = load_mounts()
        self._apply_mounts()

        # Each MPU6050 boots with its own arbitrary heading, so each gets its own
        # heading correction psi[i] (rotation about world Z).
        self.psi = [0.0, 0.0]
        self.tared = False
        self.yaw_lock = True

        self.R_sensor = [np.eye(3), np.eye(3)]
        self.R_fused = np.eye(3)
        self.disagree_deg = None

        # Position of the pen TIP (relative to where it started)
        self.track_position = True
        self.paper_mode = False
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc_bias = np.zeros(3)
        self.acc_tip = np.zeros(3)
        self.window = deque(maxlen=25)
        self.is_still = False
        self.ang_rate = 0.0                # deg/s, from successive fused orientations
        self.t_s = 0.0                     # sample-clock seconds
        self.settle_until = 0.0            # don't learn accel bias until tilt has re-levelled
        self.acc_cal = [0, 0]              # from firmware: 0 none, 1 offset, 2 offset+scale
        self.acc_poses = [0, 0]            # rest poses stored by the firmware
        self._R_prev = None
        self.mid = np.zeros(3)             # integrated point between the sensors
        self.x_ref = None
        self.damping = VEL_DAMPING
        self.acc_level = 0.0
        self.acc_slow = np.zeros(3)
        self.hp_tau = ACC_HP_TAU
        self.strokes = []                  # finished, drift-corrected strokes (lists of points)
        self.seg = []                      # (t, pos) samples of the stroke in progress
        self.seg_t = 0.0
        self.moving = False

        # Calibration wizard
        self.cal_step = 0                  # 0 = idle, 1 = lying flat, 2 = standing tip-up
        self.cal_msg = ""
        self.cal_sensors = []
        self.cal_flat_up = {}
        self.cal_flat_q = {}
        self._cal_reset_hold()
        self.notice = None                 # one-shot message for the UI
        self.acc_step = 0                  # guided accel calibration: 0 idle, else pose number
        self.acc_t0 = None
        self.acc_progress = 0.0
        self.acc_err = 180.0
        self.acc_done_t = 0.0
        self.rec = None                    # data recorder file
        self.rec_name = ""
        self.rec_t0 = 0.0

        self.last_us = None
        self.last_seq = None
        self.lost = 0
        self.total = 0
        self.last_rx = 0.0
        self.rx_times = deque(maxlen=400)
        self.src = ""

    def _apply_mounts(self):
        self.M = [mount_matrix(**self.mounts[n]) for n in SENSOR_NAMES]
        self.qM = [mat_to_q(m) for m in self.M]
        for i in (0, 1):
            if self.q_raw[i] is not None:
                self.q_pen_raw[i] = qmul(self.q_raw[i], self.qM[i])

    # ---- input -------------------------------------------------------------
    def feed(self, s):
        now = time.time()
        self.last_rx = now
        self.rx_times.append(now)
        self.total += 1
        self.src = s.get("src", "")

        if self.last_seq is not None:
            gap = s["seq"] - self.last_seq
            if 1 < gap < 1000:
                self.lost += gap - 1
        self.last_seq = s["seq"]

        dt = None
        if self.last_us is not None:
            d = ((s["us"] - self.last_us) & 0xFFFFFFFF) / 1e6
            if 0.0 < d <= 0.1:
                dt = d
        self.last_us = s["us"]

        f = s["flags"]
        self.acc_cal = [2 if f & 16 else (1 if f & 4 else 0), 2 if f & 32 else (1 if f & 8 else 0)]
        self.acc_poses = [(f >> 8) & 15, (f >> 12) & 15]
        if dt is not None:
            self.t_s += dt

        was_valid = list(self.valid)
        for i in (0, 1):
            self.valid[i] = bool((s["flags"] >> i) & 1)
            if self.valid[i]:
                self.q_raw[i] = qnorm(np.asarray(s["q"][i], float))
                self.q_pen_raw[i] = qmul(self.q_raw[i], self.qM[i])
                self.a_raw[i] = np.asarray(s["a"][i], float)
                self.up_hist[i].append(qmat(self.q_raw[i])[2, :].copy())
            else:
                self.up_hist[i].clear()

        # A sensor that just (re)started begins at heading 0 - snap it onto the other one.
        for i in (1, 0):
            j = 1 - i
            if self.valid[i] and not was_valid[i] and self.valid[j]:
                self._match_heading(i, j)

        # The pen is rigid, so both sensors must share one heading. Gyro yaw drifts
        # independently on each chip; pull them together slowly (keeps their average).
        if dt is not None and self.yaw_lock and self.valid[0] and self.valid[1]:
            d = yaw_of(self._R_world(1) @ self._R_world(0).T)
            k = min(1.0, YAW_LOCK_RATE * dt)
            self.psi[0] += 0.5 * k * d
            self.psi[1] -= 0.5 * k * d

        self._update_pose()
        if dt is not None and self._R_prev is not None:
            c = (np.trace(self._R_prev.T @ self.R_fused) - 1.0) / 2.0
            rate = math.degrees(math.acos(max(-1.0, min(1.0, c)))) / dt
            self.ang_rate += 0.3 * (rate - self.ang_rate)
            if self.ang_rate > 30.0:
                # After a tilt the pen's tilt estimate needs a moment to re-level;
                # until then a little gravity shows up as fake acceleration.
                self.settle_until = self.t_s + 0.8
        self._R_prev = self.R_fused.copy()
        self._update_motion(dt)
        if self.acc_step:
            self._acc_tick()
        if self.rec:
            self._record(s)
        if self.cal_step:
            self._cal_tick(now)

    def _R_world(self, i):
        return rotz(self.psi[i]) @ qmat(self.q_pen_raw[i])

    def _match_heading(self, i, j):
        self.psi[i] += yaw_of(self._R_world(j) @ self._R_world(i).T)

    def _update_pose(self):
        qs = []
        for i in (0, 1):
            if self.valid[i] and self.q_pen_raw[i] is not None:
                q = qmul(qz(self.psi[i]), self.q_pen_raw[i])
                qs.append(q)
                self.R_sensor[i] = qmat(q)
        if not qs:
            return
        if len(qs) == 2:
            d = float(np.dot(qs[0], qs[1]))
            q2 = qs[1] if d >= 0 else -qs[1]
            fused = qnorm(qs[0] + q2)
            self.disagree_deg = math.degrees(2 * math.acos(min(1.0, abs(d))))
        else:
            fused = qs[0]
            self.disagree_deg = None
        self.R_fused = qmat(fused)
        for i in (0, 1):
            if not self.valid[i]:
                self.R_sensor[i] = self.R_fused

    def _update_motion(self, dt):
        idx = [i for i in (0, 1) if self.valid[i] and self.a_raw[i] is not None]
        if not idx:
            return
        # The AVERAGE acceleration of the sensors is exactly the acceleration of the
        # point between them (rigid body). We integrate that point ("mid") and place
        # the tip from the orientation. So when you tilt the pen, the tip's swing comes
        # from the gyros (accurate) - not from double-integrated accelerometer noise.
        x_ref = float(np.mean([REAR_SENSOR_X if i == 0 else TIP_SENSOR_X for i in idx]))
        acc = np.mean([rotz(self.psi[i]) @ self.a_raw[i] for i in idx], axis=0)
        off = self.R_fused @ np.array([TIP_POINT_X - x_ref, 0.0, 0.0])
        if self.x_ref is None or abs(x_ref - self.x_ref) > 1e-9 or not self.track_position:
            self.mid = self.pos - off          # (re)anchor so the tip stays where it is
            self.x_ref = x_ref
        self.acc_tip = acc
        self.window.append(acc)

        still = False
        if len(self.window) == self.window.maxlen:
            still = (float(np.ptp(np.array(self.window), axis=0).max()) < STILL_THRESHOLD
                     and self.ang_rate < STILL_GYRO_DPS)
        self.is_still = still
        if still:
            # The pen has stopped: whatever velocity we still think it has is drift.
            # Use it to straighten out the stroke that just ended, then zero it.
            if self.moving:
                self._end_stroke()
            # While stationary the remaining "acceleration" is just bias: learn it.
            if self.t_s >= self.settle_until:
                self.acc_bias += 0.005 * (acc - self.acc_bias)
            self.vel[:] = 0.0
        elif self.track_position and dt is not None and len(self.window) == self.window.maxlen:
            if not self.moving:
                self.moving = True
                self.seg = [(0.0, self.mid.copy(), off.copy())]
                self.seg_t = 0.0
            a = acc - self.acc_bias
            if self.hp_tau:
                # Real hand motion pushes back and forth within a fraction of a second;
                # a push that lasts much longer is the tilt estimate leaking gravity.
                self.acc_slow += min(1.0, dt / self.hp_tau) * (a - self.acc_slow)
                a = a - self.acc_slow
            if self.paper_mode:
                a[2] = 0.0
            # Smoothed acceleration level (~0.1 s). When the hand isn't pushing the pen
            # any more, real speed dies away fast, so leftover speed is mostly drift.
            self.acc_level += min(1.0, dt / 0.1) * (float(np.linalg.norm(a)) - self.acc_level)
            damp = self.damping
            if QUIET_ACC and self.acc_level < QUIET_ACC:
                damp = max(damp, QUIET_DAMPING * (1.0 - self.acc_level / QUIET_ACC))
            v_new = (self.vel + a * dt) * math.exp(-damp * dt)
            sp = float(np.linalg.norm(v_new))
            if MAX_SPEED and sp > MAX_SPEED:
                v_new *= MAX_SPEED / sp
            self.mid = np.clip(self.mid + 0.5 * (self.vel + v_new) * dt, -POSITION_LIMIT, POSITION_LIMIT)
            self.vel = v_new
            self.seg_t += dt

        if self.track_position:
            if self.paper_mode:
                self.mid[2] = GRID_Z - off[2]     # keep the tip on the table
                self.vel[2] = 0.0
            self.pos = self.mid + off
            if self.moving:
                self.seg.append((self.seg_t, self.mid.copy(), off.copy()))

    def seg_tips(self):
        return [m + o for _, m, o in self.seg]

    def _end_stroke(self):
        """Zero-velocity correction: the pen really stopped, so the leftover velocity
        v_end is error that (assuming a constant bias) grew linearly over the stroke.
        Its effect on position grows as t^2 - subtract that from every point."""
        T = self.seg_t
        if T > 0 and len(self.seg) > 1:
            v_end = self.vel.copy()
            pts = [m - v_end * (t * t / (2.0 * T)) + o for t, m, o in self.seg]
            self.mid = self.mid - v_end * (T / 2.0)
            if self.paper_mode:
                self.mid[2] = GRID_Z - self.seg[-1][2][2]
            self.pos = self.mid + self.seg[-1][2]
            arr = np.array(pts)
            if float(np.ptp(arr, axis=0).max()) > 0.001:      # ignore sub-millimetre jitter
                self.strokes.append(pts[::2] + [pts[-1]])
            total = sum(len(s) for s in self.strokes)
            while total > TRAIL_POINTS and len(self.strokes) > 1:
                total -= len(self.strokes.pop(0))
        self.seg = []
        self.seg_t = 0.0
        self.moving = False

    def export_drawing(self):
        strokes = [np.array(s) for s in self.strokes if len(s) > 1]
        if self.seg:
            strokes.append(np.array(self.seg_tips()))
        if not strokes:
            return "Nothing drawn yet"
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            time.strftime("drawing_%Y%m%d_%H%M%S"))
        try:
            # A full Letter page at true size: print at 100% and it lines up with the paper.
            paths = []
            for s in strokes:
                xy = [world_to_page_mm(p) for p in s]
                paths.append('<path d="M ' + " L ".join(f"{x:.2f},{y:.2f}" for x, y in xy) + '"/>')
            with open(base + ".svg", "w") as f:
                f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W_MM}mm" height="{PAGE_H_MM}mm" '
                        f'viewBox="0 0 {PAGE_W_MM} {PAGE_H_MM}">\n'
                        f'<rect width="{PAGE_W_MM}" height="{PAGE_H_MM}" fill="white"/>\n'
                        '<g fill="none" stroke="#333" stroke-width="0.5" stroke-linecap="round" '
                        'stroke-linejoin="round">\n' + "\n".join(paths) + "\n</g></svg>\n")
            with open(base + ".csv", "w") as f:
                f.write("stroke,page_x_mm,page_y_mm,x_mm,y_mm,z_mm\n")
                for k, s in enumerate(strokes):
                    for p in s:
                        px, py = world_to_page_mm(p)
                        f.write(f"{k},{px:.2f},{py:.2f},{p[0] * 1000:.2f},{p[1] * 1000:.2f},{p[2] * 1000:.2f}\n")
        except OSError as exc:
            return f"Export failed: {exc}"
        print("Saved", base + ".svg", "and .csv")
        return "Saved " + os.path.basename(base) + ".svg / .csv"

    # ---- calibration wizard ------------------------------------------------
    def start_calibration(self):
        idx = [i for i in (0, 1) if self.valid[i]]
        if not idx:
            return "No sensor data - can't calibrate yet"
        self.cal_sensors = idx
        self.acc_step = 0
        self.cal_step = 1
        self.cal_flat_up, self.cal_flat_q = {}, {}
        self.cal_msg = ""
        self._cal_reset_hold()
        return "Calibration started"

    def cancel_calibration(self):
        self.cal_step = 0
        return "Calibration cancelled"

    def _cal_reset_hold(self):
        self.cal_t0 = None
        self.cal_ups = {0: [], 1: []}
        self.cal_progress = 0.0

    def _cal_tick(self, now):
        idx = self.cal_sensors
        if any(not self.valid[i] for i in idx):
            self.cal_msg = "A sensor dropped out - check wiring"
            self._cal_reset_hold()
            return

        still = True
        for i in idx:
            h = self.up_hist[i]
            if len(h) < h.maxlen:
                still = False
                break
            arr = np.array(h)
            m = unit(arr.mean(axis=0))
            if max(angle_deg(u, m) for u in arr) > CAL_STILL_DEG:
                still = False
                break
            if self.cal_step == 2 and angle_deg(m, self.cal_flat_up[i]) < CAL_MOVE_DEG:
                still = False            # not stood up yet
                break
        if not still:
            self._cal_reset_hold()
            return

        if self.cal_t0 is None:
            self.cal_t0 = now
        for i in idx:
            self.cal_ups[i].append(self.up_hist[i][-1].copy())
        self.cal_progress = min(1.0, (now - self.cal_t0) / CAL_HOLD_S)
        if self.cal_progress < 1.0:
            return

        means = {i: unit(np.mean(self.cal_ups[i], axis=0)) for i in idx}
        if self.cal_step == 1:
            self.cal_flat_up = means
            self.cal_flat_q = {i: self.q_raw[i].copy() for i in idx}
            self.cal_step = 2
            self.cal_msg = ""
            self._cal_reset_hold()
            return

        # Pose 2: whatever sensor direction now points at the ceiling is the tip axis.
        for i in idx:
            ang = angle_deg(means[i], self.cal_flat_up[i])
            if abs(ang - 90.0) > 25.0:
                self.cal_msg = (f"{SENSOR_NAMES[i]} sensor: the two poses were {ang:.0f} deg apart "
                                f"(should be ~90). Starting over.")
                self.cal_step = 1
                self._cal_reset_hold()
                return

        for i in idx:
            self.mounts[SENSOR_NAMES[i]] = dict(tip_axis=tuple(float(v) for v in means[i]),
                                                up_axis=tuple(float(v) for v in self.cal_flat_up[i]))
        self._apply_mounts()
        # Heading: in pose 1 the tip pointed "forward", so make that heading 0.
        for i in idx:
            x = qmat(qmul(self.cal_flat_q[i], self.qM[i]))[:, 0]
            self.psi[i] = -math.atan2(x[1], x[0])
        self.tared = True
        saved = save_mounts(self.mounts)
        self.cal_loaded = self.cal_loaded or saved
        self.cal_step = 0
        self.reset_motion()
        self._update_pose()
        report = ", ".join(f"{SENSOR_NAMES[i]} tip-axis {np.round(means[i], 2).tolist()}" for i in idx)
        print("Calibration:", report)
        self.notice = "Calibration saved" if saved else "Calibrated (could not write pen_calibration.json)"

    # ---- helpers -----------------------------------------------------------
    def pen_center(self):
        return self.pos - self.R_fused @ np.array([TIP_POINT_X, 0.0, 0.0])

    def sensor_world(self, i):
        x = REAR_SENSOR_X if i == 0 else TIP_SENSOR_X
        return self.pen_center() + self.R_fused @ np.array([x, 0.0, PEN_RADIUS + 0.0016])

    def reset_motion(self):
        self.pos[:] = 0.0
        self.vel[:] = 0.0
        self.acc_bias[:] = 0.0
        self.window.clear()
        self.strokes = []
        self.seg = []
        self.seg_t = 0.0
        self.moving = False
        self.x_ref = None

    # ---- guided accelerometer calibration (G) ------------------------------
    # The firmware does the actual calibration: it stores every new resting
    # orientation. This just walks you through 8 well-spread ones.
    ACC_POSES = [
        ("Lay it FLAT (as in the K calibration)", (0, 0, 1)),
        ("Roll it a QUARTER TURN onto its side", (0, 1, 0)),
        ("Roll it UPSIDE DOWN", (0, 0, -1)),
        ("Roll it a quarter turn onto its OTHER side", (0, -1, 0)),
        ("Hold it with the TIP STRAIGHT UP", (1, 0, 0)),
        ("Hold it with the TIP STRAIGHT DOWN", (-1, 0, 0)),
        ("Flat (right way up), tip RAISED about 45 deg", (0.707, 0, 0.707)),
        ("Upside down, tip LOWERED about 45 deg", (-0.707, 0, -0.707)),
    ]

    def start_acc_calibration(self):
        if not any(self.valid):
            return "No sensor data yet"
        self.cal_step = 0
        self.acc_step = 1
        self.acc_t0 = None
        self.acc_progress = 0.0
        return "Accelerometer calibration started"

    def _acc_tick(self):
        if self.acc_step > len(self.ACC_POSES):
            return
        target = unit(self.ACC_POSES[self.acc_step - 1][1])
        up_pen = self.R_fused.T @ np.array([0.0, 0.0, 1.0])     # world up, in pen axes
        self.acc_err = angle_deg(up_pen, target)
        if self.acc_err < 20.0 and self.ang_rate < 8.0:
            if self.acc_t0 is None:
                self.acc_t0 = self.t_s
            self.acc_progress = min(1.0, (self.t_s - self.acc_t0) / 1.5)
            if self.acc_progress >= 1.0:
                self.acc_step += 1
                self.acc_t0 = None
                self.acc_progress = 0.0
                if self.acc_step > len(self.ACC_POSES):
                    self.acc_done_t = self.t_s
        else:
            self.acc_t0 = None
            self.acc_progress = 0.0

    # ---- data recorder (D) -------------------------------------------------
    def toggle_recording(self):
        if self.rec:
            self.rec.close()
            self.rec = None
            return "Recording saved: " + os.path.basename(self.rec_name)
        self.rec_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     time.strftime("penlog_%Y%m%d_%H%M%S.csv"))
        try:
            self.rec = open(self.rec_name, "w")
        except OSError as exc:
            return f"Can't record: {exc}"
        self.rec.write("t,flags,rq_w,rq_x,rq_y,rq_z,ra_x,ra_y,ra_z,tq_w,tq_x,tq_y,tq_z,ta_x,ta_y,ta_z,"
                       "psi0,psi1,tip_x,tip_y,tip_z,vel_x,vel_y,vel_z,still,ang_rate\n")
        self.rec_t0 = self.t_s
        return "Recording... (D to stop)"

    def _record(self, s):
        if self.t_s - self.rec_t0 > 120:
            self.notice = self.toggle_recording()
            return
        vals = [self.t_s, s["flags"]]
        for i in (0, 1):
            vals += list(s["q"][i]) + list(s["a"][i])
        vals += self.psi + list(self.pos) + list(self.vel) + [int(self.is_still), self.ang_rate]
        self.rec.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in vals) + "\n")

    def tare(self):
        idx = [i for i in (0, 1) if self.valid[i] and self.q_pen_raw[i] is not None]
        if not idx:
            return "No data yet"
        new = list(self.psi)
        for i in idx:
            x = qmat(self.q_pen_raw[i])[:, 0]          # where this sensor thinks the tip points
            if math.hypot(x[0], x[1]) < 0.3:
                return "Hold the pen more horizontally, then press C"
            new[i] = -math.atan2(x[1], x[0])
        self.psi = new
        self.tared = True
        self.reset_motion()
        self._update_pose()
        return "Heading zeroed"

    def rate_hz(self):
        now = time.time()
        return sum(1 for t in self.rx_times if now - t <= 1.0)

    def euler_deg(self):
        R = self.R_fused
        heading = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, R[2, 0]))))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        return heading, elevation, roll


# =============================================================================
# Tiny software 3D renderer (pygame only, no OpenGL needed)
# =============================================================================
def newell(pts):
    n = np.zeros(3)
    for i in range(len(pts)):
        c, d = pts[i], pts[(i + 1) % len(pts)]
        n[0] += (c[1] - d[1]) * (c[2] + d[2])
        n[1] += (c[2] - d[2]) * (c[0] + d[0])
        n[2] += (c[0] - d[0]) * (c[1] + d[1])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def frustum(x0, x1, r0, r1, color, sides=6, seg=0.012, caps=(True, True)):
    """A hexagonal tube along +X, split into short segments so depth sorting looks right."""
    n = max(1, int(math.ceil(abs(x1 - x0) / seg)))
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)

    def ring(x, r):
        return np.column_stack([np.full(sides, x), r * np.cos(ang), r * np.sin(ang)])

    polys = []
    for k in range(n):
        xa, xb = x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n
        ra, rb = r0 + (r1 - r0) * k / n, r0 + (r1 - r0) * (k + 1) / n
        a, b = ring(xa, ra), ring(xb, rb)
        for i in range(sides):
            j = (i + 1) % sides
            polys.append((np.array([a[i], a[j], b[j], b[i]]), color))
    if caps[0] and r0 > 1e-6:
        polys.append((ring(x0, r0), color))
    if caps[1] and r1 > 1e-6:
        polys.append((ring(x1, r1), color))
    return polys


def box(center, size, color):
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)
    v = np.array([[cx + dx * sx, cy + dy * sy, cz + dz * sz]
                  for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [(v[list(f)], color) for f in faces]


class Mesh:
    def __init__(self, polys):
        verts, normals, self.polys, idx = [], [], [], 0
        for pts, color in polys:
            verts.append(pts)
            normals.append(newell(pts))
            self.polys.append((idx, len(pts), color))
            idx += len(pts)
        self.verts = np.vstack(verts)
        self.normals = np.array(normals)

    def emit(self, R, p, cam, light, out):
        world = self.verts @ R.T + p
        sx, sy, z = cam.project(world)
        pts2d = np.column_stack([sx, sy]).tolist()
        shade = 0.35 + 0.65 * np.abs((self.normals @ R.T) @ light)
        for k, (i0, n, color) in enumerate(self.polys):
            zz = z[i0:i0 + n]
            if zz.min() < 0.02:
                continue
            c = tuple(min(255, int(ch * shade[k])) for ch in color)
            out.append((float(zz.mean()), c, pts2d[i0:i0 + n]))


class Camera:
    def __init__(self):
        self.az = math.radians(-65)
        self.el = math.radians(22)
        self.dist = 0.42
        self.target = np.array([-0.03, 0.0, 0.02])

    def update(self, w, h):
        self.cx, self.cy, self.fl = w / 2, h / 2, 1.15 * h
        self.eye = self.target + self.dist * np.array([
            math.cos(self.el) * math.cos(self.az),
            math.cos(self.el) * math.sin(self.az),
            math.sin(self.el)])
        self.f = (self.target - self.eye) / self.dist
        r = np.cross(self.f, [0.0, 0.0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)

    def project(self, pts):
        d = np.asarray(pts) - self.eye
        x, y, z = d @ self.r, d @ self.u, d @ self.f
        zs = np.maximum(z, 1e-6)
        return self.cx + self.fl * x / zs, self.cy - self.fl * y / zs, z


LIGHT = np.array([0.35, -0.45, 0.82])
LIGHT /= np.linalg.norm(LIGHT)

PAPER = (247, 245, 236)
RULE_BLUE = (170, 200, 230)
MARGIN_RED = (230, 150, 150)
GRAPHITE = (95, 95, 105)
GRAPHITE_LIVE = (150, 150, 165)

COL_REAR = (70, 170, 255)
COL_TIP = (255, 110, 70)


def build_pen_mesh():
    R = PEN_RADIUS
    polys = []
    polys += frustum(-0.098, -0.088, R * 0.95, R * 0.95, (238, 130, 150))             # eraser
    polys += frustum(-0.088, -0.072, R * 1.02, R * 1.02, (190, 192, 200), seg=0.008)  # ferrule
    polys += frustum(-0.072, 0.050, R, R, (250, 200, 40))                             # body
    polys += frustum(0.050, 0.070, R, 0.0020, (222, 184, 135), caps=(False, False))   # sharpened wood
    polys += frustum(0.070, TIP_POINT_X, 0.0020, 0.0, (70, 70, 75), caps=(False, False))  # graphite
    return Mesh(polys)


def build_sensor_mesh(color):
    return Mesh(box((0, 0, 0), (0.016, 0.011, 0.003), color))


# =============================================================================
# Application window
# =============================================================================
class App:
    def __init__(self, rx_queue, source_text, size=(1100, 760)):
        pygame.init()
        pygame.display.set_caption("Pen Tracker - 3D viewer")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 16)
        self.small = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 13)
        self.big = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 20, bold=True)
        self.rx = rx_queue
        self.source = source_text
        self.tracker = Tracker()
        self.cam = Camera()
        self.pen_mesh = build_pen_mesh()
        self.sensor_meshes = [build_sensor_mesh(COL_REAR), build_sensor_mesh(COL_TIP)]
        self.show_axes = True
        self.follow = True
        self.top_view = None
        self.page_view = False
        self.page_zoom = 1.0
        self.page_pan = [0.0, 0.0]
        self.dragging = False
        self.msg = ("", 0.0)

    # ---- events / data -----------------------------------------------------
    def handle_events(self):
        t = self.tracker
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if t.cal_step:
                        self.say(t.cancel_calibration())
                    elif t.acc_step:
                        t.acc_step = 0
                        self.say("Accelerometer calibration closed")
                    else:
                        return False
                elif e.key == pygame.K_g:
                    self.say(t.start_acc_calibration())
                elif e.key == pygame.K_d:
                    self.say(t.toggle_recording())
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    k = 1.25 if e.key == pygame.K_RIGHTBRACKET else 0.8
                    t.hp_tau = max(0.3, min(4.0, t.hp_tau * k))
                    self.say(f"Drift filter {t.hp_tau:.2f} s  ([ = less drift but smaller letters, ] = the reverse)")
                elif e.key == pygame.K_q:
                    return False
                elif e.key == pygame.K_k:
                    self.say(t.start_calibration())
                elif e.key == pygame.K_c:
                    self.say(t.tare())
                elif e.key == pygame.K_p:
                    t.track_position = not t.track_position
                    t.reset_motion()
                    self.say("Position tracking " + ("ON" if t.track_position else "OFF"))
                elif e.key == pygame.K_w:
                    t.paper_mode = not t.paper_mode
                    t.reset_motion()
                    self.say("Paper mode " + ("ON - tip locked to the table" if t.paper_mode else "OFF"))
                elif e.key == pygame.K_r:
                    t.reset_motion()
                    self.say("Drawing cleared")
                elif e.key == pygame.K_e:
                    self.say(t.export_drawing())
                elif e.key == pygame.K_v:
                    if self.top_view is None:
                        self.top_view = (self.cam.az, self.cam.el)
                        self.cam.az, self.cam.el = -math.pi / 2, 1.5
                        self.say("Top view (V to go back)")
                    else:
                        self.cam.az, self.cam.el = self.top_view
                        self.top_view = None
                elif e.key == pygame.K_f:
                    self.follow = not self.follow
                    self.say("Camera follow " + ("ON" if self.follow else "OFF"))
                elif e.key == pygame.K_n:
                    self.page_view = not self.page_view
                    self.page_zoom, self.page_pan = 1.0, [0.0, 0.0]
                elif e.key == pygame.K_l:
                    t.yaw_lock = not t.yaw_lock
                    self.say("Yaw lock " + ("ON" if t.yaw_lock else "OFF"))
                elif e.key == pygame.K_a:
                    self.show_axes = not self.show_axes
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self.dragging = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.dragging = False
            elif e.type == pygame.MOUSEMOTION and self.dragging:
                if self.page_view:
                    self.page_pan[0] += e.rel[0]
                    self.page_pan[1] += e.rel[1]
                else:
                    self.cam.az -= e.rel[0] * 0.007
                    self.cam.el = max(-1.5, min(1.5, self.cam.el + e.rel[1] * 0.007))
                    self.top_view = None
            elif e.type == pygame.MOUSEWHEEL:
                if self.page_view:
                    self.page_zoom = max(0.5, min(8.0, self.page_zoom * (1.15 ** e.y)))
                else:
                    self.cam.dist = max(0.15, min(3.0, self.cam.dist * (0.9 ** e.y)))
        return True

    def say(self, text):
        self.msg = (text, time.time())

    def update(self):
        for _ in range(500):
            try:
                self.tracker.feed(self.rx.get_nowait())
            except queue.Empty:
                break
        if self.tracker.notice:
            self.say(self.tracker.notice)
            self.tracker.notice = None

    # ---- drawing -----------------------------------------------------------
    def line3(self, surf, p0, p1, color, width=1):
        sx, sy, z = self.cam.project(np.array([p0, p1]))
        if z.min() > 0.02:
            pygame.draw.line(surf, color, (sx[0], sy[0]), (sx[1], sy[1]), width)

    def draw_grid(self, surf):
        step, n = 0.05, 6
        half = n * step
        gx = round(self.cam.target[0] / step) * step       # grid follows the camera
        gy = round(self.cam.target[1] / step) * step
        for k in range(-n, n + 1):
            x, y = gx + k * step, gy + k * step
            self.line3(surf, (x, gy - half, GRID_Z), (x, gy + half, GRID_Z),
                       (95, 102, 120) if abs(x) < 1e-9 else (60, 66, 80))
            self.line3(surf, (gx - half, y, GRID_Z), (gx + half, y, GRID_Z),
                       (95, 102, 120) if abs(y) < 1e-9 else (60, 66, 80))
        o = np.array([gx - half, gy - half, GRID_Z])
        for axis, col, name in ((0, (230, 80, 80), "X"), (1, (90, 210, 90), "Y"), (2, (90, 140, 255), "Z")):
            e = o.copy()
            e[axis] += 0.06
            self.line3(surf, o, e, col, 2)
            sx, sy, z = self.cam.project(e)
            if z > 0.02:
                surf.blit(self.small.render(name, True, col), (sx + 4, sy - 8))

    def _polyline(self, surf, pts, color, width):
        if len(pts) < 2:
            return
        sx, sy, z = self.cam.project(np.asarray(pts))
        if z.min() > 0.02:
            pygame.draw.lines(surf, color, False, np.column_stack([sx, sy]).tolist(), width)

    def draw_trail(self, surf):
        t = self.tracker
        for s in t.strokes:                                   # finished, corrected strokes
            self._polyline(surf, s, GRAPHITE, 2)
        if t.seg:                                             # stroke in progress (uncorrected)
            self._polyline(surf, t.seg_tips(), GRAPHITE_LIVE, 2)

    def draw_sheet3d(self, surf):
        """The sheet of paper lying on the table, seen from the 3D camera."""
        corners = [page_mm_to_world(x, y) for x, y in
                   ((0, 0), (PAGE_W_MM, 0), (PAGE_W_MM, PAGE_H_MM), (0, PAGE_H_MM))]
        sx, sy, z = self.cam.project(np.array(corners))
        if z.min() <= 0.02:
            return
        pygame.draw.polygon(surf, PAPER, np.column_stack([sx, sy]).tolist())
        y = PAGE_TOP_LINE_MM
        while y < PAGE_H_MM - 10:
            self.line3(surf, page_mm_to_world(0, y), page_mm_to_world(PAGE_W_MM, y), RULE_BLUE)
            y += PAGE_RULE_MM
        self.line3(surf, page_mm_to_world(PAGE_MARGIN_MM, 0),
                   page_mm_to_world(PAGE_MARGIN_MM, PAGE_H_MM), MARGIN_RED)

    def draw_page(self, surf, w, h):
        """Flat, true-to-scale view of the sheet with what has been written on it."""
        t = self.tracker
        surf.fill(BG)
        s = min((w - 40) / PAGE_W_MM, (h - 90) / PAGE_H_MM) * self.page_zoom
        ox = w / 2 - PAGE_W_MM * s / 2 + self.page_pan[0]
        oy = 50 + (h - 90 - PAGE_H_MM * s) / 2 + self.page_pan[1]

        def px(p):
            x, y = world_to_page_mm(p)
            return (ox + x * s, oy + y * s)

        surf.set_clip((0, 48, w, h - 48))                    # keep ink out of the header
        pygame.draw.rect(surf, (12, 13, 16), (ox + 5, oy + 5, PAGE_W_MM * s, PAGE_H_MM * s))   # shadow
        pygame.draw.rect(surf, PAPER, (ox, oy, PAGE_W_MM * s, PAGE_H_MM * s))
        y = PAGE_TOP_LINE_MM
        while y < PAGE_H_MM - 10:
            pygame.draw.line(surf, RULE_BLUE, (ox, oy + y * s), (ox + PAGE_W_MM * s, oy + y * s))
            y += PAGE_RULE_MM
        pygame.draw.line(surf, MARGIN_RED, (ox + PAGE_MARGIN_MM * s, oy),
                         (ox + PAGE_MARGIN_MM * s, oy + PAGE_H_MM * s))

        width = max(1, int(round(0.5 * s)))                  # a 0.5 mm pencil line
        for st in t.strokes:
            if len(st) > 1:
                pygame.draw.lines(surf, GRAPHITE, False, [px(p) for p in st], width)
        if len(t.seg) > 1:
            pygame.draw.lines(surf, GRAPHITE_LIVE, False, [px(p) for p in t.seg_tips()], width)

        # the pencil itself, as a shadow from the tip toward the eraser
        tip = t.pos
        rear = tip - t.R_fused[:, 0] * PENCIL_LEN
        shadow = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.line(shadow, (40, 40, 50, 70), px(tip), px(rear), max(3, int(7 * s / 2)))
        surf.blit(shadow, (0, 0))
        pygame.draw.circle(surf, (230, 90, 60), [int(v) for v in px(tip)], max(3, int(1.2 * s)))
        surf.set_clip(None)

        heading, elev, roll = t.euler_deg()
        lines = [f"Page view (US Letter, true scale)   pencil angle to paper {abs(elev):4.0f} deg   "
                 f"strokes {len(t.strokes)}",
                 "N: back to 3D   wheel: zoom   drag: pan   R: clear page   E: save page as .svg   "
                 "C: zero heading (pencil pointing away from you)"]
        for i, ln in enumerate(lines):
            surf.blit((self.font if i == 0 else self.small).render(ln, True, (225, 230, 240)), (12, 10 + 22 * i))
        text, when = self.msg
        if text and time.time() - when < 3.0:
            m = self.font.render(text, True, (140, 255, 170))
            surf.blit(m, (w // 2 - m.get_width() // 2, h - 30))

    def draw(self):
        surf = pygame.display.get_surface()
        w, h = surf.get_size()
        surf.fill(BG)
        t = self.tracker
        if self.page_view:
            self.draw_page(surf, w, h)
            if t.cal_step:
                self.draw_calibration(surf, w, h)
            elif t.acc_step:
                self.draw_acc_calibration(surf, w, h)
            return
        if self.follow:
            goal = t.pos + np.array([-0.03, 0.0, 0.02])
            self.cam.target = self.cam.target + 0.08 * (goal - self.cam.target)
        self.cam.update(w, h)

        self.draw_grid(surf)
        self.draw_sheet3d(surf)
        self.draw_trail(surf)

        # drop line from the tip to the table, so height is readable
        tip = t.pos
        self.line3(surf, tip, (tip[0], tip[1], GRID_Z), (110, 115, 130))

        polys = []
        self.pen_mesh.emit(t.R_fused, t.pen_center(), self.cam, LIGHT, polys)
        for i in (0, 1):
            if t.valid[i]:
                self.sensor_meshes[i].emit(t.R_sensor[i], t.sensor_world(i), self.cam, LIGHT, polys)
        polys.sort(key=lambda p: -p[0])
        for _, color, pts in polys:
            pygame.draw.polygon(surf, color, pts)
            pygame.draw.polygon(surf, tuple(int(c * 0.55) for c in color), pts, 1)

        for i in (0, 1):
            if not t.valid[i]:
                continue
            origin = t.sensor_world(i)
            if self.show_axes:
                for col_idx, col in enumerate(((235, 70, 70), (80, 220, 80), (80, 130, 255))):
                    self.line3(surf, origin, origin + t.R_sensor[i][:, col_idx] * 0.03, col, 2)
            sx, sy, z = self.cam.project(origin)
            if z > 0.02:
                label = SENSOR_NAMES[i]
                surf.blit(self.small.render(label, True, COL_REAR if i == 0 else COL_TIP),
                          (sx + 10, sy - 30))

        self.draw_hud(surf, w, h)
        if t.cal_step:
            self.draw_calibration(surf, w, h)
        elif t.acc_step:
            self.draw_acc_calibration(surf, w, h)

    def draw_acc_calibration(self, surf, w, h):
        t = self.tracker
        poses = t.ACC_POSES
        pw, ph = min(w - 40, 760), 112 + 19 * len(poses)
        x0, y0 = (w - pw) // 2, h - ph - 40
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 220))
        surf.blit(panel, (x0, y0))
        pygame.draw.rect(surf, (255, 190, 90), (x0, y0, pw, ph), 2)
        y = y0 + 12
        surf.blit(self.big.render("Accelerometer calibration", True, (235, 240, 255)), (x0 + 18, y))
        y += 30
        surf.blit(self.small.render("Hold each pose still for ~2 s (resting it on the table is best). "
                                    "Order doesn't matter much.", True, (200, 206, 220)), (x0 + 18, y))
        y += 22
        for k, (name, _) in enumerate(poses):
            n = k + 1
            if n < t.acc_step:
                mark, col = "[done]", (140, 255, 170)
            elif n == t.acc_step:
                mark, col = "  >>  ", (255, 230, 150)
            else:
                mark, col = "      ", (150, 158, 175)
            surf.blit(self.font.render(f"{mark} {n}. {name}", True, col), (x0 + 18, y))
            y += 19
        y += 8
        kinds = {0: "not yet", 1: "offset only", 2: "COMPLETE"}
        fw = "  |  ".join(f"{SENSOR_NAMES[i]}: {kinds[t.acc_cal[i]]} ({t.acc_poses[i]} poses)"
                          for i in (0, 1) if t.valid[i])
        if t.acc_step <= len(poses):
            status = (f"Pose {t.acc_step}: {t.acc_err:4.0f} deg away" if t.acc_progress == 0
                      else "Hold still...")
            surf.blit(self.font.render(status, True, (140, 255, 170)), (x0 + 18, y))
            bw = pw - 330
            pygame.draw.rect(surf, (60, 66, 80), (x0 + 300, y + 4, bw, 10))
            pygame.draw.rect(surf, (140, 255, 170), (x0 + 300, y + 4, int(bw * t.acc_progress), 10))
        else:
            done = all(t.acc_cal[i] == 2 for i in (0, 1) if t.valid[i])
            surf.blit(self.font.render("All poses done - " + ("calibration COMPLETE, press Esc"
                                       if done else "waiting for the pen to confirm (or repeat a pose)"),
                                       True, (140, 255, 170)), (x0 + 18, y))
        y += 22
        surf.blit(self.small.render("Pen: " + fw, True, (225, 230, 240)), (x0 + 18, y))
        if not t.cal_loaded:
            surf.blit(self.small.render("Tip: do the K calibration first so these poses are recognised.",
                                        True, (255, 190, 110)), (x0 + 18, y + 17))
        surf.blit(self.small.render("Esc: close", True, (150, 158, 175)), (x0 + pw - 90, y0 + 14))

    def draw_calibration(self, surf, w, h):
        t = self.tracker
        if t.cal_step == 1:
            title = "Calibration 1/2: lay the pen FLAT on the table"
            sub = ["Tip pointing straight away from you (this becomes 'forward').",
                   "Sensors can face any way - just let the pen rest naturally."]
        else:
            title = "Calibration 2/2: stand the pen STRAIGHT UP"
            sub = ["Tip pointing at the ceiling, as vertical as you can.",
                   "Lean it against something if that helps you hold it still."]
        status = "Hold still..." if t.cal_progress > 0 else "Waiting for the pen to be still in this pose"
        pw, ph = min(w - 40, 720), 190
        x0, y0 = (w - pw) // 2, h - ph - 50
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 215))
        surf.blit(panel, (x0, y0))
        pygame.draw.rect(surf, (90, 160, 255), (x0, y0, pw, ph), 2)
        y = y0 + 14
        surf.blit(self.big.render(title, True, (235, 240, 255)), (x0 + 18, y))
        y += 34
        for ln in sub:
            surf.blit(self.font.render(ln, True, (200, 206, 220)), (x0 + 18, y))
            y += 22
        y += 8
        surf.blit(self.font.render(status, True, (140, 255, 170)), (x0 + 18, y))
        y += 26
        bw = pw - 36
        pygame.draw.rect(surf, (60, 66, 80), (x0 + 18, y, bw, 12))
        pygame.draw.rect(surf, (140, 255, 170), (x0 + 18, y, int(bw * t.cal_progress), 12))
        y += 20
        if t.cal_msg:
            surf.blit(self.small.render(t.cal_msg, True, (255, 190, 110)), (x0 + 18, y))
        surf.blit(self.small.render("Esc: cancel", True, (150, 158, 175)), (x0 + pw - 100, y0 + ph - 20))

    def draw_hud(self, surf, w, h):
        t = self.tracker
        now = time.time()
        receiving = (now - t.last_rx) < 1.0 and t.total > 0
        heading, elev, roll = t.euler_deg()
        lines = [f"Source : {self.source}" + (f"  ({t.src})" if t.src and "demo" not in t.src else "")]
        if receiving:
            lines.append(f"Link   : {t.rate_hz()} Hz, {t.lost} packets lost")
        else:
            lines.append("Link   : NO DATA")
        lines.append("Sensors: rear " + ("OK" if t.valid[0] else "missing") +
                     " | tip " + ("OK" if t.valid[1] else "missing"))
        if self.source.startswith("DEMO"):
            acc_txt = "n/a in demo"
        else:
            kinds = {0: "not yet", 1: "offset", 2: "full"}
            acc_txt = " | ".join(f"{SENSOR_NAMES[i]} {kinds[t.acc_cal[i]]} ({t.acc_poses[i]} poses)"
                                 for i in (0, 1) if t.valid[i])
        lines.append("Accel  : " + acc_txt)
        lines.append("Mount  : " + ("calibrated (pen_calibration.json)" if t.cal_loaded
                                    else "defaults - press K to calibrate"))
        lines.append(f"Pen    : heading {heading:6.1f}  tip elevation {elev:6.1f}  roll {roll:6.1f} deg"
                     + ("" if t.tared else "   (heading not zeroed - press C)"))
        if t.disagree_deg is not None:
            lines.append(f"Rear vs tip disagreement: {t.disagree_deg:5.1f} deg"
                         f"   yaw lock {'on' if t.yaw_lock else 'OFF'}")
        pos_txt = "off"
        if t.track_position:
            p = t.pos * 100
            pos_txt = (f"tip ({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:6.1f}) cm   "
                       f"{'STILL' if t.is_still else 'moving'}   strokes {len(t.strokes)}"
                       f"   drift filter {t.hp_tau:.2f}s")
            if t.paper_mode:
                pos_txt += "  [paper]"
        lines.append("Position: " + pos_txt)
        y = 10
        for ln in lines:
            surf.blit(self.font.render(ln, True, (225, 230, 240)), (12, y))
            y += 21
        if (receiving and not self.source.startswith("DEMO") and not t.acc_step
                and any(t.valid[i] and t.acc_cal[i] < 2 for i in (0, 1))):
            surf.blit(self.small.render("Accelerometers not fully calibrated - press G and follow the poses "
                                        "(otherwise the tip will fly off when you move).",
                                        True, (255, 200, 120)), (12, y))
            y += 17
        if t.rec:
            surf.blit(self.font.render(f"REC  {t.t_s - t.rec_t0:5.1f} s  ->  {os.path.basename(t.rec_name)}"
                                       "   (D to stop)", True, (255, 90, 90)), (12, y))
            y += 21

        keys = ("K mount cal  G accel cal  C heading  P position  W paper  N page view  V top  "
                "F follow  E export  R clear  [ ] filter  D record  Esc quit")
        surf.blit(self.small.render(keys, True, (150, 158, 175)), (12, h - 22))

        if not receiving:
            if self.source.startswith("Bluetooth"):
                tips = ["Waiting for data...  " + BLE_STATUS["state"],
                        "1. Is the pen powered, with LINK_BLE (or LINK_AUTO) set in the sketch?",
                        "2. Is Bluetooth switched on on this computer? Keep the pen within a few metres.",
                        "3. Try:  python pen_viewer.py --demo   to test the viewer alone."]
            else:
                tips = ["Waiting for data...",
                        "1. Is the pen powered and on the same WiFi (or joined to the 'PenTracker' network)?",
                        "2. Windows Firewall: allow Python on private networks (UDP port %d)." % UDP_PORT,
                        "3. Bluetooth sketch?  python pen_viewer.py --ble     Try --demo to test the viewer alone."]
            yy = h // 2 + 120
            for i, ln in enumerate(tips):
                s = (self.font if i == 0 else self.small).render(ln, True, (240, 200, 120))
                surf.blit(s, (w // 2 - s.get_width() // 2, yy))
                yy += 24 if i == 0 else 19

        text, when = self.msg
        if text and now - when < 3.0:
            s = self.font.render(text, True, (140, 255, 170))
            surf.blit(s, (w // 2 - s.get_width() // 2, 14))

    def run(self):
        clock = pygame.time.Clock()
        while self.handle_events():
            self.update()
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="3D viewer for the two-IMU pen tracker")
    ap.add_argument("--port", type=int, default=UDP_PORT, help="UDP port to listen on (default %(default)s)")
    ap.add_argument("--serial", metavar="COMx", help="read from a USB serial port instead of WiFi")
    ap.add_argument("--ble", nargs="?", const="PenTracker", metavar="NAME",
                    help="connect over Bluetooth LE (default name PenTracker; or give a name/address)")
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--demo", action="store_true", help="show synthetic motion (no hardware)")
    args = ap.parse_args()

    rx = queue.Queue(maxsize=3000)
    stop = threading.Event()

    if args.demo:
        target, source = (demo_thread, (rx, stop)), "DEMO (synthetic data)"
    elif args.ble:
        check_ble_available()
        target, source = (ble_thread, (args.ble, rx, stop)), f"Bluetooth '{args.ble}'"
    elif args.serial:
        try:
            import serial  # noqa: F401
        except ImportError:
            sys.exit("--serial needs pyserial:  pip install pyserial")
        target, source = (serial_thread, (args.serial, args.baud, rx, stop)), f"USB serial {args.serial}"
    else:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("", args.port))
            probe.close()
        except OSError as exc:
            sys.exit(f"Cannot listen on UDP port {args.port}: {exc}")
        target, source = (udp_thread, (args.port, rx, stop)), f"WiFi UDP port {args.port}"

    threading.Thread(target=target[0], args=target[1], daemon=True).start()
    try:
        App(rx, source).run()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
