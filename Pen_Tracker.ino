// ============================================================================
// Code/Pen_Tracker.ino  -  ESP32-S3 SuperMini + 2x MPU6050 -> WiFi (UDP) pen tracker
// ============================================================================
// Wiring (both sensors share the same I2C bus):
//   MPU6050 VCC -> 3V3     GND -> GND     SDA -> GPIO4     SCL -> GPIO5
//   REAR sensor: AD0 -> GND   (I2C address 0x68)
//   TIP  sensor: AD0 -> 3V3   (I2C address 0x69)
//
// Arduino IDE settings:
//   Board: "ESP32S3 Dev Module"     USB CDC On Boot: "Enabled"
//
// What it does:
//   * Runs a gravity-corrected gyro orientation filter for EACH sensor (100 Hz)
//   * AUTOMATIC ACCELEROMETER CALIBRATION: every time the pen rests still in a new
//     orientation, that reading is stored. From 4+ different resting orientations
//     it solves each accelerometer's offset (and from 6 well-spread ones, its
//     per-axis scale too). This is what stops gravity "leaking" into the motion
//     when you tilt the pen. The calibration is saved in flash and survives
//     power cycles. Easiest: press G in the viewer and follow the 6 poses.
//   * Sends one text line per sample (WiFi: UDP broadcast port 4210; BLE: UART notify):
//       $PEN,seq,micros,flags,
//            rear: qw,qx,qy,qz, ax,ay,az,      <- orientation + world linear accel (m/s^2)
//            tip : qw,qx,qy,qz, ax,ay,az
//     flags: bit0 = rear OK, bit1 = tip OK,
//            bit2/bit3 = rear/tip accel offset calibrated,
//            bit4/bit5 = rear/tip accel scale calibrated too,
//            bit6 = BOOT button held (used by pen_trainer / pen_live "hold" mode),
//            bits 8-11 / 12-15 = rear/tip number of stored rest poses
//   * Sends over Bluetooth LE or WiFi - choose with LINK below. Over WiFi it joins
//     your network, or starts its own access point ("PenTracker") if it can't.
//     Over Bluetooth it shows up as "PenTracker"; run the PC programs with --ble.
//
// Keep the pen still for ~4 seconds after power-up (gyro bias calibration).
// ============================================================================

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Preferences.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <math.h>

// ----------------------------- USER SETTINGS --------------------------------
// How the pen talks to the PC:
//   LINK_BLE   Bluetooth Low Energy - no WiFi needed. Run the PC programs with --ble
//              (pip install bleak). The ESP32-S3 only has BLE, not classic Bluetooth.
//   LINK_WIFI  join WIFI_SSID (falls back to its own "PenTracker" access point)
//   LINK_AUTO  try WiFi for 10 s, use Bluetooth if that fails
enum Link { LINK_BLE, LINK_WIFI, LINK_AUTO };
constexpr Link LINK = LINK_BLE;
const char *BLE_NAME = "PenTracker";  // the name the PC looks for

const char *WIFI_SSID = "YOUR_WIFI_NAME";
const char *WIFI_PASS = "YOUR_WIFI_PASSWORD";

const char *AP_SSID = "PenTracker";   // fallback access point
const char *AP_PASS = "pentracker";   // must be at least 8 characters

constexpr uint16_t UDP_PORT = 4210;

constexpr int SDA_PIN = 4;
constexpr int SCL_PIN = 5;
constexpr uint8_t ADDR_REAR = 0x68;
constexpr uint8_t ADDR_TIP  = 0x69;
constexpr int BUTTON_PIN = 0;          // the BOOT button on the ESP32-S3 SuperMini (pressed = LOW)

// Also print the data lines to the USB serial port (for "--serial COMx" mode
// in the Python viewer). Leave false for normal wireless use.
constexpr bool STREAM_OVER_USB = false;

// Set true for ONE upload to forget the learned accelerometer calibration
// (e.g. after re-mounting a sensor), then set it back to false and upload again.
constexpr bool RESET_ACCEL_CAL = false;
// ----------------------------------------------------------------------------

constexpr float GRAVITY = 9.80665f;
constexpr float DEG2RAD = 0.017453292f;
constexpr uint32_t SAMPLE_US = 10000;      // 100 Hz
constexpr int CAL_SAMPLES = 150;           // ~1.5 s per sensor at start-up

constexpr float KP_STILL = 5.0f;           // tilt correction while resting (fast re-levelling)
constexpr float KP_MOVE = 0.5f;            // tilt correction while moving (trust the gyro)
constexpr float GATE_MOVE = 0.05f;         // while moving, only correct if |a| is within 5% of 1 g

constexpr int STILL_BLOCK = 50;            // rest detection works on 0.5 s blocks
constexpr float STILL_GYRO = 0.05f;        // rad/s (~3 deg/s)
constexpr float STILL_ACC_SPREAD = 0.20f;  // m/s^2
constexpr float POSE_GYRO = 0.15f;         // looser limits for collecting calibration poses,
constexpr float POSE_ACC_SPREAD = 0.60f;   // so a pen held in the hand still counts

constexpr int MAX_POSES = 12;              // stored rest orientations per sensor
constexpr float POSE_SEP_DEG = 25.0f;      // a new pose must differ this much from stored ones

#define LOG(...) \
  do { if (Serial) Serial.printf(__VA_ARGS__); } while (0)

struct Imu {
  const char *name = "";
  uint8_t addr = 0;
  bool ok = false;
  uint8_t fails = 0;
  uint32_t lastTry = 0;
  float q[4] = {1, 0, 0, 0};      // orientation quaternion (sensor body -> world), W X Y Z
  float gyroBias[3] = {0, 0, 0};  // rad/s
  float lin[3] = {0, 0, 0};       // world-frame linear acceleration, gravity removed (m/s^2)

  // accelerometer calibration: true = (raw - accOff) * accScale
  float accOff[3] = {0, 0, 0};
  float accScale[3] = {1, 1, 1};
  int calKind = 0;                // 0 none, 1 offset, 2 offset + scale
  float poses[MAX_POSES][3] = {};
  uint8_t nPoses = 0;
  uint8_t nextSlot = 0;

  // rest detection
  int blk = 0;
  float aMin[3], aMax[3], aSum[3], gRawSum[3];
  float gMax = 0;
  bool still = false;
};

Imu rear, tip;
Preferences prefs;
WiFiUDP udp;
IPAddress destination;
uint32_t lastSample = 0;
uint32_t sequence = 0;

// ---- ACCEL CAL MATH BEGIN ---------------------------------------------------
// Solves an n x n linear system held as an augmented matrix a[i][0..n].
static bool solveLinear(double a[][7], int n, double x[]) {
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++)
      if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
    if (fabs(a[p][c]) < 1e-9) return false;
    if (p != c)
      for (int k = 0; k <= n; k++) { double t = a[c][k]; a[c][k] = a[p][k]; a[p][k] = t; }
    for (int r = 0; r < n; r++) {
      if (r == c) continue;
      double f = a[r][c] / a[c][c];
      for (int k = c; k <= n; k++) a[r][k] -= f * a[c][k];
    }
  }
  for (int i = 0; i < n; i++) x[i] = a[i][n] / a[i][i];
  return true;
}

static bool calSane(const double o[3], const double s[3]) {
  for (int i = 0; i < 3; i++)
    if (!(fabs(o[i]) < 0.3) || !(s[i] > 0.85 && s[i] < 1.15)) return false;   // also rejects NaN
  return true;
}

// Finds offset (and per-axis scale when the poses are spread widely enough) so that
// every resting reading has a magnitude of exactly 1 g. Poses are raw m/s^2.
// Returns 0 = not enough data / implausible, 1 = offset only, 2 = offset + scale.
static int fitAccelCal(const float poses[][3], int n, float off[3], float scale[3]) {
  if (n < 4) return 0;
  double p[MAX_POSES][3];
  double lo[3] = {1e9, 1e9, 1e9}, hi[3] = {-1e9, -1e9, -1e9};
  for (int k = 0; k < n; k++)
    for (int i = 0; i < 3; i++) {
      p[k][i] = poses[k][i] / GRAVITY;          // work in g for good numerics
      if (p[k][i] < lo[i]) lo[i] = p[k][i];
      if (p[k][i] > hi[i]) hi[i] = p[k][i];
    }
  // Every sensor axis must have been seen tilted clearly up AND clearly down, otherwise
  // its offset can't be told apart from a scale error and the fit would make things worse.
  // (0.45 g, not ~1 g: a board mounted at an angle on the pen never points an axis
  // straight up, but the 6 basic pen poses always give at least 0.57 g each way.)
  for (int i = 0; i < 3; i++)
    if (!(hi[i] > 0.45 && lo[i] < -0.45)) return 0;

  double o[3] = {0, 0, 0}, s[3] = {1, 1, 1};
  int kind = 0;

  if (n >= 6) {
    // A x^2 + B y^2 + C z^2 + D x + E y + F z = 1   (axis-aligned ellipsoid)
    double a[6][7] = {};
    for (int k = 0; k < n; k++) {
      double x = p[k][0], y = p[k][1], z = p[k][2];
      double row[6] = {x * x, y * y, z * z, x, y, z};
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) a[i][j] += row[i] * row[j];
        a[i][6] += row[i];
      }
    }
    double v[6];
    if (solveLinear(a, 6, v) && v[0] > 0 && v[1] > 0 && v[2] > 0) {
      double G = 1;
      for (int i = 0; i < 3; i++) {
        o[i] = -v[3 + i] / (2 * v[i]);
        G += v[i] * o[i] * o[i];
      }
      if (G > 0) {
        for (int i = 0; i < 3; i++) s[i] = sqrt(v[i] / G);
        if (calSane(o, s)) kind = 2;
      }
    }
  }

  if (kind == 0) {
    // Sphere: |x - o| = 1  ->  2 x.o + c = |x|^2
    double a[4][7] = {};
    for (int k = 0; k < n; k++) {
      double x = p[k][0], y = p[k][1], z = p[k][2];
      double row[4] = {2 * x, 2 * y, 2 * z, 1};
      double rhs = x * x + y * y + z * z;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) a[i][j] += row[i] * row[j];
        a[i][4] += row[i] * rhs;
      }
    }
    double v[4];
    if (!solveLinear(a, 4, v)) return 0;
    for (int i = 0; i < 3; i++) { o[i] = v[i]; s[i] = 1; }
    if (!calSane(o, s)) return 0;
    kind = 1;
  }

  for (int i = 0; i < 3; i++) {
    off[i] = (float)(o[i] * GRAVITY);
    scale[i] = (float)s[i];
  }
  return kind;
}
// ---- ACCEL CAL MATH END -----------------------------------------------------

// ------------------------------- I2C helpers --------------------------------
bool readBytes(uint8_t address, uint8_t reg, uint8_t *data, size_t count) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom(address, count) != count) return false;
  for (size_t i = 0; i < count; i++) data[i] = Wire.read();
  return true;
}

bool writeReg(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

int16_t signedWord(const uint8_t *b) {
  return (int16_t)(((uint16_t)b[0] << 8) | b[1]);
}

// Reads RAW accel (m/s^2) and gyro (rad/s). Ranges: +/-4 g and +/-500 deg/s.
bool readSample(const Imu &s, float acc[3], float gyro[3]) {
  uint8_t b[14];
  if (!readBytes(s.addr, 0x3B, b, 14)) return false;
  for (int i = 0; i < 3; i++) {
    acc[i]  = signedWord(b + 2 * i) * (GRAVITY / 8192.0f);
    gyro[i] = signedWord(b + 8 + 2 * i) * (DEG2RAD / 65.5f);
  }
  return true;
}

void applyAccelCal(const Imu &s, const float raw[3], float out[3]) {
  for (int i = 0; i < 3; i++) out[i] = (raw[i] - s.accOff[i]) * s.accScale[i];
}

// ------------------------- Calibration storage (flash) -----------------------
void refit(Imu &s) {
  float off[3], sc[3];
  int kind = fitAccelCal(s.poses, s.nPoses, off, sc);
  if (kind == 0) return;            // keep whatever we had
  for (int i = 0; i < 3; i++) { s.accOff[i] = off[i]; s.accScale[i] = sc[i]; }
  s.calKind = kind;
  LOG("# %s accel cal (%s): offset %.3f %.3f %.3f m/s^2, scale %.4f %.4f %.4f\n",
      s.name, kind == 2 ? "offset+scale" : "offset", off[0], off[1], off[2], sc[0], sc[1], sc[2]);
}

void loadCal(Imu &s) {
  char k[16];
  snprintf(k, sizeof(k), "%s_n", s.name);
  s.nPoses = prefs.getUChar(k, 0);
  if (s.nPoses > MAX_POSES) s.nPoses = 0;
  snprintf(k, sizeof(k), "%s_p", s.name);
  if (s.nPoses && prefs.getBytes(k, s.poses, sizeof(s.poses)) != sizeof(s.poses)) s.nPoses = 0;
  s.nextSlot = s.nPoses % MAX_POSES;
  LOG("# %s: %u stored rest poses\n", s.name, (unsigned)s.nPoses);
  refit(s);
}

void saveCal(Imu &s) {
  char k[16];
  snprintf(k, sizeof(k), "%s_p", s.name);
  prefs.putBytes(k, s.poses, sizeof(s.poses));
  snprintf(k, sizeof(k), "%s_n", s.name);
  prefs.putUChar(k, s.nPoses);
}

// Called with the mean RAW accel of a 0.5 s rest period.
void addPose(Imu &s, const float m[3]) {
  float n = sqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
  if (fabsf(n - GRAVITY) > 0.15f * GRAVITY) return;
  float cosSep = cosf(POSE_SEP_DEG * DEG2RAD);
  for (int k = 0; k < s.nPoses; k++) {
    const float *p = s.poses[k];
    float pn = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    if ((m[0] * p[0] + m[1] * p[1] + m[2] * p[2]) / (n * pn) > cosSep) return;   // already known
  }
  int slot;
  if (s.nPoses < MAX_POSES) slot = s.nPoses++;
  else { slot = s.nextSlot; s.nextSlot = (s.nextSlot + 1) % MAX_POSES; }
  for (int i = 0; i < 3; i++) s.poses[slot][i] = m[i];
  LOG("# %s: new rest pose stored (%u total)\n", s.name, (unsigned)s.nPoses);
  saveCal(s);
  refit(s);
}

// Rest detection on 0.5 s blocks; resting blocks feed the calibration and
// keep refining the gyro bias.
void trackStillness(Imu &s, const float aRaw[3], const float gRaw[3], const float gCorr[3]) {
  float gm = sqrtf(gCorr[0] * gCorr[0] + gCorr[1] * gCorr[1] + gCorr[2] * gCorr[2]);
  if (s.blk == 0) {
    for (int i = 0; i < 3; i++) {
      s.aMin[i] = s.aMax[i] = aRaw[i];
      s.aSum[i] = 0;
      s.gRawSum[i] = 0;
    }
    s.gMax = 0;
  }
  for (int i = 0; i < 3; i++) {
    if (aRaw[i] < s.aMin[i]) s.aMin[i] = aRaw[i];
    if (aRaw[i] > s.aMax[i]) s.aMax[i] = aRaw[i];
    s.aSum[i] += aRaw[i];
    s.gRawSum[i] += gRaw[i];
  }
  if (gm > s.gMax) s.gMax = gm;
  if (gm > 2 * STILL_GYRO) s.still = false;          // react to motion immediately

  if (++s.blk < STILL_BLOCK) return;
  s.blk = 0;

  bool still = s.gMax < STILL_GYRO;
  bool poseOk = s.gMax < POSE_GYRO;
  for (int i = 0; i < 3; i++) {
    float spread = s.aMax[i] - s.aMin[i];
    still = still && spread < STILL_ACC_SPREAD;
    poseOk = poseOk && spread < POSE_ACC_SPREAD;
  }
  s.still = still;

  float mean[3];
  for (int i = 0; i < 3; i++) mean[i] = s.aSum[i] / STILL_BLOCK;
  if (still)
    for (int i = 0; i < 3; i++)
      s.gyroBias[i] += 0.2f * (s.gRawSum[i] / STILL_BLOCK - s.gyroBias[i]);
  if (poseOk) addPose(s, mean);
}

// ----------------------------- Sensor start-up -------------------------------
bool startImu(Imu &s) {
  uint8_t id = 0;
  // Both MPU6050s report WHO_AM_I = 0x68, even the one at address 0x69.
  if (!readBytes(s.addr, 0x75, &id, 1) || id != 0x68) return false;

  if (!writeReg(s.addr, 0x6B, 0x01)) return false;  // wake up, gyro-X PLL clock
  delay(100);

  if (!writeReg(s.addr, 0x1A, 0x03) ||   // digital low-pass filter ~44 Hz
      !writeReg(s.addr, 0x19, 0x09) ||   // sample rate 1 kHz / (1+9) = 100 Hz
      !writeReg(s.addr, 0x1B, 0x08) ||   // gyro  +/-500 deg/s
      !writeReg(s.addr, 0x1C, 0x08))     // accel +/-4 g
    return false;
  delay(50);

  LOG("# %s sensor found at 0x%02X - keep the pen still\n", s.name, s.addr);

  float accSum[3] = {0, 0, 0}, gyroSum[3] = {0, 0, 0};
  for (int n = 0; n < CAL_SAMPLES; n++) {
    float a[3], g[3];
    if (!readSample(s, a, g)) return false;
    for (int i = 0; i < 3; i++) {
      accSum[i] += a[i];
      gyroSum[i] += g[i];
    }
    delay(10);
  }
  float accMean[3], acc[3];
  for (int i = 0; i < 3; i++) {
    s.gyroBias[i] = gyroSum[i] / CAL_SAMPLES;
    accMean[i] = accSum[i] / CAL_SAMPLES;
  }
  applyAccelCal(s, accMean, acc);

  // Initial tilt from gravity, initial yaw = 0.
  float roll  = atan2f(acc[1], acc[2]) / 2;
  float pitch = atan2f(-acc[0], hypotf(acc[1], acc[2])) / 2;
  s.q[0] = cosf(roll) * cosf(pitch);
  s.q[1] = sinf(roll) * cosf(pitch);
  s.q[2] = cosf(roll) * sinf(pitch);
  s.q[3] = -sinf(roll) * sinf(pitch);

  for (int i = 0; i < 3; i++) s.lin[i] = 0;
  s.fails = 0;
  s.blk = 0;
  s.still = true;
  return true;
}

// ----------------------------- Orientation filter ----------------------------
void updateOrientation(Imu &s, const float a[3], float g[3], float dt) {
  float w = s.q[0], x = s.q[1], y = s.q[2], z = s.q[3];
  float length = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

  // Correct tilt with gravity: strongly while resting, gently (and only when the
  // reading is very close to 1 g) while moving.
  float kp = s.still ? KP_STILL : KP_MOVE;
  float gate = s.still ? 0.10f : GATE_MOVE;
  if (length > 1e-3f && fabsf(length - GRAVITY) < gate * GRAVITY) {
    float ax = a[0] / length, ay = a[1] / length, az = a[2] / length;
    float ux = 2 * (x * z - w * y);
    float uy = 2 * (w * x + y * z);
    float uz = 1 - 2 * (x * x + y * y);

    g[0] += kp * (ay * uz - az * uy);
    g[1] += kp * (az * ux - ax * uz);
    g[2] += kp * (ax * uy - ay * ux);
  }

  float h = 0.5f * dt;
  s.q[0] += h * (-x * g[0] - y * g[1] - z * g[2]);
  s.q[1] += h * ( w * g[0] + y * g[2] - z * g[1]);
  s.q[2] += h * ( w * g[1] + z * g[0] - x * g[2]);
  s.q[3] += h * ( w * g[2] + x * g[1] - y * g[0]);

  float norm = sqrtf(s.q[0] * s.q[0] + s.q[1] * s.q[1] +
                     s.q[2] * s.q[2] + s.q[3] * s.q[3]);
  for (int i = 0; i < 4; i++) s.q[i] /= norm;
}

// Rotate the (calibrated) acceleration into world axes and remove gravity.
void updateLinearAccel(Imu &s, const float a[3]) {
  float w = s.q[0], x = s.q[1], y = s.q[2], z = s.q[3];
  float tx = 2 * (y * a[2] - z * a[1]);
  float ty = 2 * (z * a[0] - x * a[2]);
  float tz = 2 * (x * a[1] - y * a[0]);

  s.lin[0] = a[0] + w * tx + y * tz - z * ty;
  s.lin[1] = a[1] + w * ty + z * tx - x * tz;
  s.lin[2] = a[2] + w * tz + x * ty - y * tx - GRAVITY;
}

void stepImu(Imu &s, float dt) {
  if (!s.ok) {
    // Sensor missing or dropped out: try to (re)start it every 3 seconds.
    if (millis() - s.lastTry > 3000) {
      s.lastTry = millis();
      s.ok = startImu(s);
      if (s.ok) LOG("# %s sensor ready\n", s.name);
      lastSample = micros();
    }
    return;
  }

  float aRaw[3], gRaw[3];
  if (!readSample(s, aRaw, gRaw)) {
    if (++s.fails >= 20) {
      s.ok = false;
      LOG("# %s sensor lost - check wiring\n", s.name);
    }
    return;
  }
  s.fails = 0;

  float g[3], a[3];
  for (int i = 0; i < 3; i++) g[i] = gRaw[i] - s.gyroBias[i];
  trackStillness(s, aRaw, gRaw, g);
  applyAccelCal(s, aRaw, a);
  updateOrientation(s, a, g, dt);
  updateLinearAccel(s, a);
}

// --------------------------------- Network -----------------------------------
// Bluetooth: the standard "Nordic UART" service; each data line is sent as
// notifications (split if the connection's packet size is smaller than a line).
#define NUS_SERVICE "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX      "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX      "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

bool useWifi = false, useBle = false;
BLEServer *bleServer = nullptr;
BLECharacteristic *bleTx = nullptr;
volatile bool bleConnected = false;
uint16_t bleConnId = 0;

// Written against the common callbacks so it builds with either BLE stack the
// ESP32 Arduino core may use (Bluedroid: core 2.x/3.x, NimBLE: newer 3.x).
class PenServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    bleConnId = server->getConnId();
    bleConnected = true;
#if defined(CONFIG_NIMBLE_ENABLED)
    server->requestConnParams(bleConnId, 6, 12, 0, 400);   // 7.5-15 ms interval for 100 Hz
#endif
    LOG("# Bluetooth connected\n");
  }
  void onDisconnect(BLEServer *server) override {
    bleConnected = false;
    LOG("# Bluetooth disconnected - advertising again\n");
    BLEDevice::startAdvertising();
  }
#if defined(CONFIG_BLUEDROID_ENABLED)
  void onConnect(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
    server->updateConnParams(param->connect.remote_bda, 6, 12, 0, 400);   // 7.5-15 ms interval
  }
#endif
};

void startBle() {
  BLEDevice::init(BLE_NAME);
  BLEDevice::setMTU(247);                       // room for a whole line per packet
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new PenServerCallbacks());
  BLEService *svc = bleServer->createService(NUS_SERVICE);
  bleTx = svc->createCharacteristic(NUS_TX, BLECharacteristic::PROPERTY_NOTIFY);
#if defined(CONFIG_BLUEDROID_ENABLED)
  bleTx->addDescriptor(new BLE2902());          // NimBLE adds this descriptor by itself
#endif
  svc->createCharacteristic(NUS_RX, BLECharacteristic::PROPERTY_WRITE);   // unused
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
  useBle = true;
  LOG("# Bluetooth: advertising as \"%s\" - run the PC program with --ble\n", BLE_NAME);
}

void bleSend(const char *line, int n) {
  if (!bleConnected || bleTx == nullptr) return;
  uint16_t mtu = bleServer->getPeerMTU(bleConnId);
  int chunk = (mtu > 23 ? mtu : 23) - 3;
  for (int i = 0; i < n; i += chunk) {
    int len = (n - i < chunk) ? n - i : chunk;
    bleTx->setValue((uint8_t *)line + i, len);
    bleTx->notify();
  }
}

void startNetwork() {
  if (LINK == LINK_BLE) {
    WiFi.mode(WIFI_OFF);
    startBle();
    return;
  }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // lower latency
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  LOG("# Connecting to WiFi \"%s\" ...\n", WIFI_SSID);

  uint32_t t0 = millis();
  uint32_t limit = (LINK == LINK_AUTO) ? 10000 : 15000;
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < limit) delay(250);

  if (WiFi.status() == WL_CONNECTED) {
    useWifi = true;
    uint32_t ip = (uint32_t)WiFi.localIP();
    uint32_t mask = (uint32_t)WiFi.subnetMask();
    destination = IPAddress(ip | ~mask);   // subnet broadcast address
    LOG("# Connected. Pen IP %s, broadcasting to %s:%u\n",
        WiFi.localIP().toString().c_str(), destination.toString().c_str(), UDP_PORT);
  } else if (LINK == LINK_AUTO) {
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    LOG("# WiFi failed - using Bluetooth instead\n");
    startBle();
  } else {
    useWifi = true;
    WiFi.disconnect(true);
    WiFi.mode(WIFI_AP);
    WiFi.setSleep(false);
    WiFi.softAP(AP_SSID, AP_PASS);
    destination = IPAddress(192, 168, 4, 255);
    LOG("# WiFi failed. Started access point \"%s\" (password \"%s\").\n"
        "# Connect your PC to it, then run the viewer.\n", AP_SSID, AP_PASS);
  }
}

// ---------------------------------- Arduino ----------------------------------
void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) delay(10);   // don't hang on battery

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  rear.name = "rear";
  rear.addr = ADDR_REAR;
  tip.name = "tip";
  tip.addr = ADDR_TIP;

  prefs.begin("pencal", false);
  if (RESET_ACCEL_CAL) {
    prefs.clear();
    LOG("# Accelerometer calibration erased\n");
  }
  loadCal(rear);
  loadCal(tip);

  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  delay(200);

  LOG("# Keep the pen still for a few seconds...\n");
  rear.ok = startImu(rear);
  tip.ok = startImu(tip);
  LOG("# rear: %s, tip: %s\n", rear.ok ? "OK" : "missing", tip.ok ? "OK" : "missing");

  startNetwork();
  lastSample = micros();
}

void loop() {
  uint32_t now = micros();
  uint32_t elapsed = now - lastSample;
  if (elapsed < SAMPLE_US) return;      // run at 100 Hz
  lastSample = now;

  float dt = (elapsed > 100000) ? SAMPLE_US * 1e-6f : elapsed * 1e-6f;
  stepImu(rear, dt);
  stepImu(tip, dt);

  unsigned flags = (rear.ok ? 1u : 0u) | (tip.ok ? 2u : 0u) |
                   (rear.calKind ? 4u : 0u) | (tip.calKind ? 8u : 0u) |
                   (rear.calKind == 2 ? 16u : 0u) | (tip.calKind == 2 ? 32u : 0u) |
                   (digitalRead(BUTTON_PIN) == LOW ? 64u : 0u) |
                   ((unsigned)(rear.nPoses & 15) << 8) | ((unsigned)(tip.nPoses & 15) << 12);

  char line[256];
  int n = snprintf(line, sizeof(line),
      "$PEN,%lu,%lu,%u,"
      "%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,"
      "%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
      (unsigned long)sequence++, (unsigned long)now, flags,
      rear.q[0], rear.q[1], rear.q[2], rear.q[3], rear.lin[0], rear.lin[1], rear.lin[2],
      tip.q[0], tip.q[1], tip.q[2], tip.q[3], tip.lin[0], tip.lin[1], tip.lin[2]);
  if (n <= 0 || n >= (int)sizeof(line)) return;

  if (useWifi) {
    udp.beginPacket(destination, UDP_PORT);
    udp.write((const uint8_t *)line, (size_t)n);
    udp.endPacket();
  }
  if (useBle) bleSend(line, n);

  if (STREAM_OVER_USB && Serial) Serial.write((const uint8_t *)line, (size_t)n);
}
