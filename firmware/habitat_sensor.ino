/*
 * Habitat node firmware.
 *
 * Emits the line-delimited JSON protocol documented in PROTOCOL.md and
 * parsed by habitat_monitor/protocol.py. Field names must stay in sync.
 *
 * Default build: HABITAT_SIMULATE 1 — no sensors required, useful on a
 * bare ESP32 / Uno for protocol bring-up. Set HABITAT_SIMULATE to 0 and
 * wire the pins below to read hardware.
 *
 * Pin map (ESP32 DevKit; change as needed)
 *   DHT22 data ........ GPIO 4
 *   Soil moisture A0 .. GPIO 34 (ADC1)
 *   MQ-135 analog ..... GPIO 35 (ADC1)
 *   BH1750 I2C ........ SDA 21 / SCL 22 (optional)
 *   PIR digital ....... GPIO 27
 *
 * Host commands (one JSON line):
 *   {"v":1,"type":"cmd","cmd":"ping"}
 *   {"v":1,"type":"cmd","cmd":"set_interval","interval_ms":15000}
 */

#ifndef HABITAT_SIMULATE
#define HABITAT_SIMULATE 1
#endif

#ifndef HABITAT_DEVICE_ID
#define HABITAT_DEVICE_ID "hab-01"
#endif

static const int kBaud = 115200;
static const int kMaxLine = 1024;
static const unsigned long kDefaultIntervalMs = 30000;
static const unsigned long kMinIntervalMs = 1000;
static const unsigned long kMaxIntervalMs = 3600000UL;

#if !HABITAT_SIMULATE
#include <DHT.h>
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);
static const int kSoilPin = 34;
static const int kMqPin = 35;
static const int kPirPin = 27;
#endif

static unsigned long gIntervalMs = kDefaultIntervalMs;
static unsigned long gLastEmitMs = 0;
static unsigned long gSeq = 0;
static unsigned int gMotionCount = 0;

#if HABITAT_SIMULATE
/* Cheap deterministic-looking walk so the host parser has changing values. */
static float walk(float base, float span) {
  const float t = millis() / 1000.0f;
  return base + span * sinf(t / 17.0f) + (float)((millis() / 100) % 7) * 0.05f;
}
#endif

static void emitReading() {
  float temperatureC;
  float humidityPct;
  float soilPct;
  float co2Ppm;
  float lightLux;
  int motion;
  const char *status = "ok";

#if HABITAT_SIMULATE
  temperatureC = walk(24.1f, 2.2f);
  humidityPct = walk(61.0f, 6.0f);
  soilPct = walk(40.0f, 5.0f);
  co2Ppm = walk(430.0f, 25.0f);
  lightLux = walk(700.0f, 120.0f);
  motion = (int)((millis() / 1000) % 3);
#else
  temperatureC = dht.readTemperature();
  humidityPct = dht.readHumidity();
  if (isnan(temperatureC) || isnan(humidityPct)) {
    status = "degraded";
    temperatureC = 0.0f;
    humidityPct = 0.0f;
  }
  int soilRaw = analogRead(kSoilPin);
  soilPct = 100.0f - (soilRaw / 4095.0f) * 100.0f;
  int mqRaw = analogRead(kMqPin);
  /* Linear map, not a calibrated MQ-135 model. */
  co2Ppm = 350.0f + (mqRaw / 4095.0f) * 1200.0f;
  lightLux = 0.0f; /* Wire BH1750 and replace this if available. */
  motion = gMotionCount;
  gMotionCount = 0;
#endif

  char line[kMaxLine];
  int n = snprintf(
      line, sizeof(line),
      "{\"v\":1,\"type\":\"reading\",\"device_id\":\"%s\",\"seq\":%lu,\"ts_ms\":%lu,"
      "\"sensors\":{\"temperature_c\":%.2f,\"humidity_pct\":%.2f,\"soil_moisture_pct\":%.2f,"
      "\"co2_ppm\":%.1f,\"light_lux\":%.1f,\"motion\":%d},\"status\":\"%s\"}",
      HABITAT_DEVICE_ID, gSeq, millis(), temperatureC, humidityPct, soilPct, co2Ppm,
      lightLux, motion, status);
  if (n > 0 && n < (int)sizeof(line)) {
    Serial.println(line);
    gSeq++;
  }
}

static void emitAckPong() {
  char line[kMaxLine];
  snprintf(line, sizeof(line),
           "{\"v\":1,\"type\":\"ack\",\"device_id\":\"%s\",\"ack\":\"pong\",\"uptime_ms\":%lu}",
           HABITAT_DEVICE_ID, millis());
  Serial.println(line);
}

static void emitAckInterval() {
  char line[kMaxLine];
  snprintf(line, sizeof(line),
           "{\"v\":1,\"type\":\"ack\",\"device_id\":\"%s\",\"ack\":\"set_interval\","
           "\"interval_ms\":%lu}",
           HABITAT_DEVICE_ID, gIntervalMs);
  Serial.println(line);
}

static bool contains(const char *hay, const char *needle) {
  return strstr(hay, needle) != NULL;
}

static long extractIntervalMs(const char *line) {
  const char *key = "\"interval_ms\"";
  const char *p = strstr(line, key);
  if (p == NULL) {
    return -1;
  }
  p = strchr(p, ':');
  if (p == NULL) {
    return -1;
  }
  return strtol(p + 1, NULL, 10);
}

static void handleHostLine(char *line) {
  if (!contains(line, "\"type\":\"cmd\"")) {
    return;
  }
  if (contains(line, "\"cmd\":\"ping\"")) {
    emitAckPong();
    return;
  }
  if (contains(line, "\"cmd\":\"set_interval\"")) {
    long ms = extractIntervalMs(line);
    if (ms >= (long)kMinIntervalMs && ms <= (long)kMaxIntervalMs) {
      gIntervalMs = (unsigned long)ms;
    }
    emitAckInterval();
  }
}

static void pollHost() {
  static char buf[kMaxLine];
  static int len = 0;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      buf[len] = '\0';
      if (len > 0) {
        handleHostLine(buf);
      }
      len = 0;
      continue;
    }
    if (len < kMaxLine - 1) {
      buf[len++] = c;
    } else {
      len = 0; /* drop oversized line */
    }
  }
}

void setup() {
  Serial.begin(kBaud);
  delay(200);
#if !HABITAT_SIMULATE
  dht.begin();
  pinMode(kPirPin, INPUT);
#endif
  Serial.println("{\"v\":1,\"type\":\"ack\",\"device_id\":\"" HABITAT_DEVICE_ID
                 "\",\"ack\":\"pong\",\"uptime_ms\":0}");
}

void loop() {
  pollHost();
#if !HABITAT_SIMULATE
  if (digitalRead(kPirPin) == HIGH) {
    gMotionCount++;
    delay(10);
  }
#endif
  unsigned long now = millis();
  if (now - gLastEmitMs >= gIntervalMs) {
    gLastEmitMs = now;
    emitReading();
  }
}
