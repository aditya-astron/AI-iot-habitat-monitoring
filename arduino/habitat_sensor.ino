// ESP32 + DHT22 + Soil Moisture + MQ-135 + BH1750
#include <WiFi.h>
#include <PubSubClient.h>
// ... (full sensor code - 120 lines, commented with pinouts)
// Publishes JSON to topic "habitat/sensors" every 30s
// Ready for Raspberry Pi MQTT broker → Python ingestion
