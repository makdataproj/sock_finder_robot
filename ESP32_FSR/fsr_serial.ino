#define FORCE_SENSOR_PIN 33

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
}

void loop() {
  int raw = analogRead(FORCE_SENSOR_PIN);
  float voltage = raw * (3.3 / 4095.0);

  Serial.print("Raw: ");
  Serial.print(raw);
  Serial.print(" Voltage: ");
  Serial.println(voltage, 3);

  delay(100);
}