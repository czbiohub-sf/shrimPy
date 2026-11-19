#define PWM_PIN 9

// Variables for PWM parameters
unsigned int dutyCycle = 0;
unsigned long duration = 0; // Duration in milliseconds
float pwmFrequency = 0.0; // Frequency as a float
unsigned long pwmPeriod = 0; // Period in milliseconds

void setup() {
  Serial.begin(115200);
  pinMode(PWM_PIN, OUTPUT);
  Serial.println("Send 'U,duty,freq,duration' to set PWM. Send 'S' to start PWM.");
}

void loop() {
  static unsigned long lastToggleTime = 0;
  static bool pinState = LOW;
  static bool pwmRunning = false;
  static unsigned long pwmStartTime = 0;

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');

    if (command.startsWith("U,")) {
      // Update PWM settings
      int firstComma = command.indexOf(',');
      int secondComma = command.indexOf(',', firstComma + 1);
      int thirdComma = command.indexOf(',', secondComma + 1);

      dutyCycle = command.substring(firstComma + 1, secondComma).toInt();
      pwmFrequency = command.substring(secondComma + 1, thirdComma).toFloat(); // Use float for frequency
      duration = command.substring(thirdComma + 1).toInt();

      // Convert frequency to period (milliseconds)
      if (pwmFrequency > 0) {
        pwmPeriod = (unsigned long)(1000.0 / pwmFrequency);
      } else {
        pwmPeriod = 0; // Avoid division by zero
      }

      Serial.println("Duty Cycle: " + String(dutyCycle));
      Serial.println("PWM Frequency (Hz): " + String(pwmFrequency));
      Serial.println("PWM Period (ms): " + String(pwmPeriod));
      Serial.println("Duration (ms): " + String(duration));

      Serial.println("PWM settings updated. Send 'S' to start.");
    }
    else if (command.startsWith("S")) {
      // Start PWM
      pwmRunning = true;
      pwmStartTime = millis();
      lastToggleTime = millis(); // Use millis() for longer periods
      pinState = HIGH;
      digitalWrite(PWM_PIN, pinState);
    }
  }

  if (pwmRunning && pwmPeriod > 0) {
    unsigned long currentTime = millis(); // Use millis() for compatibility with long periods
    unsigned long elapsedTime = currentTime - lastToggleTime;
    unsigned long highTime = pwmPeriod * dutyCycle / 100;
    unsigned long lowTime = pwmPeriod - highTime;

    // Toggle pin state based on the duty cycle
    if ((pinState == HIGH && elapsedTime >= highTime) || (pinState == LOW && elapsedTime >= lowTime)) {
      pinState = !pinState;
      digitalWrite(PWM_PIN, pinState);
      lastToggleTime = currentTime;
    }

    // Stop PWM after the specified duration
    if (currentTime - pwmStartTime >= duration) {
      pwmRunning = false;
      digitalWrite(PWM_PIN, LOW); // Ensure pin is low after stopping
      Serial.println("PWM cycle complete. Update settings or start again.");
    }
  }
}
