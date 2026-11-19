#define PWM_PIN 9

// Variables for PWM parameters
unsigned int dutyCycle = 0;
unsigned long duration = 0;
unsigned long pwmFrequency = 0;

void setup() {
  Serial.begin(115200);
  pinMode(PWM_PIN, OUTPUT);
  Serial.println("Send 'U,duty,frequency,duration' to set PWM. Send 'S' to start PWM.");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');

    if (command.startsWith("U,")) {
      // Update PWM settings
      int firstComma = command.indexOf(',');
      int secondComma = command.indexOf(',', firstComma + 1);
      int thirdComma = command.indexOf(',', secondComma + 1);

      dutyCycle = command.substring(firstComma + 1, secondComma).toInt();
      pwmFrequency = command.substring(secondComma + 1, thirdComma).toInt();
      duration = command.substring(thirdComma + 1).toInt();

      Serial.println("Duty Cycle");
      Serial.println(dutyCycle);
      Serial.println("PWM Freq");
      Serial.println(pwmFrequency);
      Serial.println("Duration");
      Serial.println(duration);
      setPWMFrequency(pwmFrequency);

      Serial.println("PWM updated. Send 'S' to start.");
    }
    else if (command.startsWith("S")) {
      // Start PWM with the set parameters
      int pwmValue = map(dutyCycle, 0, 100, 0, 255);
      analogWrite(PWM_PIN, pwmValue);

      // Wait for the specified duration to stop PWM
      delay(duration);
      analogWrite(PWM_PIN, 0);

      // Notify completion
      Serial.println("PWM cycle complete. Update settings or start again.");
    }
  }
}

void setPWMFrequency(unsigned long frequency) {
  // Stop the timer
  TCCR1B &= 0b11111000;

  // Calculate and set the appropriate prescaler for the desired frequency
  if (frequency < 500) {
    TCCR1B |= 0b101; // Prescaler set to 1024
  } else if (frequency < 1000) {
    TCCR1B |= 0b100; // Prescaler set to 256
  } else if (frequency < 4000) {
    TCCR1B |= 0b011; // Prescaler set to 64
  } else if (frequency < 12000) {
    TCCR1B |= 0b010; // Prescaler set to 8
  } else {
    TCCR1B |= 0b001; // No prescaling
  }

  // Adjust the timer count based on desired frequency
  unsigned long timerCount = 16000000 / (2 * frequency) - 1;
  ICR1 = timerCount;
}
