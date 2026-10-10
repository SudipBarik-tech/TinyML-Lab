#include "q_policy.h"

const char* actionName(int action) {
  switch (action) {
    case 0: return "UP";
    case 1: return "RIGHT";
    case 2: return "DOWN";
    case 3: return "LEFT";
    default: return "STOP";
  }
}

int chooseAction(int state) {
  if (state < 0 || state >= Q_STATES) return -1;
  return Q_POLICY[state];
}

void setup() {
  Serial.begin(115200);
  int state = 0;
  Serial.print("State ");
  Serial.print(state);
  Serial.print(" -> action: ");
  Serial.println(actionName(chooseAction(state)));
}

void loop() {}
