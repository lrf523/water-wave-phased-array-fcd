/*For ESP32*/

#include <Arduino.h>
#include <EEPROM.h>
#include <Adafruit_MCP4728.h>
#include <Wire.h>

#define SYSSerial Serial
#define DebugSerial Serial2

const uint32_t SerialBaud = 115200;
const uint32_t Serial2Baud = 115200;

#define CH 8

#define TX2_PIN 17
#define RX2_PIN 16
#define SYNC_PIN 25
#define SDA_PIN 21
#define SCL_PIN 22
#define LDAC_PIN 4
#define FILTER_EN_PIN 5

const bool ldac_invert = false;

Adafruit_MCP4728 MCP1;
Adafruit_MCP4728 MCP2;

uint32_t count = 0;
uint32_t timercycle = 1000; //us

uint8_t wavemode = 0; //0: Sinewave, 1: Arbitrary

//Sinewave Parameters
uint32_t cycle[CH] = {100, 100, 100, 100, 100, 100, 100, 100};
float phase[CH] = {0.0F, 0.125F, 0.25F, 0.375F, 0.5F, 0.625F, 0.75F, 0.875F};
float amp[CH] = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};

//Arbitrarywave Parameters
uint16_t arbitrarywave_len = 16;
uint16_t arbitrarywave_phase[CH] = {0};
const uint16_t max_arbitrarywave_len = 256;
uint16_t arbitrarywave[max_arbitrarywave_len] = {0};

bool filter_en = false;
uint8_t enable = 255;
uint32_t last_trg_micros = 0;
const uint32_t minimum_trg_interval = 800;
bool flag = false;

void fresh();
void waveGene();
void WriteEEPROM();
void ReadEEPROM();
void handleSerial();

void setup() {
  // put your setup code here, to run once:
  //pinMode(MS_PIN, INPUT_PULLDOWN);

  pinMode(LDAC_PIN, OUTPUT);
  pinMode(SYNC_PIN, INPUT);
  pinMode(FILTER_EN_PIN, OUTPUT);
  filter_en = true;
  digitalWrite(FILTER_EN_PIN, HIGH);
  Serial.begin(SerialBaud);
  Serial2.begin(Serial2Baud, SERIAL_8N1, RX2_PIN, TX2_PIN);
  Wire.begin(SDA_PIN, SCL_PIN, 400000);

  DebugSerial.println("Slave Mode");
  DebugSerial.print("Chip:");
  DebugSerial.println(ESP.getChipModel());
  DebugSerial.print("Serial Baud:");
  DebugSerial.println(SerialBaud);
  DebugSerial.print("Serial2 Baud:");
  DebugSerial.println(Serial2Baud);

  if (MCP1.begin(0x60)) {
    DebugSerial.println("MCP1 Found");
  }
  else {
    DebugSerial.println("Failed to find MCP1");
  }
  if (MCP2.begin(0x61)) {
    DebugSerial.println("MCP2 Found");
  }
  else {
    DebugSerial.println("Failed to find MCP2");
  }

  bool success = true;
  success &= MCP1.setChannelValue(MCP4728_CHANNEL_A, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  success &= MCP1.setChannelValue(MCP4728_CHANNEL_B, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  success &= MCP1.setChannelValue(MCP4728_CHANNEL_C, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  success &= MCP1.setChannelValue(MCP4728_CHANNEL_D, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  success &= MCP2.setChannelValue(MCP4728_CHANNEL_A, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  success &= MCP2.setChannelValue(MCP4728_CHANNEL_B, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  success &= MCP2.setChannelValue(MCP4728_CHANNEL_C, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  success &= MCP2.setChannelValue(MCP4728_CHANNEL_D, 2048, MCP4728_VREF_VDD, MCP4728_GAIN_1X, MCP4728_PD_MODE_NORMAL, true);
  if(success) DebugSerial.println("MCP Setup Success");
  else {
    DebugSerial.println("MCP Setup Failed");
    while(true) {
      delay(1000);
    }
  }

  EEPROM.begin(128);
  if(EEPROM.read(127) != 6) {
    WriteEEPROM();
    EEPROM.write(127, 6);
    EEPROM.commit();
  } else {
    ReadEEPROM();
  }
  
  // 初始化arbitrarywave数组为全2048
  for(int i = 0; i < max_arbitrarywave_len; i++) {
    arbitrarywave[i] = 2048;
  }

  attachInterrupt(SYNC_PIN, fresh, FALLING);
}

void loop() {
  if(flag) {
    waveGene();
    flag = false;
  }
  handleSerial();
  // delay(1);
}

void fresh() {
  if(micros() - last_trg_micros < minimum_trg_interval) return;
  last_trg_micros = micros();
  digitalWrite(LDAC_PIN, LOW xor ldac_invert);
  count++;
  flag = true;
}
void waveGene() {
  uint16_t val[CH];
  if(wavemode == 0) { //Sinewave
    for(uint8_t i = 0; i < CH; i++) {
      if(enable & ((uint8_t)1 << i)) {
        float angle = float(count % cycle[i]) / float(cycle[i]) - phase[i];
        angle = angle * 2.0F * 3.1415926F;
        float a = sin(angle);
        val[i] = 2048 + 2047 * a * amp[i];
      } else {
        val[i] = 2048;
      }
    }
  } else if(wavemode == 1) { //Arbitrarywave
    for(uint8_t i = 0; i < CH; i++) {
      if(enable & ((uint8_t)1 << i)) {
        val[i] = arbitrarywave[(count + arbitrarywave_phase[i]) % arbitrarywave_len];
      } else {
        val[i] = 2048;
      }
    }
  }
  digitalWrite(LDAC_PIN, HIGH xor ldac_invert);
  MCP1.fastWrite(val[0], val[1], val[2], val[3]);
  MCP2.fastWrite(val[4], val[5], val[6], val[7]);
}
void WriteEEPROM() {
  for(uint8_t i = 0; i < CH; i++) {
    EEPROM.writeFloat(i * 12, amp[i]);
    EEPROM.writeFloat(i * 12 + 4, phase[i]);
    EEPROM.writeUInt(i * 12 + 8, cycle[i]);
  }
  EEPROM.write(97, enable);
  EEPROM.commit();
}
void ReadEEPROM() {
  for(uint8_t i = 0; i < CH; i++) {
    amp[i] = EEPROM.readFloat(i * 12);
    phase[i] = EEPROM.readFloat(i * 12 + 4);
    cycle[i] = EEPROM.readUInt(i * 12 + 8);
  }
  enable = EEPROM.read(97);
}
void handleSerial() {
  String str = "";
  bool received = false;
  if(SYSSerial.available()) {
    str = SYSSerial.readStringUntil('\n');
    str.trim();
    received = true;
    DebugSerial.print("SYSSerial Received: ");
    DebugSerial.println(str);
  } else if(DebugSerial.available()) {
    str = DebugSerial.readStringUntil('\n');
    str.trim();
    received = true;
    DebugSerial.print("DebugSerial Received: ");
    DebugSerial.println(str);
  }
  if(received) {
    if(str.equals("HELLO")) {
      SYSSerial.println("HELLO");
    } else if(str.equals("RST")) { // 重置计数器
      count = 0;
      SYSSerial.println("DONE");
      DebugSerial.println("Counter reset");
    } else if(str.startsWith("M")) { // 切换模式
      wavemode = strtoumax(str.substring(1).c_str(), NULL, 16);
      if(wavemode == 0) {
        filter_en = true;
        digitalWrite(FILTER_EN_PIN, HIGH);
        DebugSerial.println("Sinewave mode");
      } else if(wavemode == 1) {
        filter_en = false;
        digitalWrite(FILTER_EN_PIN, LOW);
        DebugSerial.println("Arbitrarywave mode");
      } else {
        wavemode = 0;
        DebugSerial.println("Unknown mode, set to Sinewave mode");
      }
      SYSSerial.println("DONE");
    } else if(str.startsWith("#")) { // 写入正弦波参数
      str = str.substring(1);
      if(str.length() == CH * 16) {
        for(uint8_t i = 0; i < CH; i++) {
          // uint8_t buff1[5];
          // str.getBytes(buff1, 5, i * 16);
          // uint16_t a = strtoumax((char *)buff1, NULL, 16);
          // amp[i] = (float)a / 65535.0F;
          // uint8_t buff2[5];
          // str.getBytes(buff2, 5, i * 16 + 4);
          // uint16_t b = strtoumax((char *)buff2, NULL, 16);
          // phase[i] = (float)b / 65535.0F;
          // uint8_t buff3[9];
          // str.getBytes(buff3, 9, i * 16 + 8);
          // cycle[i] = strtoumax((char *)buff3, NULL, 16);
          // if(cycle[i] < 10) cycle[i] = 10;
          String ampstr = str.substring(i * 16, i * 16 + 4);
          uint16_t a = strtoumax(ampstr.c_str(), NULL, 16);
          amp[i] = (float)a / 65535.0F;
          String phasestr = str.substring(i * 16 + 4, i * 16 + 8);
          uint16_t b = strtoumax(phasestr.c_str(), NULL, 16);
          phase[i] = (float)b / 65535.0F;
          String cyclestr = str.substring(i * 16 + 8, i * 16 + 16);
          cycle[i] = strtoumax(cyclestr.c_str(), NULL, 16);
          if(cycle[i] < 10) cycle[i] = 10;
        }
        WriteEEPROM();
        SYSSerial.println("#");
        DebugSerial.println("Sinewave parameters updated");
      } else {
        SYSSerial.println("?"); // 长度错误
        DebugSerial.print("Unexpected String length, got ");
        DebugSerial.println(str.length());
      }
    } else if(str.startsWith("?")) { // 读取正弦波参数
      String s = "";
      for(uint8_t i = 0; i < CH; i++) {
        String str1 = String(uint16_t(amp[i] * 65535.0F), 16);
        while (str1.length() < 4) {
          str1 = "0" + str1;
        }
        String str2 = String(uint16_t(phase[i] * 65535.0F), 16);
        while (str2.length() < 4) {
          str2 = "0" + str2;
        }
        String str3 = String(cycle[i], 16);
        while (str3.length() < 8) {
          str3 = "0" + str3;
        }
        s = s + str1 + str2 + str3;
      }
      s = "$" + s;
      SYSSerial.println(s);
      DebugSerial.println("Sinewave parameters read");
    } else if(str.startsWith("E")) { // 写入使能状态
      str = str.substring(1);
      if(str.length() == 2) {
        // uint8_t buff1[3];
        // str.getBytes(buff1, 3, 0);
        // enable = strtoumax((char *)buff1, NULL, 16);
        String enablestr = str.substring(0, 2);
        enable = strtoumax(enablestr.c_str(), NULL, 16);  
        EEPROM.write(97, enable);
        EEPROM.commit();
        SYSSerial.println("E");
        DebugSerial.println("Enable status updated");
        DebugSerial.println("Enable status: " + String(enable, 2));
      }
    } else if(str.startsWith("W")) { // 读取使能状态
      String s = String(enable, 16);
      while (s.length() < 2) {
        s = "0" + s;
      }
      s = "$" + s;
      SYSSerial.println(s);
      DebugSerial.println("Enable status read");
      DebugSerial.println("Enable status: " + String(enable, 2));
    } else if(str.startsWith("A")) { // 写入任意波形参数
      str = str.substring(1);
      if(str.startsWith("?")) { // 读取任意波形数据
        str = str.substring(1);
        if(str.equals("L")) { // 读取任意波形长度
          String s = String(arbitrarywave_len, 16);
          while (s.length() < 2) {
            s = "0" + s;
          }
          s = "$" + s;
          SYSSerial.println(s);
          DebugSerial.println("Arbitrary waveform length read");
        } else if(str.equals("P")) { // 读取任意波形相位
          String s = "";
          for(uint8_t i = 0; i < CH; i++) {
            String str1 = String(arbitrarywave_phase[i], 16);
            while (str1.length() < 2) {
              str1 = "0" + str1;
            }
            s = s + str1;
          }
          s = "$" + s;
          SYSSerial.println(s);
          DebugSerial.println("Arbitrary waveform phase read");
        } else if(str.startsWith("D")) { // 读取任意波形中某16个数据
          str = str.substring(1);
          uint16_t index = strtoumax(str.c_str(), NULL, 16);
          if(index >= max_arbitrarywave_len) index = max_arbitrarywave_len - 16;
          String s = "";
          for(uint16_t i = index; i < index + 16; i++) {
            String str1 = String(arbitrarywave[i], 16);
            while (str1.length() < 3) {
              str1 = "0" + str1;
            }
            s = s + str1;
          }
          s = "$" + s;
          SYSSerial.println(s);
          DebugSerial.println("Arbitrary waveform data read from index " + String(index));
        }
      } else { // 写入任意波形相关参数
        if(str.length() == 2) { // 写入任意波形长度
          uint16_t len = strtoumax(str.c_str(), NULL, 16);
          if(len < 16) len = 16;
          if(len > max_arbitrarywave_len) len = max_arbitrarywave_len;
          arbitrarywave_len = len; // 设置波形长度
          SYSSerial.println("A"); // 回复确认
          DebugSerial.println("Arbitrary waveform length: " + String(arbitrarywave_len));
        } else if(str.length() == CH * 2) { // 写入任意波形相位差
          for(uint8_t i = 0; i < CH; i++) {
            String phasestr = str.substring(i * 2, i * 2 + 2);
            uint16_t phase = strtoumax(phasestr.c_str(), NULL, 16);
            phase = phase % arbitrarywave_len;
            arbitrarywave_phase[i] = phase; // 设置相位差
          }
          SYSSerial.println("A"); // 回复确认
          DebugSerial.println("Arbitrary waveform phase updated");
        } else if(str.length() == 2 + CH * 2) { // 写入波形长度和相位差
          String lenstr = str.substring(0, 2);
          uint16_t len = strtoumax(lenstr.c_str(), NULL, 16);
          if(len < 16) len = 16;
          if(len > max_arbitrarywave_len) len = max_arbitrarywave_len;
          arbitrarywave_len = len; // 设置波形长度
          str = str.substring(2);
          for(uint8_t i = 0; i < CH; i++) {
            String phasestr = str.substring(i * 2, i * 2 + 2);
            uint16_t phase = strtoumax(phasestr.c_str(), NULL, 16);
            phase = phase % arbitrarywave_len;
            arbitrarywave_phase[i] = phase; // 设置相位差
          }
          SYSSerial.println("A"); // 回复确认
          DebugSerial.println("Arbitrary waveform length: " + String(arbitrarywave_len));
          DebugSerial.println("Arbitrary waveform phase updated");
        } else if(str.length() == 50) { // 从index处写入16个点波形数据
          String indexstr = str.substring(0, 2);
          uint8_t index = strtoumax(indexstr.c_str(), NULL, 16);
          if(index > max_arbitrarywave_len - 16) index = max_arbitrarywave_len - 16;
          str = str.substring(2);
          // 将48字节字符串转换为16个uint16_t数值
          for(uint8_t i = 0; i < 16; i++) {
            String hexstr = str.substring(i * 3, i * 3 + 3); // 提取每3个字节（3个十六进制字符）
            uint16_t tempValue = strtoumax(hexstr.c_str(), NULL, 16); // 将3字节十六进制字符串转换为uint16_t
            if(tempValue > 4095) tempValue = 4095;
            arbitrarywave[index + i] = tempValue; // 写入任意波形数据
          }
          SYSSerial.println("A"); // 回复确认
          DebugSerial.println("Arbitrary waveform updated with 16 points");
        } else {
          SYSSerial.println("?"); // 长度错误
          DebugSerial.print("Unexpected waveform updating string length, got ");
          DebugSerial.println(str.length());
        }
      }
    } else {
      SYSSerial.println("?");
      DebugSerial.println("Unexpected String: " + str);
    }
  }
}