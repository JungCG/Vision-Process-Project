/*
 * Vision Process (Capstone)
 * Mimic Robot 
 * https://github.com/JungCG/Vision-Process-Project
 */

#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebSocketServer.h>

WiFiServer server(80);
WebSocketServer webSocketServer;

// 공유기(AP) 입력
const char* ssid = "1";
const char* password = "11111111";

// 호스트 이름 설정
const char* hostname = "ESP32";

void setup() {
  Serial.begin(115200);

  //Static IP address configuration (고정 IP 설정)
  IPAddress staticIP(192, 168, 137, 50); // ESP static IP
  IPAddress gateway(192, 168, 137, 1); // IP Address of your WiFi Router(Gateway)
  IPAddress subnet(255, 255, 255, 0); // Subnet mask
  IPAddress dns(8, 8, 8, 8); // DNS
  if(!WiFi.config(staticIP, gateway, subnet, dns)){
    Serial.println("Static IP failed");
  }

  WiFi.begin(ssid, password);
  WiFi.setHostname(hostname); // ESP32 cam hostname

  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  server.begin();

  Serial.print("ESP32 IP : ");
  Serial.println(WiFi.localIP());
  Serial.print("Hostname : ");
  Serial.println(WiFi.getHostname());
}
//
void loop() {
  WiFiClient client = server.available();
  
  // 클라이언트와 통신 코드
  if(client.connected() && webSocketServer.handshake(client)){
    
    // 클라이언트에서 받은 데이터를 저장할 변수
    String data;
    int angleR;
    int angleL;
    int angleBodyrArm;
    int anlgeBodylArm;
    int head;
  
    // 클라이언트가 접속을 끊을 수 있기 때문에 확인하면서 while 루프를 돈다.
    while(client.connected()){
      
      // 수신한 데이터 저장
      data = webSocketServer.getData();
  
      if(data.length() > 0){
        Serial.println(data);
        int index1 = data.indexOf(',');
        int index2 = data.indexOf(',', index1+1);
        int index3 = data.indexOf(',', index2+1);
        int index4 = data.indexOf(',', index3+1);
        int index5 = data.indexOf(',', index4+1);

        angleR = data.substring(0, index1).toInt();
        angleBodyrArm = data.substring(index1+1, index2).toInt();
        angleL = data.substring(index2+1, index3).toInt();
        angleBodylArm = data.substring(index3+1, index4).toInt();
        head = data.substring(index4+1, index5).toInt();

        Serial.println(angleR);
        Serial.println(angleBodyrArm);
        Serial.println(angleL);
        Serial.println(angleBodylArm);
        Serial.println(head);
      }
    }
    client.stop();
  }
}
