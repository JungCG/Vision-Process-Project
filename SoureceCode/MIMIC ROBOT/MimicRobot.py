# MPII를 사용한 신체부위 검출 소스 코드
# 정적인 digital data가 아닌 동적인 image, video data 처리
# OpenCV를 이용하기 위해 cv2 import
import cv2

# 병렬처리를 위한 numpy import
import numpy as np

# 관절들의 각도 계산(atan2)을 위해 math import
import math

# 프로그램 실행시간등을 계산하기 위해 time import
import time

import websocket
ws = websocket.WebSocket()
ESP32 = "192.168.137.50"
ws.connect ("ws://" + ESP32 + "/")

# 세 점 사이 각도 구하기
# 이 함수는 두 직선의 교점을 이용해서 각도를 계산한다.
# 교점을 이용하기 어려운 경우에는 def 내에 ang = math.degrees(~)를 사용하여 좌표값을 대입해주면 된다.
def getAngle(a, b, c):
    # tangent 계산에서 분모에 0이 들어갈 경우 통제
    Angle1 = c[0]-b[0]
    Angle2 = a[0]-b[0]
    if Angle1 == 0 :
        Angle1 = 1
    if Angle2 == 0 :
        Angle2 = 1
    
    # ArcTangent를 이용한 각도는 공식 -> 벡터로 확장가능
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))

    # 내각이 아닌 외각을 구했을 경우 통제
    if ang < 0 :
        ang = ang + 360 
        if ang > 180 :
            ang = 180 - (ang%180)
            return int(ang)
        else :
            return int(ang)
    else :
        if ang > 180 :
            ang = 180 - (ang%180)
            return int(ang)
        else :
            return int(ang)
    
# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }
POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
# protoFile : NN 구조 파일 / weightsFile : trained model의 weight 저장
# 각 파일 path
protoFile = "C:\\Users\\user\\Desktop\\captstone\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "C:\\Users\\user\\Desktop\\captstone\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_iter_160000.caffemodel"
 
# 위의 path에 있는 network를 memory로 load한다.
# Caffe : trained on Caffe Deep Learning FrameWork (Caffe Model)
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# 영상 가지고 오기 (동영상 directory / 0 = Notebook Camera)
# 비디오 캡쳐 객체를 생성한다.
cap = cv2.VideoCapture(cv2.CAP_DSHOW +0) #동영상 캡쳐 생성 + image일 경우 cv2.imread()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("너비는 {} 높이는 {}".format(width,height))

# 영상 크기 조정 => 빠른 연산
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("바뀐 너비는 {} 높이는 {}".format(width,height))

# 비디오 캡쳐 객체가 정상 open되었는지 확인한다.
# if : 정상 open
# else : open 실패
if cap.isOpened() :

    # output을 저장할 file_path
    file_path ="C:\\Users\\user\\Desktop\\captstone\\whawawaw.avi"
    fps = cap.get(cv2.CAP_PROP_FPS) #프레임수 구하기
    
    # 코덱 정보를 저장한다.
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    size = (int(width), int(height))
    
    # VideoWriter('저장할 파일 이름', 코덱 정보, 초당 저장될 프레임, (저장될 사이즈))
    # ex) VideoWriter('output.avi', fourcc, 25.0, (640, 480))
    out = cv2.VideoWriter(file_path, fourcc, 10, size)
    
    # delay = int(1000/fps)
    
    # 실행 시간을 계산하기 위해 현재 시간을 저장.
    start = time.time()

    # 특정 키를 누를 때까지 무한 반복하기 위해 while True로 작성.
    while True:

        # ret : 제대로 읽었을 경우 True, 실패할 경우 False
        # image : 비디오의 한 프레임을 읽어서 저장한다.
        ret, image = cap.read()

        # if : 프레임을 읽어온 경우
        # else : 프레임을 읽어오지 못한 경우
        if ret :
            # frame.shape = 불러온 이미지에서 height, width, color 받아옴
            imageHeight = int(height)
            imageWidth  = int(width)
            
            # network에 넣기 위해 전처리를 해준다.
            # OpenCV를 Caffe blob로 형식을 변환해준다.
            # blobFromImage(image, Normalizing, (공간 크기),
            #               (빼야 하는 평균값), swapRB = RGB 또는 BGR swap필요 여부, crop)
            # OpenCV와 Caffe model 모두 BGR 순서 사용
            inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
            
            # network에 INPUT을 대기시킨다.
            net.setInput(inpBlob)

            # network에 forward하고 결과를 받아온다.
            output = net.forward()

            # output.shape[0] = image ID,
            #             [1] = index of Keypoint(MPII는 44, Keypoint Confidence Map, Background, Part Affinity Map),
            #             [2] = Height
            #             [3] = Width
            H = output.shape[2]
            W = output.shape[3]
            print("이미지 ID : ", output.shape[0], ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

            # 각 part의 좌표값을 저장하기 위한 points
            # detect 성공시 좌표값을 저장 / 실패시 None 값 저장
            points = []
            
            # Head, Neck, RShoulder, RElbow, RWrist,
            # LShoulder, LElbow, LWrist, Chest(Body)
            # 각도와 거리 계산을 위해 필요한 parts들이 detect 될때
            # 그 픽셀값을 저장해두고, detect 성공 여부를 저장하는 check 선언
            PartsXY = [[0]*2 for _ in range(9)]
            checkH = False
            checkN = False
            checkRs = False
            checkLs = False
            checkRe = False
            checkLe = False
            checkRw = False
            checkLw = False
            checkBody = False
            # 차렷자세 구현을 위해 나머지 골격 부분도 모두 선언
            checkRH = False
            checkRK = False
            checkRA = False
            checkLH = False
            checkLK = False
            checkLA = False

            # parts 번호 0번부터 15번까지 반복문을 실행하여
            # detect가 성공했을 때 좌표값을 points에 저장
            # 실패했을 경우에는 None을 points에 저장
            for i in range(0,15):
                # 해당 신체부위 신뢰도 얻음.
                probMap = output[0, i, :, :]
            
                # 템플릿 매칭(Template Matching)을 이용하여
                # 이미지 내에서 i번 part와 비교, 최대값 찾기
                # cv2.minMaxLoc 을 사용하면
                # 비교 값의 최소, 최대 값과 해당 픽셀 좌표를 계산
                minVal, maxval, minLoc, maxLoc = cv2.minMaxLoc(probMap)

                # 원래 이미지에 맞게 점 위치 변경
                x = (imageWidth * maxLoc[0]) / W
                y = (imageHeight * maxLoc[1]) / H

                # if : 키포인트 검출한 결과가 0.1보다 크면
                #      (검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가,
                # else : 검출했는데 부위가 없으면 None으로
                if maxval > 0.1 : # dect 성공했을 시

                    # circle(image, center(원의 중심), radian, color, thickness)
                    cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    # BODY_PARTS 별 x, y 좌표값
                    print("i : ", i , ", x : ", x, ", y : ", y)

                    # putText(img, text, img 위치, font, font Scale, color)
                    cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))
                    if i == 0 : # Head
                        checkH = True # Head detect 성공시 
                        PartsXY[0][0] = x
                        PartsXY[0][1] = y
                    elif i == 1 : # Neck
                        checkN = True
                        PartsXY[1][0] = x
                        PartsXY[1][1] = y
                    elif i == 2 : # RShoulder
                        checkRs = True
                        PartsXY[2][0] = x
                        PartsXY[2][1] = y
                    elif i == 3 : # RElbow
                        checkRe = True
                        PartsXY[3][0] = x
                        PartsXY[3][1] = y
                    elif i == 4 : # RWrist
                        checkRw = True
                        PartsXY[4][0] = x
                        PartsXY[4][1] = y
                    elif i == 5 : # LShoulder
                        checkLs = True
                        PartsXY[5][0] = x
                        PartsXY[5][1] = y
                    elif i == 6 : # LElbow
                        checkLe = True
                        PartsXY[6][0] = x
                        PartsXY[6][1] = y
                    elif i == 7 : # LWrist
                        checkLw = True
                        PartsXY[7][0] = x
                        PartsXY[7][1] = y
                    elif i == 8 : # RHip
                        checkRH = True
                    elif i == 9 : # RKnee
                        checkRK = True
                    elif i == 10 : # RAnkle
                        checkRA = True
                    elif i == 11 : # LHip
                        checkLH = True
                    elif i == 12 : # LKnee
                        checkLK = True
                    elif i == 13 : # LAnkle
                        checkLA = True
                    elif i == 14 : # Chest(Body)
                        checkBody = True
                        PartsXY[8][0] = x
                        PartsXY[8][1] = y
                else : 
                    points.append(None)
            
            # 각도 초기화 (= 차렷자세)
            angleL = 180
            angleR = 180
            angleBodylArm = 0
            angleBodyrArm = 0
            angleH = 50
            
            # 사람이 탐지되지 않을 경우에는 "Human Not Detected" 메세지를 윈도우에 출력
            # 로봇의 동작은 차렷
            # 골격체크가 모두 False인 경우에만 CheckHuman을 False로 변환
            checkHuman = True
            if checkH == False and checkN == False and checkRs == False :
                if checkLs == False and checkRe == False and checkLe == False :
                    if checkRw == False and checkLw == False and checkBody == False :
                        if checkRH == False and checkRK == False and checkRA == False :
                            if checkLH == False and checkLK == False and checkLA == False :
                                checkHuman = False
                                ws.send("{},{},{},{},{}".format(angleR, angleL, angleBodyrArm, angleBodylArm, angleH))
                                cv2.putText(image, "Human Not Detected", ((int)(imageWidth/2)-(int)(imageWidth/4), (int)(imageHeight/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            
            if checkHuman :
                # 양쪽 팔꿈치의 내각을 계산 (getAngle 함수 사용)
                # Elbow를 교점으로 하여 Shoulder와 Wrist 세 parts 사이의 각도를 계산한다.
                if checkLs and checkLe and checkLw :
                    angleL = getAngle(PartsXY[5], PartsXY[6], PartsXY[7])
                    cv2.putText(image, str(angleL), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, lineType=cv2.LINE_AA)
                if checkRs and checkRe and checkRw :
                    angleR = getAngle(PartsXY[2], PartsXY[3], PartsXY[4])
                    cv2.putText(image, str(angleR), (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, lineType=cv2.LINE_AA)
                # 몸과 양쪽 팔의 각도를 계산
                # Shoulder와 Elbow의 좌표값을 Chest의 수직선과 교점이 생기게끔 이동하여
                # temp에 새로운 좌표값을 저장한다.
                # 그 다음 getAngle을 이용하여 각도를 구한다.
                if checkN and checkBody and checkRs and checkRe :
                    temp = [[0] * 2 for _ in range(2)]
                    temp[0][0] = round((PartsXY[1][0] + PartsXY[8][0])/2, 0)
                    temp[0][1] = PartsXY[2][1]
                    temp[1][0] = PartsXY[3][0] + (round((PartsXY[1][0] + PartsXY[8][0]) / 2, 0)-PartsXY[2][0])
                    temp[1][1] = PartsXY[3][1]
                    angleBodyrArm = getAngle(temp[1], temp[0], PartsXY[8])
                    cv2.putText(image, str(angleBodyrArm), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, lineType=cv2.LINE_AA)
                if checkN and checkBody and checkLs and checkLe :
                    temp = [[0] * 2 for _ in range(2)]
                    temp[0][0] = round((PartsXY[1][0] + PartsXY[8][0])/2, 0)
                    temp[0][1] = PartsXY[5][1]
                    temp[1][0] = PartsXY[6][0] - (PartsXY[5][0] - round((PartsXY[1][0] + PartsXY[8][0]) / 2, 0))
                    temp[1][1] = PartsXY[6][1]
                    angleBodylArm = getAngle(temp[1], temp[0], PartsXY[8])
                    cv2.putText(image, str(angleBodylArm), (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, lineType=cv2.LINE_AA)
                # 고개 각도를 계산
                # 임의의 temp 좌표 생성 
                # x좌표 : 왼쪽 어깨와 목 중간 점
                # y좌표 : 머리와 목의 중간 점
                # 목을 교점으로하여 임의의 temp점과 머리, 세 점 각도를 계산
                if checkH and checkN and checkLs:
                    temp = [[0] * 2 for _ in range(1)]
                    temp[0][0] = int(round(PartsXY[5][0]+PartsXY[1][0] /2 ,2))
                    temp[0][1] = int(round(PartsXY[0][1]+PartsXY[1][1] /2 ,2))
                    angleH = getAngle(temp[0], PartsXY[1], PartsXY[0])-50
                    cv2.putText(image, str(angleH), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, lineType=cv2.LINE_AA)

                # 계산결과 변경된 각도를 전송한다.
                ws.send("{},{},{},{},{}".format(angleR, angleL, angleBodyrArm, angleBodylArm, angleH))
            # parts의 계산이 끝난 image를 복사하여 새로운 image 생성
            imageCopy = image


            # 각 POSE_PAIRS를 선으로 연결 시키는 반복문.
            # 선의 시작부분 partA와 선의 끝부분 partB의 좌표값이
            # 모두 points에 None이 아닌 상태로 존재할 경우
            # 두 parts를 이어준다.
            for pair in POSE_PAIRS:
                partA = pair[0]             # Head
                partA = BODY_PARTS[partA]   # 0
                partB = pair[1]             # Neck
                partB = BODY_PARTS[partB]   # 1
                
                # print(partA," 와 ", partB, " 연결\n")
                
                # line(img, start, end, color, thickness)
                if points[partA] and points[partB] and partA == 0:
                    cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)
                elif points[partA] and points[partB] and partA == 1:
                    cv2.line(imageCopy, points[partA], points[partB], (255, 0, 0), 2)
                elif points[partA] and points[partB] and partA == 2:
                    cv2.line(imageCopy, points[partA], points[partB], (0, 0, 255), 2)
                elif points[partA] and points[partB] and partA == 3:
                    cv2.line(imageCopy, points[partA], points[partB], (250, 128, 114), 2)
                elif points[partA] and points[partB] and partA == 4:
                    cv2.line(imageCopy, points[partA], points[partB], (255, 20, 147), 2)
                elif points[partA] and points[partB] and partA == 5:
                    cv2.line(imageCopy, points[partA], points[partB], (0, 255, 255), 2)
                elif points[partA] and points[partB] and partA == 6:
                    cv2.line(imageCopy, points[partA], points[partB], (255, 140, 0), 2)
                elif points[partA] and points[partB] and partA == 7:
                    cv2.line(imageCopy, points[partA], points[partB], (72, 209, 24), 2)
                elif points[partA] and points[partB] and partA == 8:
                    cv2.line(imageCopy, points[partA], points[partB], (189, 183, 107), 2)
                elif points[partA] and points[partB] and partA == 9:
                    cv2.line(imageCopy, points[partA], points[partB], (138, 43, 226), 2)
                elif points[partA] and points[partB] and partA == 10:
                    cv2.line(imageCopy, points[partA], points[partB], (106, 30, 205), 2)
                elif points[partA] and points[partB] and partA == 11:
                    cv2.line(imageCopy, points[partA], points[partB], (85, 107, 47), 2)
                elif points[partA] and points[partB] and partA == 12:
                    cv2.line(imageCopy, points[partA], points[partB], (186, 85, 211), 2)
                elif points[partA] and points[partB] and partA == 13:
                    cv2.line(imageCopy, points[partA], points[partB], (128, 128, 128), 2)
                elif points[partA] and points[partB] and partA == 14:
                    cv2.line(imageCopy, points[partA], points[partB], (32, 128, 248), 2)

            # imshow("title(윈도우 창의 제목)", 원하는 이미지)
            cv2.imshow("Output-Keypoints",imageCopy)
            out.write(imageCopy)

            # 키보드 입력을 대기하는 함수
            # waitKey(choice)
            # choice == 0 : key 입력까지 무한 대기
            #        == ? : ?ms 대기
            #        == 27 : ESC
            # 2ms 대기하면서 어떤 입력이든 들어올 경우 break하고 아닐경우 진행한다.
            if cv2.waitKey(2) != -1 :
                break

        else :
            print("no frame")
            break
    out.release()
else:
    print("열지 못했다")

# 비디에 캡쳐 객체 해제
cap.release()

# 화면에 나타난 윈도우를 종료한다.
cv2.destroyAllWindows()