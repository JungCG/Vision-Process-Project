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

# 세점 사이 각도 구하기
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
    ang = math.degrees(math.atan2(c[1]-b[1], Angle1) - math.atan2(a[1]-b[1], Angle2))
    
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

# 정답갯수
AnswerCount = 0

# 포기할경우 계산
# 중간 동작이 패스 동작과 비슷할 경우를 대비하여 PassCount를 선언
# 패스 동작이 2번을 초과하면 Pass로 간주한다.
Pass = 0
PassCount = 0
check = 0

# 이미지 파일의 번호를 저장하는 배열
image_list = [1, 2]

# 문제 푸는 시간만 계산
totaltime = 0

for k in image_list :
    # 이미지 읽어오기
    image_1 = cv2.imread("C:\\Users\\user\\Desktop\\captstone\\perfect_image_file\\yoga"+str(k)+".png")
    
    # 이미지 축소하기
    imagei = cv2.resize(image_1, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    imageHeight, imageWidth, _ = imagei.shape
    
    # network에 넣기 위해 전처리를 해준다.
    # OpenCV를 Caffe blob로 형식을 변환해준다.
    # blobFromImage(image, Normalizing, (공간 크기),
    #               (빼야 하는 평균값), swapRB = RGB 또는 BGR swap필요 여부, crop)
    # OpenCV와 Caffe model 모두 BGR 순서 사용
    inpBlob = cv2.dnn.blobFromImage(imagei, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
    
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
    print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

    # 각 part의 좌표값을 저장하기 위한 pointsi (image)
    # detect 성공시 좌표값을 저장 / 실패시 None 값 저장
    pointsi = []

    # 각도와 거리 계산을 위해 필요한 parts들이 detect 될때
    # 그 픽셀값을 저장해두고, detect 성공 여부를 저장하는 check 선언
    ImagePartsXY = [[0]*2 for _ in range(15)]
    #ImageCheckH = False
    #ImageCheckN = False
    ImageCheckRS = False
    ImageCheckRE = False
    ImageCheckRW = False
    ImageCheckLS = False
    ImageCheckLE = False
    ImageCheckLW = False
    ImageCheckRH = False
    ImageCheckRK = False
    ImageCheckRA = False
    ImageCheckLH = False
    ImageCheckLK = False
    ImageCheckLA = False
    #ImageCheckC = False

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
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H
        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.1 :    
            cv2.circle(imagei, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(imagei, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            pointsi.append((int(x), int(y)))
            #if i == 0 :
            #    ImageCheckH = True
            #    ImagePartsXY[0][0] = x
            #    ImagePartsXY[0][1] = y
            #elif i == 1 :
            #    ImageCheckN = True
            #    ImagePartsXY[1][0] = x
            #    ImagePartsXY[1][1] = y
            if i == 2 :
                ImageCheckRS = True
                ImagePartsXY[2][0] = x
                ImagePartsXY[2][1] = y
            elif i == 3 :
                ImageCheckRE = True
                ImagePartsXY[3][0] = x
                ImagePartsXY[3][1] = y
            elif i == 4 :
                ImageCheckRW = True
                ImagePartsXY[4][0] = x
                ImagePartsXY[4][1] = y
            elif i == 5 :
                ImageCheckLS = True
                ImagePartsXY[5][0] = x
                ImagePartsXY[5][1] = y
            elif i == 6 :
                ImageCheckLE = True
                ImagePartsXY[6][0] = x
                ImagePartsXY[6][1] = y
            elif i == 7 :
                ImageCheckLW = True
                ImagePartsXY[7][0] = x
                ImagePartsXY[7][1] = y
            elif i == 8 :
                ImageCheckRH = True
                ImagePartsXY[8][0] = x
                ImagePartsXY[8][1] = y
            elif i == 9 :
                ImageCheckRK = True
                ImagePartsXY[9][0] = x
                ImagePartsXY[9][1] = y
            elif i == 10 :
                ImageCheckRA = True
                ImagePartsXY[10][0] = x
                ImagePartsXY[10][1] = y
            elif i == 11 :
                ImageCheckLH = True
                ImagePartsXY[11][0] = x
                ImagePartsXY[11][1] = y
            elif i == 12 :
                ImageCheckLK = True
                ImagePartsXY[12][0] = x
                ImagePartsXY[12][1] = y
            elif i == 13 :
                ImageCheckLA = True
                ImagePartsXY[13][0] = x
                ImagePartsXY[13][1] = y
            #elif i == 14 :
            #    ImageCheckC = True
            #    ImagePartsXY[14][0] = x
            #    ImagePartsXY[14][1] = y
        else :
            pointsi.append(None)

    # 이미지 복사
    imageCopyi = imagei

    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    for pair in POSE_PAIRS:
        partA = pair[0]             # Head
        partA = BODY_PARTS[partA]   # 0
        partB = pair[1]             # Neck
        partB = BODY_PARTS[partB]   # 1
        #print(partA," 와 ", partB, " 연결\n")
        if pointsi[partA] and pointsi[partB] and partA == 0:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (0, 255, 0), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 1:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (255, 0, 0), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 2:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (0, 0, 255), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 3:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (250, 128, 114), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 4:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (255, 20, 147), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 5:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (0, 255, 255), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 6:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (255, 140, 0), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 7:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (72, 209, 24), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 8:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (189, 183, 107), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 9:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (138, 43, 226), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 10:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (106, 30, 205), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 11:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (85, 107, 47), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 12:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (186, 85, 211), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 13:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (128, 128, 128), 2)
        elif pointsi[partA] and pointsi[partB] and partA == 14:
            cv2.line(imageCopyi, pointsi[partA], pointsi[partB], (32, 128, 248), 2)

    cv2.imshow("Image-Output-Keypoints",imageCopyi)
    
    #이미지 띄운 후 동영상처리~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # "C:\\Users\\user\\Desktop\\captstone\\final5.avi"
    cap = cv2.VideoCapture(0) #동영상 캡쳐 생성
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("너비는 {} 높이는 {}".format(width,height))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageHeight)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageWidth)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("바뀐 너비는 {} 높이는 {}".format(width,height))

    # 판별
    ImageAndVideo = False
    PassCount = 0

    if cap.isOpened() :
        file_path ="C:\\Users\\user\\Desktop\\captstone\\nowQuestion_"+str(k)+"_solving.avi"
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        size = (int(width), int(height))
        out = cv2.VideoWriter(file_path, fourcc, 10, size)
        
        # 실행 시간 계산
        start = time.time()

        while True:
            ImageAndVideo = False

            ret, image = cap.read()
            if ret :
                # frame.shape = 불러온 이미지에서 height, width, color 받아옴
                imageHeight = int(height)
                imageWidth  = int(width)
                
                # network에 넣기위해 전처리
                inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
                
                # network에 넣어주기
                net.setInput(inpBlob)

                # 결과 받아오기
                output = net.forward()
                H = output.shape[2]
                W = output.shape[3]
                
                # 키포인트 검출시 이미지에 그려줌
                points = []
                # Shoulder Balance / Forward Head Posture
                
                VideoPartsXY = [[0]*2 for _ in range(15)]
                #VideoCheckH = False
                #VideoCheckN = False
                VideoCheckRS = False
                VideoCheckRE = False
                VideoCheckRW = False
                VideoCheckLS = False
                VideoCheckLE = False
                VideoCheckLW = False
                VideoCheckRH = False
                VideoCheckRK = False
                VideoCheckRA = False
                VideoCheckLH = False
                VideoCheckLK = False
                VideoCheckLA = False
                #VideoCheckC = False

                for i in range(0,15):
                    # 해당 신체부위 신뢰도 얻음.
                    probMap = output[0, i, :, :]
                
                    # global 최대값 찾기
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                    # 원래 이미지에 맞게 점 위치 변경
                    x = (imageWidth * point[0]) / W
                    y = (imageHeight * point[1]) / H

                    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
                    if prob > 0.1 :    
                        #cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
                        # BODY_PARTS 별 x, y 좌표값
                        print("i : ", i , ", x : ", x, ", y : ",y)
                        #cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                        points.append((int(x), int(y)))
                        #if i == 0 :
                        #    VideoCheckH = True
                        #    VideoPartsXY[0][0] = x
                        #    VideoPartsXY[0][1] = y
                        #elif i == 1 :
                        #    VideoCheckN = True
                        #    VideoPartsXY[1][0] = x
                        #    VideoPartsXY[1][1] = y
                        if i == 2 :
                            VideoCheckRS = True
                            VideoPartsXY[2][0] = x
                            VideoPartsXY[2][1] = y
                        elif i == 3 :
                            VideoCheckRE = True
                            VideoPartsXY[3][0] = x
                            VideoPartsXY[3][1] = y
                        elif i == 4 :
                            VideoCheckRW = True
                            VideoPartsXY[4][0] = x
                            VideoPartsXY[4][1] = y
                        elif i == 5 :
                            VideoCheckLS = True
                            VideoPartsXY[5][0] = x
                            VideoPartsXY[5][1] = y
                        elif i == 6 :
                            VideoCheckLE = True
                            VideoPartsXY[6][0] = x
                            VideoPartsXY[6][1] = y
                        elif i == 7 :
                            VideoCheckLW = True
                            VideoPartsXY[7][0] = x
                            VideoPartsXY[7][1] = y
                        elif i == 8 :
                            VideoCheckRH = True
                            VideoPartsXY[8][0] = x
                            VideoPartsXY[8][1] = y
                        elif i == 9 :
                            VideoCheckRK = True
                            VideoPartsXY[9][0] = x
                            VideoPartsXY[9][1] = y
                        elif i == 10 :
                            VideoCheckRA = True
                            VideoPartsXY[10][0] = x
                            VideoPartsXY[10][1] = y
                        elif i == 11 :
                            VideoCheckLH = True
                            VideoPartsXY[11][0] = x
                            VideoPartsXY[11][1] = y
                        elif i == 12 :
                            VideoCheckLK = True
                            VideoPartsXY[12][0] = x
                            VideoPartsXY[12][1] = y
                        elif i == 13 :
                            VideoCheckLA = True
                            VideoPartsXY[13][0] = x
                            VideoPartsXY[13][1] = y
                        #elif i == 14 :
                        #    VideoCheckC = True
                        #    VideoPartsXY[14][0] = x
                        #    VideoPartsXY[14][1] = y
                    else :
                        points.append(None)

                # Image Angle check
                ImageAngleRE = 0
                ImageAngleLE = 0
                ImageAngleRK = 0
                ImageAngleLK = 0
                if ImageCheckRS and ImageCheckRE and ImageCheckRW :
                    ImageAngleRE = getAngle(ImagePartsXY[2], ImagePartsXY[3], ImagePartsXY[4])
                #    cv2.putText(image, "ImageAngleRE : "+str(ImageAngleRE), (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if ImageCheckLS and ImageCheckLE and ImageCheckLW :
                    ImageAngleLE = getAngle(ImagePartsXY[5], ImagePartsXY[6], ImagePartsXY[7])
                #    cv2.putText(image, "ImageAngleLE : "+str(ImageAngleLE), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if ImageCheckRH and ImageCheckRK and ImageCheckRA :
                    ImageAngleRK = getAngle(ImagePartsXY[8], ImagePartsXY[9], ImagePartsXY[10])
                #    cv2.putText(image, "ImageAngleRK : "+str(ImageAngleRK), (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if ImageCheckLH and ImageCheckLK and ImageCheckLA :
                    ImageAngleLK = getAngle(ImagePartsXY[11], ImagePartsXY[12], ImagePartsXY[13])
                #    cv2.putText(image, "ImageAngleLK : "+str(ImageAngleLK), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                
                # Video Angle check
                VideoAngleRE = 0
                VideoAngleLE = 0
                VideoAngleRK = 0
                VideoAngleLK = 0
                if VideoCheckRS and VideoCheckRE and VideoCheckRW :
                    VideoAngleRE = getAngle(VideoPartsXY[2], VideoPartsXY[3], VideoPartsXY[4])
                #    cv2.putText(image, "VideoAngleRE : "+str(VideoAngleRE), (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if VideoCheckLS and VideoCheckLE and VideoCheckLW :
                    VideoAngleLE = getAngle(VideoPartsXY[5], VideoPartsXY[6], VideoPartsXY[7])
                #    cv2.putText(image, "VideoAngleLE : "+str(VideoAngleLE), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if VideoCheckRH and VideoCheckRK and VideoCheckRA :
                    VideoAngleRK = getAngle(VideoPartsXY[8], VideoPartsXY[9], VideoPartsXY[10])
                #    cv2.putText(image, "VideoAngleRK : "+str(VideoAngleRK), (10,105), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if VideoCheckLH and VideoCheckLK and VideoCheckLA :
                    VideoAngleLK = getAngle(VideoPartsXY[11], VideoPartsXY[12], VideoPartsXY[13])
                #    cv2.putText(image, "VideoAngleLK : "+str(VideoAngleLK), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                


                # Error Rate
                ErrorRateRE = 100
                ErrorRateLE = 100
                ErrorRateRK = 100
                ErrorRateLK = 100
                if ImageCheckRS and ImageCheckRE and ImageCheckRW and VideoCheckRS and VideoCheckRE and VideoCheckRW :
                    ErrorRateRE = round((abs(ImageAngleRE-VideoAngleRE))/ImageAngleRE*100, 1)
                    #cv2.putText(image, "ErrorRateRE : "+str(ErrorRateRE), (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if ImageCheckLS and ImageCheckLE and ImageCheckLW and VideoCheckLS and VideoCheckLE and VideoCheckLW :
                    ErrorRateLE = round((abs(ImageAngleLE-VideoAngleLE))/ImageAngleLE*100, 1)
                    #cv2.putText(image, "ErrorRateLE : "+str(ErrorRateLE), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if ImageCheckRH and ImageCheckRK and ImageCheckRA and VideoCheckRH and VideoCheckRK and VideoCheckRA :
                    ErrorRateRK = round((abs(ImageAngleRK-VideoAngleRK))/ImageAngleRK*100, 1)
                    #cv2.putText(image, "ErrorRateRK : "+str(ErrorRateRK), (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                if ImageCheckLH and ImageCheckLK and ImageCheckLA and VideoCheckLH and VideoCheckLK and VideoCheckLA :
                    ErrorRateLK = round((abs(ImageAngleLK-VideoAngleLK))/ImageAngleLK*100, 1)
                    #cv2.putText(image, "ErrorRateLK : "+str(ErrorRateLK), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                
                # Correct?
                if ErrorRateRE < 20 and ErrorRateLE < 20 and ErrorRateRK < 20 and ErrorRateLK < 20:
                    ImageAndVideo = True
                    print("Correct!!!!!")
                    AnswerCount = AnswerCount+1
                cv2.putText(image, "AnswerCount : "+ str(AnswerCount), (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                cv2.putText(image, "Pass (PassCount>=3) : " + str(Pass), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)
                cv2.putText(image, "PassCount : "+str(PassCount), (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,lineType=cv2.LINE_AA)

                # Pass Image
                if VideoAngleLE < 90 and VideoAngleRE  < 90 and VideoCheckLS and VideoCheckLE and VideoCheckLW and VideoCheckRS and VideoCheckRE and VideoCheckRW:
                    check = 1
                if VideoAngleLE >= 130 and VideoAngleRE >= 130 and VideoCheckLS and VideoCheckLE and VideoCheckLW and VideoCheckRS and VideoCheckRE and VideoCheckRW:
                    if check == 1 :
                        PassCount =  PassCount +1
                        check = 0

                # 이미지 복사
                imageCopy = image
                
                # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
                for pair in POSE_PAIRS:
                    partA = pair[0]             # Head
                    partA = BODY_PARTS[partA]   # 0
                    partB = pair[1]             # Neck
                    partB = BODY_PARTS[partB]   # 1
                    
                    #print(partA," 와 ", partB, " 연결\n")
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
                
                cv2.imshow("Output-Keypoints",imageCopy)
                out.write(imageCopy)

                if ImageAndVideo :
                    totaltime = totaltime + int(time.time()-start)
                    break
                else :
                    if PassCount >= 3 :
                        totaltime = totaltime + int(time.time()-start)
                        Pass = Pass + 1
                        break
                    if cv2.waitKey(1) != -1 :
                        totaltime = totaltime + int(time.time()-start)
                        Pass = Pass + 1
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

print("총 이미지 개수 : " + str(len(image_list)))
print("최종 정답 개수 : "+str(AnswerCount))
print("패스 한 개수 : "+str(Pass))
print("총 걸린 시간(내 동작) : "+str(totaltime))