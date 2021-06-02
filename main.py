import cv2 as cv
# import numpy as np

BODY_PARTS = {"Nose": 0, "Neck": 1,
              "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7,
              "RHip": 8, "RKnee": 9,"RAnkle": 10,
              "LHip": 11, "LKnee": 12, "LAnkle": 13,
              "REye": 14,"LEye": 15,
              "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv.dnn.readNetFromTensorflow("resources/graph_opt.pb")
thr = 0.20


def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    pxy=[]
    arr=[int]*19
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        points.append((int(x), int(y)) if conf > thr else None)
        if conf > thr :
           arr[i]=1
           pxy.append((int(x), int(y)))
        else:
           arr[i]=0
           pxy.append((0,0))
    # ff="both arms down"
    ff=""
    # for i in range(len(BODY_PARTS)):
        # print(i," ",arr[i])
    if pxy[4][1] < pxy[3][1] and arr[4]==1 and arr[3]==1 :
        ff="right arm up"
    if pxy[7][1] < pxy[6][1] and arr[7]==1 and arr[6]==1 :
        ff="left arm up"
    if pxy[4][1] < pxy[3][1] and arr[4] == 1 and arr[3] == 1 and pxy[7][1] < pxy[6][1] and arr[7]==1 and arr[6]==1:
        ff = "both arms up"
    if pxy[4][1] > pxy[3][1] and arr[4] == 1 and arr[3] == 1 and pxy[7][1] > pxy[6][1] and arr[7]==1 and arr[6]==1:
        ff = "both arms down"
    if arr[14] ==0 and arr[15]==0 and arr[0]==0 and arr[16]==1 and arr[17]==1:
        ff=ff+" turned back"
    if arr[14] == 1 and arr[15] == 1 and arr[0] == 1:
        ff=ff+" turned front"
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        fl=1
        # if idFrom==2 and idTo==3:
        #     if points[2][0]> points[3][0]:
        #         fl=1
        #     else:
        #         fl=0
        # elif idFrom == 3 and idTo == 4:
        #     if points[3][0] > points[4][0]:
        #         fl = 1
        #     else:
        #         fl = 0
        # elif idFrom==8 and idTo==9:
        #     if points[8][0]> points[9][0]:
        #         fl=1
        #     else:
        #         fl=0
        # elif idFrom==9 and idTo==10:
        #     if points[9][0]> points[10][0]:
        #         fl=1
        #     else:
        #         fl=0
        if points[idFrom] and points[idTo] and fl==1:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    # freq = cv.getTickFrequency() / 1000
    cv.putText(frame,ff, (60, 60), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255))
    #'%.2fms' % (t / freq)
    return frame

#imgae running code
# input = cv.imread("resources/image1.jpg")
# output = poseDetector(input)
# cv.imshow("window", output)
# cv.waitKey(0)


# cap = cv.VideoCapture('resources/dance1.mp4')
# ret, frame = cap.read()
# frame_height, frame_width, _ = frame.shape
# out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# print("Processing Video...")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         out.release()
#         break
#     output = poseDetector(frame)
#     cv.imshow("video frames", output)
#     cv.waitKey(1)
#     out.write(output)
#     out.release()
#
# print("Done processing video")


# camera wala
cap = cv.VideoCapture(0)
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape
out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
print("Processing Video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        out.release()
        break
    output = poseDetector(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    cv.imshow("video frames", output)
    cv.waitKey(1)
    out.write(output)
    out.release()
print("Done processing video")