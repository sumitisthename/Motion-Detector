import cv2, time
import pandas 
from datetime import datetime

first_frame=None
status_list=[None,None]
time=[]
df =  pd.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status=0 #when no motion
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(1,21),0)


    if first_frame is None:
        first_frame= gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)

    #we are setting a threshold of 30.Value above 30 are set to 255
    threshold_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    threshold_frame=cv2.dilate(threshold_frame,None,iterations=3)

    (cnts,_) = cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #if area of contour is lesser than 1000 we keep checking. 
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1 #when motion is detected the flag is set to one.
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,2550,0),3)
    status_list.append(status)

    if status_list[-1]==1 and status_list[-2]==0:
        time.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        time.append(datetime.now())
    
    #cv2.imshow("Gray Frame",first_frame)
    #cv2.imshow("Gaussian Blur",delta_frame)

    #cv2.imshow("Gray Frame",gray)
    cv2.imshow("Gaussian Blur Frame",delta_frame)
    cv2.imshow("Threshold Frame",threshold_frame)
    cv2.imshow("Colour Frame",frame)
    


    key=cv2.waitKey(1)
    #print(gray)

    if key==ord('q'):
        if status==1:
            time.append(datetime.now())
        break
print(time)    
#print(status_list)
for i in range(0,len(time),2):
    df=df.append({"Start":time[i],"End":time[i+1]},ignore_index=True)


df.to_csv("Times.csv")
    
video.release()
cv2.destroyAllWindows

