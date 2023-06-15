import csv
import datetime
import time
import tkinter as tk
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

window = tk.Tk()
window.title("Facial recognition system")
window.geometry('1600x900')
window.configure(background='#F1FAEE')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(window, text="Facial recognition system" ,bg="#E63946"  ,fg="#F1FAEE"  ,width=48  ,height=2,font=('times', 30, 'bold'))
message.place(x=200, y=40)
lbl = tk.Label(window, text="ID",width=20  ,height=2  ,fg="#1D3557"  ,bg="#F1FAEE" ,font=('times', 15, ' bold ') )
lbl.place(x=400, y=200)
txt = tk.Entry(window,width=30  ,bg="#F1FAEE" ,fg="#1D3557",font=('times', 15, ' bold '))
txt.place(x=600, y=215)
lbl2 = tk.Label(window, text="Name",width=20  ,fg="#1D3557"  ,bg="#F1FAEE"    ,height=2 ,font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)
txt2 = tk.Entry(window,width=30 ,bg="#F1FAEE"  ,fg="#1D3557",font=('times', 15, ' bold ')  )
txt2.place(x=600, y=315)
lbl3 = tk.Label(window, text="Notification: ",width=20  ,fg="#1D3557"  ,bg="#F1FAEE"  ,height=2 ,font=('times', 15, ' bold '))
lbl3.place(x=400, y=400)
message = tk.Label(window, text="" ,bg="#F1FAEE"  ,fg="#1D3557"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
message.place(x=700, y=400)
lbl3 = tk.Label(window, text="Information: ",width=20  ,fg="#E63946"  ,bg="#F1FAEE"  ,height=2 ,font=('times', 15, ' bold '))
lbl3.place(x=400, y=650)
message2 = tk.Label(window, text="" ,fg="#E63946"   ,bg="#F1FAEE",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold '))
message2.place(x=700, y=650)

def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text= res)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def TakeImages():
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        print(Id , name)
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml" # model phát hiện khuôn mặt haarcascade
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 7) #1,3 5
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

                sampleNum=sampleNum+1

                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #luu anh train vao folder

                cv2.imshow('frame',img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif sampleNum>500:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Ảnh đã được lưu với ID : " + Id +" - Tên : "+ name
        print(" ảnh đã được lưu với ID : " + Id +" - Tên : "+ name)
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)

def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id)) #creact & training model
    recognizer.save("TrainingImageLabel\Trainner.yml")
    print("-------------------------------------------------------------")
    print("Hoàn thành huấn luyện mô hình")
    print("-------------------------------------------------------------")
    res = "Train thành công" #+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids

acc=[]
totalImage = 0

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names =  ['Id','Name','Date','Time']

    path = "TrainingImage"
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    totalImage = len(imagePaths)
    print(totalImage)

    attendance = pd.DataFrame(columns = col_names)
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 7)

        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)

            res = recognizer.predict(gray)
            if(res[1] <500):
                confidence = float("{0:.2f}".format((100*(1-(res[1])/300))))
                acc.append(confidence)
                dis = str(confidence) +"% realtime"

            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]

            else:
                Id='Unknown'
                tt=str(Id)
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        print("Realtime acc: " + dis)
        cv2.imshow('im',im)
        if (cv2.waitKey(1)==ord('q')):
            break
    print("-------------------------------------------------------------")
    print('Highest: ' + str((np.max(acc))))

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)


clearButton = tk.Button(window, text="Clear", command=clear  ,fg="#1D3557"  ,bg="#F1FAEE"  ,width=10  ,height=1 ,activebackground = "white" ,font=('times', 15))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="#1D3557"  ,bg="#F1FAEE"  ,width=10  ,height=1, activebackground = "white" ,font=('times', 15))
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Take pic", command=TakeImages  ,fg="#F1FAEE"  ,bg="#457B9D"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train", command=TrainImages  ,fg="#F1FAEE"  ,bg="#457B9D"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Identify", command=TrackImages  ,fg="#F1FAEE"  ,bg="#E63946"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Exit", command=window.destroy  ,fg="#F1FAEE"  ,bg="#457B9D"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.configure(state="disabled",fg="white"  )
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)

window.mainloop()