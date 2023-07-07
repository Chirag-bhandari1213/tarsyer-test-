import cv2
haar_cascade = 'cars.xml'
video = 'video.avi'
      
cap = cv2.VideoCapture(video)
car_cascade = cv2.CascadeClassifier(haar_cascade)

# reads frames from a video
ret, frames = cap.read()
        
# convert frames to gray scale 
gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        
# Detects cars of different sizes in the input image
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

# To draw a rectangle in each cars
for (x,y,w,h) in cars:
  cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
    
# Display frames in a window 
cv2.imshow('video', frames)

cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
    cnt += 1
print(cnt, " cars found")
