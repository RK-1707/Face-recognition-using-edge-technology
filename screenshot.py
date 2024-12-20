from imutils.video import VideoStream
import numpy as np
import time
import os
import imutils
import cv2

#For taking screenshots
currentframe = 1

# load our serialized face detector from disk
print("Loading face detector")
protopath = "face_detection_model/deploy.prototxt" 
modelpath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel" 
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch("face_embedding_model/openface_nn4.small2.v1.t7")

# initialize the video stream
vs = VideoStream(src=0).start()		

print("Taking Pictures")
while currentframe<301:
    frame = vs.read()

	# resize the frame to have a width of 600 pixels 
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

	# construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage( cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
	
	# store output of detector model in detections
    detections = detector.forward()   

	# loop over the detections /   Multiple face detection
    for i in range(0, detections.shape[2]):
		# extract the confidence or probability associated with the prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections
        if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            
			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(cv2.resize(face, (96, 96)), 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()            
            
            #Make folder to store images if folder does not exist
            if not os.path.exists('data'): 
                os.makedirs('data')
            
            #Format and save images for dataset in the folder created
            crop_img = frame[startY - 50:endY + 50, startX - 50:endX + 50]    #(startX, startY)---- Top Left corner of face
            time.sleep(0.1)
            crop_img = cv2.resize(crop_img, (45 , 50))
            name = './data/frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, crop_img) 
            currentframe = currentframe + 1
	
print("Dataset complete")
vs.stop()