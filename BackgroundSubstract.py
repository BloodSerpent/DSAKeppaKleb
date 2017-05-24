import numpy as np
import cv2

class BackgroundSubstract:

	def __init__(self, height, width, alpha):
		self.height = height
		self.width = width
		self.alpha = alpha
		self.model = np.zeros((self.height, self.width, 3))
	def updateModel(self, frame):
		
		self.model = (1 - self.alpha)*self.model + self.alpha*frame
		return self.model


cap = cv2.VideoCapture('Vid1.mp4')
newObj = BackgroundSubstract(int(cap.get(4)),int(cap.get(3)), 0.01)


while (cap.isOpened):
	ret ,frame = cap.read()
	model = newObj.updateModel(frame)
	image = np.abs(frame-model).astype(np.uint8)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	image[image >50] = 255
	image[image <=50] = 0
	image = cv2.erode(image,(3,3),iterations = 2)
	# image = cv2.dilate(image,None,iterations = 1)
	param = cv2.SimpleBlobDetector_Params()
	param.filterByInertia = False
	param.filterByConvexity = False
	param.filterByColor = True
	param.blobColor = 255
	detector = cv2.SimpleBlobDetector_create(param)
	keypoint = detector.detect(image)
	imagedraw = cv2.drawKeypoints(image,keypoint,np.array([]),(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)






	if ret:
		cv2.imshow('anjas',imagedraw)
		cv2.imshow('njz',frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

