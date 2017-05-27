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




