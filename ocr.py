from PIL import Image, ImageOps
from pyocr import pyocr
import pyocr.builders
from io import StringIO
import os
import pyscreenshot as ImageGrab
from time import sleep
import mss
import numpy as np
import pytesseract
import cv2
import pyautogui
import datetime
from random import random 
import pickle

#pyautogui.PAUSE = 1
pyautogui.FAILESAFE = True
intial = 1
prevState = None
prevHigh = 0
theta = None
cur = 0
game_start = 0
state = None
score = 0
max_length = 10
thetas = []
scores = []

def image_collection():
		file = "images/test.png"
	# part of the screen	
		#take a screen shot of the the correct section
		#invert the image then grazy scale it so the white numbers
		#turn black
		sct = mss.mss()
		im = sct.grab({"left":160,"top":460,"width": 650 - 160, "height": 580-460})
		im = Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")
		im = ImageOps.invert(im)
		im = im.convert('LA')

		#save the image so i can open it back up with cv2
		#idk another way to do this because using the np array 
		#wasn't working
		im.save(file)

		#read the image back in and gray sacle it again, idk 
		#if it will do anything but w/e
		image = cv2.imread(file)
		data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		#time to get rid of grayscale and go straight black and white
		#also thresh_outsu picks the best threshold
		data = cv2.threshold(data, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		data = cv2.medianBlur(data, 5)

		#we have to write this image back down 
		cv2.imwrite(file, data)

		#open back the image 		
		img = Image.open(file)
		#img.show()

		#now tesseract will extract images from the string 
		#the language is set to english and the configs mean
		# psm is the page segmentation mode 10 is the setting to have
		#image be single set
		#oem is the ocr engine mode which 3 means the default mode
		#could set it to 2 to use LSTM and tesseract 
		#then the white liist is set for characters

		txt = pytesseract.image_to_string(img, lang='eng',
           config='--psm 10')

		# txt = pytesseract.image_to_string(img, lang='eng',
  #          config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

		return txt


#gets the gameboad so it can be fed into the NN
def game_board():
	#grabs the bounding box of the game and gray scales it
	global game_start
	sct = mss.mss()
	if(game_start):
		imag = sct.grab({"left":161,"top":760,"width": 650 - 161, "height": 210})
	else:
		imag = sct.grab({"left":161,"top":570,"width": 650 - 161, "height": 210})
		
	imag = Image.frombytes("RGB", imag.size, imag.bgra, "raw", "BGRX")
	imag = ImageOps.invert(imag)
	imag = imag.convert('LA')
	#imag.show()
	#converts it into the matrix form then flattens it into a vector
	data = np.array(imag).flatten()
	data = np.append([1], data)
	data = data.reshape(data.size, 1)
	data = data / 255
	data = data / data.size
	return data


def storeNet():
	global thetas
	global theta
	global scores

	if(len(thetas) < max_length):
		thetas.append(theta)
		scores.append(cur)
	else: 
		minpos = scores.index(min(scores))
		theta[minpos] = theta
		scores[minpos] = scores

def avgNet():
	global state
	avg = np.zeros((state.size, 4 ))
	global thetas
	for theta in thetas:
		avg = theta + avg

	return avg / len(thetas)

prevTheta = None
def evolve():
	global prevTheta
	global cur
	global prevHigh
	global theta
	global state
	global score
	dif = cur - prevHigh
	growth = 0

	storeNet()
	theta = avgNet()

	if(prevHigh > 0):
		growth = dif/prevHigh
		score = dif/prevHigh + 1 * cur
		theta = theta + growth * theta
	elif(game_start):
		theta = theta * 1.5 / (prevHigh + 1)
		score = 1
	else:
		#print("rand")
		grow = random()
		#print(np.shape(grow))
		#print(np.shape(theta))
		prevTheta = theta
		theta =  theta + (theta * (random() - .5)) * 5

	
	cur = 0
	#print("evolve")

def network():
	# P x 1 vector
	global state
	state = game_board()
	#normalize the scale of rgb
	X = state
	#checks if the game board from before is the same,
	#if so we need some changes
	global prevState
	global theta
	global intial
	global game_start
	#checks if this is the first run if so then randomize
	if(intial):
		theta = np.random.rand(4, state.size )
		intial = 0;

	#print(theta)
	if(not np.array_equal(prevState,state) and not (prevState is None)):
		game_start = 1
	else :
		if(not game_start):
			evolve()
		else:
			sleep(15)
			retarting = 1
	#store the state
	prevState = state

	
	#4 x P times P x 1
	y = np.matmul(theta, X)
	#print(X)
	print(y)
	y = 1 / (1 + np.exp(-y))

	#returns the output 
	return y


def scale(num, max, min):
	return -num * (max - min) + max

def play(): 


	global game_start
	global restarting
	outputs = network()
	if(restarting):
		return

	left = 160
	right = 650
	if(game_start):
		top = 760
		bottom = 950
	else:
		top = 570
		bottom = 780

	scaleX = scale(outputs[0], right, left)
	scaleY = scale(outputs[1], bottom, top)

	click = 1 if outputs[2] < outputs[3] else 0
	#print(outputs)
	#print(scaleX, scaleY, click)
	if(click):
		()
		#print(scaleX)
		pyautogui.mouseDown(scaleX, scaleY)
		pyautogui.mouseUp()
	else:
		()
		pyautogui.mouseDown(scaleX, scaleY)
		pyautogui.mouseUp()

	

#clicks the restart button
def restart_game():
	global game_start
	global prevState
	global cur 
	prevState = None
	game_start = 0
	cur = 0
	pyautogui.mouseUp()
	pyautogui.mouseDown(400, 940)
	pyautogui.mouseUp()

#Varibles needed for later 
prevNum = -1;
highscore = 0;
cur = 0
game_start = 0
restarting = 0

try:
	theta = pickle.load(open("Model","rb"))
	thetas = pickle.load(open("thetas", "rb"))
	highscore = pickle.load(open("highscore", "rb"))
	prevHigh = pickle.load(open("prevHigh", "rb"))

	intial = 0;
except (OSError, IOError) as e:
	()

while True:
	try:
		#now = datetime.datetime.now()
		#print(game_start)
		txt = image_collection()

		#print("Here is the number" + txt)
		#so we're just going to keep checking if there is a number
		#because when there isn't one we're still playing
		#but becasue the numbers cout up during the score
		#we're going to have to check if the numbers stop
		#before we keep going
		#print(game_start)
		#So only play the game if the numbers aren't there 
		if(str.isdigit(txt)):
			#set the current value 
			cur = int(txt)
			#print(txt)
			#check if the value is the same 
			if(prevNum == cur):
				#check if the cur score is greater than highscore
				if(highscore < cur):
					#set the highscore
					prevHigh = highscore
					highscore = cur
				#reset tehe previous number
				prevNum = -1
				#this is where the NN will shift weights based on score
				#restarts the game
				while(str.isdigit(image_collection())):
					restart_game()

				restarting = 0
				sleep(2)
				evolve()
			#change the value of the previous score
			prevNum = cur
			sleep(.01)

		else:
			#this will attempt to play the game with the NN
			if(not restarting):
				play()
		#now1 = datetime.datetime.now() - now
		#print(now1)
		sleep(.1)
	except KeyboardInterrupt:
		#pickle.dump(theta, open("Model","wb"))
		pickle.dump(thetas,  open("Theats","wb"))
		pickle.dump(theta, open("Model","wb"))
		highscore = pickle.dump(highscore, open("highscore", "wb"))
		pickle.dump(prevHigh ,open("prevHigh", "wb"))
		break

