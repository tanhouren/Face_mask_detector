print("Importing library. This might take a while...")
import numpy as np
import tensorflow as tf
import cv2
import os
from threading import Thread
import logging
print("Make sure your surrounding is bright")

# You can change the model by simply change the path
model_path= 'ssd_mobilenet_v2_fpnlite.tflite'
####################################################

type_list = ['got mask', 'no mask','wear incorrectly']
WIDTH = 640
HEIGHT = 480
frame = None
output = None
done = False
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

def model_init(path):
	interpreter = tf.lite.Interpreter(model_path=path)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	return interpreter, input_details, output_details

def imread(img,shape):
	if img is not None:
		img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_ = cv2.resize((img_*2/255)-1,(shape,shape))
		img_ = img_[np.newaxis,:,:,:].astype('float32')
		return img_

def cam_running(cam):
	global frame
	global done
	while not done:
		_, frame_ = cam.read()
		frame = frame_

def get_output(interpreter,output_details,i_detail,cam,shape):
	global output
	global done
	while not done:
		_,img = cam.read()
		output_frame = imread(img,shape)
		interpreter.set_tensor(i_detail[0]['index'], output_frame)
		interpreter.invoke()
		boxes = interpreter.get_tensor(output_details[0]['index'])
		classes = interpreter.get_tensor(output_details[1]['index'])
		scores = interpreter.get_tensor(output_details[2]['index'])
		num = interpreter.get_tensor(output_details[3]['index'])
		output = [boxes,classes,scores,num]

def draw_and_show(box,classes,scores,num,frame):
	for i in range(int(num[0])):
		# print(scores[0][int(i)-1])
		if scores[0][i] > 0.8:
			y,x,bottom,right = box[0][i]
			x,right = int(x*WIDTH),int(right*WIDTH)
			y,bottom = int(y*HEIGHT),int(bottom*HEIGHT)
			class_type=type_list[int(classes[0][i])]
			label_size = cv2.getTextSize(class_type,cv2.FONT_HERSHEY_DUPLEX,0.5,1)
			cv2.rectangle(frame, (x, y), (right, bottom), (0,255,0), thickness=2)
			cv2.rectangle(frame,(x,y-18),(x+label_size[0][0],y),(0,255,0),thickness=-1)
			cv2.putText(frame,class_type,(x,y-5),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
	return frame
	

def main():
	global frame
	global done
	cam = cv2.VideoCapture(0)
	cam.set(3,WIDTH)
	cam.set(4,HEIGHT)
	interpret, i_detail, o_detail = model_init(os.path.join(os.getcwd(),model_path))
	camera = Thread(target=cam_running,args=(cam,))
	inference = Thread(target=get_output,args=(interpret,o_detail,i_detail,cam,i_detail[0]['shape'][1]))
	logging.info(msg="Start inference")
	camera.start()
	inference.start()
	while not done:
		if output == None:
			pass
		else:
			frames = draw_and_show(*output,frame)
			cv2.imshow('DETECT',frames)
		key = cv2.waitKey(10)
		if key == 27:
			done = True
			logging.info(msg="Exiting")
			camera.join()
			inference.join()
			exit()
			break
			

	cv2.destroyAllWindows() 
	cam.release()
	

if __name__ =='__main__':
	main()