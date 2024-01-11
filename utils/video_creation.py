##################################################################################
###################### CREATE VIDEO ##############################################
##################################################################################
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

def create_video(count,\
                 repetitive,\
                 max_imprints,\
                 imprint_every,\
                 vort_x,\
                 vort_charge,\
                 n1,n3
                 ):

  FPS=10
  if repetitive:
    SimulationName=f'{len(vort_x)}vortex__batch{vort_charge[0]}__total_imprints{max_imprints}__every{imprint_every}_fps{FPS}_frame{count}'
  else:
    SimulationName=f'{len(vort_x)}_{d}_{D}'

  VideoDims=(n1,n3)
  frames=count
  video=VideoWriter(f'{SimulationName}.mp4', 0x7634706d, float(FPS), VideoDims)

  for framenum in range(frames):
    file_path = f'R-{framenum:003}-cd.dat'
    file = open(file_path,'r')
    img=np.reshape(np.loadtxt(file, delimiter=',', usecols=2),VideoDims)
    file.close()

    zeroed=img-np.min(img)*np.ones(img.shape)
    intImg= np.uint8((255/np.max(zeroed))*zeroed)

    intImgR = np.where(intImg<230 , intImg, 255 )
    intImgG = np.where((intImg<=127) ,intImg, 255-intImg)
    intImgB = np.where((intImg>40) , 255-intImg, 0)
    img3=np.dstack((intImgB,intImgG,intImgR))

    video.write(img3)
  video.release()


def create_velocity_video(count,\
                 simulation_name,\
                 n1,n3
                 ):

  FPS=10
  SimulationName = simulation_name + f'_fps{FPS}_frame{count}'

  VideoDims=(n1,n3)
  frames=count
  video=VideoWriter(f'{SimulationName}_velocity.mp4', 0x7634706d, float(FPS), VideoDims)

  for framenum in range(frames):
    file_path = f'V-{framenum:003}-cd.dat'
    file = open(file_path,'r')
    img=np.reshape(np.loadtxt(file, delimiter=',', usecols=2),VideoDims)
    file.close()

    zeroed=img-np.min(img)*np.ones(img.shape)
    intImg= np.uint8((255/np.max(zeroed))*zeroed)

    intImgR = np.where(intImg<230 , intImg, 255 )
    intImgG = np.where((intImg<=127) ,intImg, 255-intImg)
    intImgB = np.where((intImg>40) , 255-intImg, 0)
    img3=np.dstack((intImgB,intImgG,intImgR))

    video.write(img3)
  video.release()