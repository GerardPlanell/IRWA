import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from models_folder.models_extractor import getModel_OR,getModel_MOT, compute_OR, compute_MOT


#FLAGS
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')

OR_model_name = None
OR_model_version = None
MOT_model_name = None
MOT_model_version = None
print_results = True

#Initialize colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]



# Take into account 
'''physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''

def main(_argv):
    #Define parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0


    # Initialize Object recognition model
    model_OR, class_names = getModel_OR(OR_model_name,OR_model_version)

    # Initialize MOT model
    encoder, tracker = getModel_MOT(MOT_model_name,MOT_model_version)


    #Load video stream
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    # Start Loop for video stream
    fps = 0.0
    count = 0 
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        # Transform the Image
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        # Compute the object Recognition model
        detections = compute_OR(OR_model_name, model, image = img_in)

        # Compute the Object tracking model
        tracker, detections = compute_MOT(MOT_model_name, tracker, detections)

        #print results
        if print_results:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv2.imshow('output', img)
            

        # Detect the lines and Compute the Speed
        # Empty by now

        # Send results
        # Every minute send the updated info with the results
        # Empty by now




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
