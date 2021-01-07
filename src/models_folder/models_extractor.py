



def getModel_OR(OR_model_name,OR_model_version):
	if OR_model_name == 'Yolov3':
		return (model_load_yolo3(OR_model_version))


def compute_OR(OR_model_name, model, **kwargs):
	if OR_model_name == 'Yolov3':
		return(compute_Yolo3(model, **kwargs))


def model_load_yolo3(OR_model_version):
	import tensorflow as tf
	from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny)

	number_of_classes = 80
	
	if OR_model_version == 'Tiny':
        yolo = YoloV3Tiny(classes=number_of_classes)

		# weights path
		weights_path = 'models_weight\yolo3\yolov3-tiny.weights'
		yolo.load_weights(weights_path)
    else:
		yolo = YoloV3(classes=number_of_classes)

		# weights path
		weights_path = 'models_weight\yolo3\yolov3.weights'
		yolo.load_weights(weights_path)
		
	
	# weights path
	classes_path = 'models_data\yolo3\coco.names'
	class_names = [c.strip() for c in open(classes_path).readlines()]
    logging.info('classes loaded')
	return (yolo, class_names)

def compute_Yolo3(model, **kwargs):
	img_in = kwargs['image']
	boxes, scores, classes, nums = model.predict(img_in)
	classes = classes[0]
	names = []
	for i in range(len(classes)):
		names.append(class_names[int(classes[i])])
	names = np.array(names)
	converted_boxes = convert_boxes(img, boxes[0])
	features = encoder(img, converted_boxes)    
	detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
	return(detections)


def getModel_MOT(MOT_model_name,MOT_model_version):
if MOT_model_name == 'Deep_SORT':
	return(model_load_deepSORT())


def compute_MOT(MOT_model_name, tracker, detections, **kwargs):
	if MOT_model_name == 'Deep_SORT':	
		return(compute_deepSORT(tracker, detections, **kwargs))


def model_load_deepSORT(OR_model_version):
    #initialize deep sort
	from deep_sort import preprocessing
	from deep_sort import nn_matching
	from tools import generate_detections as gdet
	from deep_sort.tracker import Tracker

    model_filename = 'model_data\deepSort\mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

	return(encoder,tracker)

def compute_deepSORT(tracker, detections):
	# run non-maxima suppresion
	nms_max_overlap = 1.0

	boxs = np.array([d.tlwh for d in detections])
	scores = np.array([d.confidence for d in detections])
	classes = np.array([d.class_name for d in detections])
	indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
	detections = [detections[i] for i in indices]        

	# Call the tracker
	tracker.predict()
	tracker.update(detections)
	return(tracker,detections)