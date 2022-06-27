import cv2


def write_images_to_video(images, output_path):
	image_width = images[0].shape[1]
	image_height = images[0].shape[0]
	frameSize = (image_width, image_height)
	out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize)
	for image in images:
		out.write(image.astype(np.uint8))
	out.release()