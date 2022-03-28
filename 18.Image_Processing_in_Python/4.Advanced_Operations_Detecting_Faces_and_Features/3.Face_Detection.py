import matplotlib.patches as patches

def crop_face(result, detected, title="Face detected"):
    for d in detected:
        print(d)
        rostro= result[d['r']:d['r']+d['width'], d['c']:d['c']+d['height']]
    
        plt.figure(figsize=(8, 6))
        plt.imshow(rostro)    
        plt.title(title)
        plt.axis('off')
        plt.show()

def show_detected_face(result, detected, title="Face image"):
    plt.figure()
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')

    for patch in detected:
        img_desc.add_patch(
            patches.Rectangle((patch['c'], patch['r']), 
                            patch['width'], patch['height'], 
                            fill = False, color = 'r', linewidth = 2))
    plt.show()
    crop_face(result, detected)


from skimage.feature import Cascade

trained_file = data.lbp_frontal_face_cascade_filename()

detector = Cascade(trained_file)

# Apply detector on the image
detected = detector.detect_multi_scale(img = image, scale_factor = 1.2, step_ratio = 1, min_size(10, 10), max_size = (200, 200))

print(detected)
show_detected_face(image, detected)









# --------------------------------------------------------------------------------------------------------- #
from skimage import data
from skimage.feature import Cascade
from skimage.io import imread

night_image = imread('Face_Det3.jpg')

##                  Is someone there?                  ##
# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with min and max size of searching window
detected = detector.detect_multi_scale(img = night_image,
                                       scale_factor = 1.2,
                                       step_ratio = 1,
                                       min_size = (10, 10),
                                       max_size = (200, 200))

# Show the detected faces
show_detected_face(night_image, detected)
# output : 
    {'r': 774, 'c': 131, 'width': 40, 'height': 40}






##                  Multiple faces                  ##
friends_image = imread('Face_Det_Friends22.jpg')

# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with scale factor to 1.2 and step ratio to 1
detected = detector.detect_multi_scale(img=friends_image,
                                       scale_factor = 1.2,
                                       step_ratio = 1,
                                       min_size = (10, 10),
                                       max_size = (200, 200))
# Show the detected faces
show_detected_face(friends_image, detected)

# output:
    {'r': 199, 'c': 405, 'width': 52, 'height': 52}
    {'r': 207, 'c': 152, 'width': 47, 'height': 47}
    {'r': 217, 'c': 311, 'width': 39, 'height': 39}
    {'r': 202, 'c': 31, 'width': 36, 'height': 36}
    {'r': 219, 'c': 533, 'width': 48, 'height': 48}
    {'r': 224, 'c': 443, 'width': 45, 'height': 45}
    {'r': 242, 'c': 237, 'width': 41, 'height': 41}







##                  Segmentation and face detection                  ##
from skimage.segmentation import slic
from skimage.color import label2rgb

profile_image = imread('Face_Det9.jpg')

# Obtain the segmentation with default 100 regions
segments = slic(profile_image, n_segments = 100)

# Obtain segmented image using label2rgb
segmented_image = label2rgb(segments, profile_image, kind = 'avg')

# Detect the faces with multi scale method
detected = detector.detect_multi_scale(img = segmented_image, 
                                       scale_factor = 1.2, 
                                       step_ratio = 1, 
                                       min_size = (10, 10), max_size = (1000, 1000))

# Show the detected faces
show_detected_face(segmented_image, detected)

# output:
    {'r': 110, 'c': 169, 'width': 340, 'height': 340}
