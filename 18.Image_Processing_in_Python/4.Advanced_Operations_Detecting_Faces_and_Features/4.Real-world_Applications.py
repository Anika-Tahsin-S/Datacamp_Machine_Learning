# Privacy Protection
from skimage.feature import Cascade
from skimage.filters import gaussian


def getFaceRectangle(d):
    ''' Extracts the face from the image using the coordinates of the detected image '''
    # X and Y starting points of the face rectangle
    x, y  = d['r'], d['c']
    
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'],  d['c'] + d['height']
    
    # Extract the detected face
    face = image[x:width, y:height]
    return face

def mergeBlurryFace(original, gaussian_image):
     # X and Y starting points of the face rectangle
    x, y  = d['r'], d['c']
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'],  d['c'] + d['height']
    
    original[x:width, y:height] = gaussian_image
    return original

detected = detector.detect_multi_scale(img = image,
                                       scale_factor = 1.2,
                                       step_ratio = 1,
                                       min_size = (10, 10),
                                       max_size = (200, 200))

for d in detected:
    # Obtain the face rectangle from detected coordinates
    face = getFace(d)

    gaussian_face = gaussian(face, multichannel = True, sigma = 10)

    resulting_image = mergeBlurryFace(image, gaussian_face)




# --------------------------------------------------------------------------------------------------------- #
from skimage import data
from skimage.feature import Cascade
from skimage.io import imread

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')

group_image = imread('Group_Image.jpg')

##                  Privacy protection                  ##
# Detect the faces
detected = detector.detect_multi_scale(img = group_image, 
                                       scale_factor = 1.2, step_ratio = 1, 
                                       min_size = (10, 10), max_size = (100, 100))
# For each detected face
for d in detected:  
    # Obtain the face rectangle from detected coordinates
    face = getFaceRectangle(d)
    
    # Apply gaussian filter to extracted face
    blurred_face = gaussian(face, multichannel = True, sigma = 8)
    
    # Merge this blurry face to our final image and show it
    resulting_image = mergeBlurryFace(group_image, blurred_face) 
show_image(resulting_image, "Blurred faces")





##                  Help Sally restore her graduation photo                  ##
damaged_image = imread('Sally_Damaged_Image.jpg')

def get_mask(image):
    # Create mask with three defect regions: left, middle, right respectively
    mask_for_solution = np.zeros(image.shape[:-1])
    mask_for_solution[450:475, 470:495] = 1
    mask_for_solution[320:355, 140:175] = 1
    mask_for_solution[130:155, 345:370] = 1
    return mask_for_solution


# Import the necessary modules
from skimage.restoration import denoise_tv_chambolle, inpaint
from skimage import transform

# Transform the image so it's not rotated
upright_img = transform.rotate(damaged_image, 20)

# Remove noise from the image, using the chambolle method
upright_img_without_noise = denoise_tv_chambolle(upright_img,weight = 0.1, multichannel = True)

# Reconstruct the image missing parts
mask = get_mask(upright_img)
result = inpaint.inpaint_biharmonic(upright_img_without_noise, mask, multichannel = True)

show_image(result)
