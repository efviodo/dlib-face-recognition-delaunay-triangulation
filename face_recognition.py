import cv2
import dlib


def detect_landmarks(img):
    # set up the 68 point facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces_in_image = detector(img_gray, 0)

    # Process only first image
    face = faces_in_image[0]

    # assign the facial landmarks
    landmarks = predictor(img_gray, face)

    # unpack the 68 landmark coordinates from the dlib object into a list
    landmarks_list = []
    for i in range(0, landmarks.num_parts):
        landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

    return face, landmarks_list


if __name__ == '__main__':
    # set up the 68 point facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # bring in the input image
    img = cv2.imread('obama.jpg', 1)

    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces_in_image = detector(img_gray, 0)

    # loop through each face in image
    for face in faces_in_image:

        # assign the facial landmarks
        landmarks = predictor(img_gray, face)

        # unpack the 68 landmark coordinates from the dlib object into a list
        landmarks_list = []
        for i in range(0, landmarks.num_parts):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

        # for each landmark, plot and write number
        for landmark_num, xy in enumerate(landmarks_list, start=1):
            cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
            cv2.putText(img, str(landmark_num), (xy[0] - 7, xy[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                        1)

    # visualise the image with landmarks
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
