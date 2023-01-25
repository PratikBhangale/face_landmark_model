from turtle import color
import mediapipe
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img_base = cv2.imread('C:\\Users\\Pratik_Bhangale\\Downloads\\test_img.jpg')
img = img_base.copy

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
results = face_mesh.process(img)
landmarks = results.multi_face_landmarks[0]

xs = [], ys = [], zs = []

for landmark in landmarks.landmark:
    x = landmark.x
    y = landmark.y
    z = landmark.z

    xs.append(x)
    ys.append(y)
    zs.append(z)

fig = plt.figure()
ax = Axes3D(fig)
projection = ax.scatter(xs, ys, zs, color='green')
plt.show()

