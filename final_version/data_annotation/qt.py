import cv2
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from utils import (
    initialize_render, merge_meshes,
    load_motion
)
import torch
from PIL import Image
from natsort import natsorted
from model import JOHMRLite
import os
import json
import pdb
import scipy.misc
import matplotlib.pyplot as plt
import open3d as o3d
import torch.optim as optim
from pytorch3d.io import save_obj


# load cad model
device = torch.device("cuda:0")
obj_path = 'annotate/45354'
verts, faces, part_segs, _ = merge_meshes(obj_path, device)
verts[:,1:] *= -1  # pytorch3d -> world coordinate
obj_verts = verts.to(device)
obj_faces = faces.to(device)
obj_size = np.asarray([51.5,73.4,46.0]) # length, height, width (x, y, z), cm

# fridge (home) np.asarray([600, 1450, 620])
# dishwasher (yang)  obj_size = np.asarray([600, 800, 600])
# laptop (large) obj_size = np.asarray([415, 15, 280])
# fridge (small)  obj_size = np.asarray([450, 825, 472])
# trashcan (outside) obj_size = np.asarray([650, 1050, 635])
# 677 x 1100 x 665
# load image
img_path = 'annotate/images-0001.jpg'
focal_x = 983#1505 #1680
focal_y = 983#1505 #1680
img_square = 1280
img_small = 256

#mask_path = 'annotate/objmask-init.png'
#smplv2d_path = 'annotate/smpl2d-0001.npy'
#smplmesh_path = 'annotate/smplmesh-0001.obj'
#smplmesh_calibrate_path = 'annotate/smplmesh_calibrate.obj'


#smplv2d = np.load(smplv2d_path)                  # load 2D points
#smplmesh =  o3d.io.read_triangle_mesh(smplmesh_path) # load smpl mesh
#smplmesh_calibrate =  o3d.io.read_triangle_mesh(smplmesh_calibrate_path) # load smpl mesh
#hverts = torch.from_numpy(np.asarray(smplmesh.vertices)).float()
#hfaces = torch.from_numpy(np.asarray(smplmesh.triangles)).long()
#hverts_cal = torch.from_numpy(np.asarray(smplmesh_calibrate.vertices)).float()
#hfaces_cal = torch.from_numpy(np.asarray(smplmesh_calibrate.triangles)).long()
#human_height = 1770 #mm

# render
_, phong_renderer = initialize_render(device, focal_x, focal_y, img_square, img_small)


global x_offset, y_offset, z_offset, yaw, pitch, roll, rot_alpha, rot_beta, rot_gamma
x_offset = 0.0
y_offset = 0.0
z_offset = 2.0
yaw = 0.0
pitch = 0.0
roll = 0.0
valstep = 0.01

# initialize model
image_bg = np.array(Image.open(img_path))/255.0
img_h = image_bg.shape[0]
img_w = image_bg.shape[1]

model = JOHMRLite(obj_verts, obj_faces, phong_renderer, img_h, img_w)

'''
# optimize human mesh
optimizer = optim.Adam(model.parameters(), lr=0.1)
for iteration in range(150):
    optimizer.zero_grad()
    _, _, _, _, loss = model(obj_size, x_offset, y_offset, z_offset, yaw, pitch, roll)
    loss.backward()
    optimizer.step()
    print('Iteration %d loss %.4f' % (iteration, loss.data))
print(model.smpl_offset)
'''


def display_img(model, alpha):
    image_bg = np.array(Image.open(img_path))/255.0
    img_h = image_bg.shape[0]
    img_w = image_bg.shape[1]

    global x_offset, y_offset, z_offset, yaw, pitch, roll, rot_alpha, rot_beta, rot_gamma

    vis_image, rot_alpha, rot_beta, rot_gamma = model(obj_size, x_offset, y_offset, z_offset, yaw, pitch, roll)
    image = vis_image.detach().cpu().numpy().squeeze()

    #objmask = np.array(Image.open(mask_path))#/255.0  # object mask
    #objmask = np.repeat(objmask[:,:,np.newaxis], 3, axis=2)
    #objmask[:,:,0] *= 0
    #objmask[:,:,1] *= 0
    rgb_mask = image_bg.astype(np.float32) #cv2.addWeighted(objmask.astype(np.float32), 0.5, image_bg.astype(np.float32), 0.5, 0.0)

    frame_img = np.zeros((img_square, img_square,3))
    start = int((max(img_h, img_w) - min(img_h, img_w))/2) - 1
    end = start + min(img_h, img_w)
    if img_h > img_w:
        frame_img[:, start:end,  :] = rgb_mask
    else:
        frame_img[start:end, :, :] = rgb_mask
    rgb_mask = frame_img
    alpha = min(1.0, max(0.0,alpha))
    img_blend = cv2.addWeighted(image.astype(np.float32), alpha, rgb_mask.astype(np.float32), 1-alpha, 0.0)
    img_blend = cv2.resize(img_blend, dsize=(800, 800), interpolation=cv2.INTER_NEAREST)

    return img_blend


img_blend = display_img(model, 0.5)
h,w,_ = img_blend.shape
img_blend = np.uint8(img_blend*255)
qimage = QtGui.QImage(img_blend.data, h, w, 3*h, QtGui.QImage.Format_RGB888)



class Annotate(QtWidgets.QWidget):

    def __init__(self):
        super(Annotate, self).__init__()
        self.initUI()

    def initUI(self):
        QtWidgets.QToolTip.setFont(QtGui.QFont('Test', 10))

        # Show  image
        self.pic = QtWidgets.QLabel(self)
        self.pic.setGeometry(10, 10, 800, 800)
        self.pic.setPixmap(QtGui.QPixmap(qimage))


        self.alpha = 0.5

        # Show button
        btn1 = QtWidgets.QPushButton('Offset Z-', self)
        btn1.resize(btn1.sizeHint())
        btn1.clicked.connect(lambda: self.fun('dec_oz'))
        btn1.move(900, 10)
        btn2 = QtWidgets.QPushButton('Offset Z+', self)
        btn2.resize(btn2.sizeHint())
        btn2.clicked.connect(lambda: self.fun('inc_oz'))
        btn2.move(1000, 10)
        self.textbox1 = QtWidgets.QLineEdit(self)
        self.textbox1.move(1150, 10)
        self.textbox1.resize(100,25)
        self.textbox1.setText(str(z_offset))

        btn7 = QtWidgets.QPushButton('Offset X-', self)
        btn7.resize(btn7.sizeHint())
        btn7.clicked.connect(lambda: self.fun('dec_ox'))
        btn7.move(900, 150)
        btn8 = QtWidgets.QPushButton('Offset X+', self)
        btn8.resize(btn8.sizeHint())
        btn8.clicked.connect(lambda: self.fun('inc_ox'))
        btn8.move(1000, 150)
        self.textbox4 = QtWidgets.QLineEdit(self)
        self.textbox4.move(1150, 150)
        self.textbox4.resize(100,25)
        self.textbox4.setText(str(x_offset))

        btn9 = QtWidgets.QPushButton('Offset Y-', self)
        btn9.resize(btn9.sizeHint())
        btn9.clicked.connect(lambda: self.fun('dec_oy'))
        btn9.move(900, 190)
        btn10 = QtWidgets.QPushButton('Offset Y+', self)
        btn10.resize(btn10.sizeHint())
        btn10.clicked.connect(lambda: self.fun('inc_oy'))
        btn10.move(1000, 190)
        self.textbox5 = QtWidgets.QLineEdit(self)
        self.textbox5.move(1150, 190)
        self.textbox5.resize(100,25)
        self.textbox5.setText(str(y_offset))

        btn11 = QtWidgets.QPushButton('Yaw-', self)
        btn11.resize(btn11.sizeHint())
        btn11.clicked.connect(lambda: self.fun('dec_yaw'))
        btn11.move(900, 250)
        btn12 = QtWidgets.QPushButton('Yaw+', self)
        btn12.resize(btn12.sizeHint())
        btn12.clicked.connect(lambda: self.fun('inc_yaw'))
        btn12.move(1000, 250)
        self.textbox6 = QtWidgets.QLineEdit(self)
        self.textbox6.move(1150, 250)
        self.textbox6.resize(100,25)
        self.textbox6.setText(str(yaw))

        btn13 = QtWidgets.QPushButton('Pitch-', self)
        btn13.resize(btn13.sizeHint())
        btn13.clicked.connect(lambda: self.fun('dec_pitch'))
        btn13.move(900, 290)
        btn14 = QtWidgets.QPushButton('Pitch+', self)
        btn14.resize(btn14.sizeHint())
        btn14.clicked.connect(lambda: self.fun('inc_pitch'))
        btn14.move(1000, 290)
        self.textbox7 = QtWidgets.QLineEdit(self)
        self.textbox7.move(1150, 290)
        self.textbox7.resize(100,25)
        self.textbox7.setText(str(pitch))

        btn15 = QtWidgets.QPushButton('Roll-', self)
        btn15.resize(btn15.sizeHint())
        btn15.clicked.connect(lambda: self.fun('dec_roll'))
        btn15.move(900, 330)
        btn16 = QtWidgets.QPushButton('Roll+', self)
        btn16.resize(btn16.sizeHint())
        btn16.clicked.connect(lambda: self.fun('inc_roll'))
        btn16.move(1000, 330)
        self.textbox8 = QtWidgets.QLineEdit(self)
        self.textbox8.move(1150, 330)
        self.textbox8.resize(100,25)
        self.textbox8.setText(str(roll))


        btn22 = QtWidgets.QPushButton('Vis-', self)
        btn22.resize(btn22.sizeHint())
        btn22.clicked.connect(lambda: self.fun('dec_vis'))
        btn22.move(900, 550)
        btn23 = QtWidgets.QPushButton('Vis+', self)
        btn23.resize(btn23.sizeHint())
        btn23.clicked.connect(lambda: self.fun('inc_vis'))
        btn23.move(1000, 550)

        btn21 = QtWidgets.QPushButton('Save', self)
        btn21.resize(btn21.sizeHint())
        btn21.clicked.connect(lambda: self.fun('save'))
        btn21.move(1000, 500)


        self.setGeometry(300, 300, 2000, 1500)
        self.setWindowTitle('JOHMR Annotation Tool -- Sam Xu')
        self.show()

    # Connect button to image updating
    def fun(self, arguments):
        global x_offset, y_offset, z_offset, yaw, pitch, roll, rot_alpha, rot_beta, rot_gamma

        if arguments == 'dec_oz':
            z_offset -= valstep
        elif arguments == 'inc_oz':
            z_offset += valstep
        elif arguments == 'dec_ox':
            x_offset -= valstep
        elif arguments == 'inc_ox':
            x_offset += valstep
        elif arguments == 'dec_oy':
            y_offset -= valstep
        elif arguments == 'inc_oy':
            y_offset += valstep
        elif arguments == 'dec_yaw':
            yaw -= valstep
        elif arguments == 'inc_yaw':
            yaw += valstep
        elif arguments == 'dec_pitch':
            pitch -= valstep
        elif arguments == 'inc_pitch':
            pitch += valstep
        elif arguments == 'dec_roll':
            roll -= valstep
        elif arguments == 'inc_roll':
            roll += valstep

        elif arguments == 'save':
            # save obj orientation
            text_file = 'annotate/3d_info.txt'
            with open(text_file, "w") as myfile:
                myfile.write('yaw: '+str(round(yaw,3))+'\n')
                myfile.write('pitch: '+str(round(pitch,3))+'\n')
                myfile.write('roll: '+str(round(roll,3))+'\n')
                myfile.write('rot_alpha: '+str(round(rot_alpha,3))+'\n')
                myfile.write('rot_beta: '+str(round(rot_beta,3))+'\n')
                myfile.write('rot_gamma: '+str(round(rot_gamma,3))+'\n')
                myfile.write('x_offset: '+str(round(x_offset,3))+'\n')
                myfile.write('y_offset: '+str(round(y_offset,3))+'\n')
                myfile.write('z_offset: '+str(round(z_offset,3))+'\n')
                myfile.write('obj_size: '+str(obj_size[0])+','+str(obj_size[1])+','+str(obj_size[2])+'\n')
                myfile.write('\n')

            # save human & obj meshes
            #save_hverts = model.smpl_verts_output.detach().cpu().numpy()
            #save_hfaces = hfaces.detach().cpu().numpy()
            save_objverts = model.obj_verts_output.detach().cpu().numpy()
            save_objfaces = obj_faces.detach().cpu().numpy()

            #verts = np.concatenate((save_hverts, save_objverts), axis=0)
            #faces = np.concatenate((save_hfaces, save_objfaces + save_hverts.shape[0]), axis=0)
            save_obj('annotate/object.obj', torch.from_numpy(save_objverts), torch.from_numpy(save_objfaces))
            #save_obj('annotate/person.obj', torch.from_numpy(save_hverts), torch.from_numpy(save_hfaces))
            #save_obj('annotate/joint.obj', torch.from_numpy(verts), torch.from_numpy(faces))

        elif arguments == 'dec_vis':
            self.alpha -= 0.1

        elif arguments == 'inc_vis':
            self.alpha += 0.1

        else:
            print('not implemented')

        self.textbox4.setText(str(round(x_offset,3)))
        self.textbox5.setText(str(round(y_offset,3)))
        self.textbox6.setText(str(round(yaw,3)))
        self.textbox7.setText(str(round(pitch,3)))
        self.textbox8.setText(str(round(roll,3)))
        self.textbox1.setText(str(round(z_offset,3)))

        img = display_img(model, self.alpha)
        img = np.uint8(img*255)
        h,w,_ = img.shape
        qimage = QtGui.QImage(img.data, h, w, 3*h, QtGui.QImage.Format_RGB888)
        self.pic.setPixmap(QtGui.QPixmap(qimage))

def main():

    app = QtWidgets.QApplication(sys.argv)
    ex = Annotate()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
