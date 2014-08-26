#!/usr/bin/env python

import argparse
import cv2
from numpy import empty, nan
import os
import sys
import time

import CMT
import numpy as np
import util


CMT = CMT.CMT()

parser = argparse.ArgumentParser(description='Track an object.')

parser.add_argument('inputpath', nargs='?', help='The input path.')
parser.add_argument('--challenge', dest='challenge', action='store_true', help='Enter challenge mode.')
parser.add_argument('--preview', dest='preview', action='store_const', const=True, default=None, help='Force preview')
parser.add_argument('--no-preview', dest='preview', action='store_const', const=False, default=None, help='Disable preview')
parser.add_argument('--no-scale', dest='estimate_scale', action='store_false', help='Disable scale estimation')
parser.add_argument('--with-rotation', dest='estimate_rotation', action='store_true', help='Enable rotation estimation')
parser.add_argument('--bbox', dest='bbox', help='Specify initial bounding box.')
parser.add_argument('--pause', dest='pause', action='store_true', help='Specify initial bounding box.')
parser.add_argument('--output-dir', dest='output', help='Specify a directory for output data.')
parser.add_argument('--quiet', dest='quiet', action='store_true', help='Do not show graphical output (Useful in combination with --output-dir ).')
parser.add_argument('--skip', dest='skip', action='store', default=None, help='Skip the first n frames', type=int)

args = parser.parse_args()

CMT.estimate_scale = args.estimate_scale
CMT.estimate_rotation = args.estimate_rotation

if args.pause:
    pause_time = 0
else:
    pause_time = 10

if args.output is not None:
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not os.path.isdir(args.output):
        raise Exception(args.output + ' exists, but is not a directory')

if args.challenge:
    with open('images.txt') as f:
        images = [line.strip() for line in f]

    init_region = np.genfromtxt('region.txt', delimiter=',')
    num_frames = len(images)

    results = empty((num_frames, 4))
    results[:] = nan

    results[0, :] = init_region

    frame = 0

    im0 = cv2.imread(images[frame])
    im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im_draw = np.copy(im0)

    tl, br = (util.array_to_int_tuple(init_region[:2]), util.array_to_int_tuple(init_region[:2] + init_region[2:4]))

    try:
        CMT.initialise(im_gray0, tl, br)
        while frame < num_frames:
            im = cv2.imread(images[frame])
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            CMT.process_frame(im_gray)
            results[frame, :] = CMT.bb

            # Advance frame number
            frame += 1
    except:
        pass  # Swallow errors

    np.savetxt('output.txt', results, delimiter=',')

else:
    # Clean up
    cv2.destroyAllWindows()

    preview = args.preview

    if args.inputpath is not None:

        # If a path to a file was given, assume it is a single video file
        if os.path.isfile(args.inputpath):
            cap = cv2.VideoCapture(args.inputpath)

            #Skip first frames if required
            if args.skip is not None:
                cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, args.skip)


        # Otherwise assume it is a format string for reading images
        else:
            cap = util.FileVideoCapture(args.inputpath)

            #Skip first frames if required
            if args.skip is not None:
                cap.frame = 1 + args.skip

        # By default do not show preview in both cases
        if preview is None:
            preview = False

    else:
        # If no input path was specified, open camera device
        cap = cv2.VideoCapture(0)
        if preview is None:
            preview = True

    # Check if videocapture is working
    if not cap.isOpened():
        print 'Unable to open video input.'
        sys.exit(1)

    while preview:
        status, im = cap.read()
        cv2.imshow('Preview', im)
        k = cv2.waitKey(10)
        if not k == -1:
            break

    # Read first frame
    status, im0 = cap.read()
    im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im_draw = np.copy(im0)

    if args.bbox is not None:
        # Try to disassemble user specified bounding box
        values = args.bbox.split(',')
        try:
            values = [int(v) for v in values]
        except:
            raise Exception('Unable to parse bounding box')
        if len(values) != 4:
            raise Exception('Bounding box must have exactly 4 elements')
        bbox = np.array(values)

        # Convert to point representation, adding singleton dimension
        bbox = util.bb2pts(bbox[None, :])

        # Squeeze
        bbox = bbox[0, :]

        tl = bbox[:2]
        br = bbox[2:4]
    else:
        # Get rectangle input from user
        (tl, br) = util.get_rect(im_draw)

    print 'using', tl, br, 'as init bb'


    CMT.initialise(im_gray0, tl, br)

    # Aspect ratio
    width = br[0] - tl[0]
    height = br[1] - tl[1]
    AR = float(width) / float(height)
    numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    data = np.zeros((6, numFrames-1))
    frame = 0
    itr = 0
    basePath = os.path.dirname(args.inputpath)
    # make the directory for storing images
    savePath = os.path.splitext(args.inputpath)[0]
    if not os.path.exists(savePath):
        os.mkdir(savePath)
        os.mkdir(savePath + '/frame')
        os.mkdir(savePath + '/imgs')
    F_frame = open(savePath + '/full.txt', 'w')
    F_img = open(savePath + '/imgs.txt', 'w')

    #save the first frame
    line = "frame/frame-" + '%04d'%frame + '.jpg 1 ' + '%4d'%tl[0] + ' ' + '%4d'%tl[1] + ' ' + '%4d'%(br[0] - tl[0]) + ' ' + '%4d'%(br[1] - tl[1]) + '\n'
    F_frame.write(line)
    cv2.imwrite(savePath + "/frame/frame-" + '%04d'%frame + '.jpg', im0)
    line = "imgs/img-" + '%04d'%frame + '.jpg\n'
    F_img.write(line)
    ROI = im0[tl[1]:br[1], tl[0]:br[0]]
    cv2.imwrite(savePath + "/imgs/img-" + '%04d'%frame + '.jpg', ROI)
    frame += 1
    while True:
        # Read image
        status, im = cap.read()
        if not status:
            break
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_draw = np.copy(im)

        tic = time.time()
        scale = CMT.process_frame(im_gray)
        toc = time.time()

        # Display results

        # Draw updated estimate
        if CMT.has_result:

            cv2.line(im_draw, CMT.tl, CMT.tr, (255, 0, 0), 4)
            cv2.line(im_draw, CMT.tr, CMT.br, (255, 0, 0), 4)
            cv2.line(im_draw, CMT.br, CMT.bl, (255, 0, 0), 4)
            cv2.line(im_draw, CMT.bl, CMT.tl, (255, 0, 0), 4)
            data[0, itr] = frame
            data[1, itr] = scale
            data[2, itr] = CMT.tl[0]
            data[3, itr] = CMT.tl[1]
            data[4, itr] = CMT.br[0]
            data[5, itr] = CMT.br[1]
            itr += 1

        util.draw_keypoints(CMT.tracked_keypoints, im_draw, (255, 255, 255))
        # this is from simplescale
        util.draw_keypoints(CMT.votes[:, :2], im_draw)  # blue
        util.draw_keypoints(CMT.outliers[:, :2], im_draw, (0, 0, 255))

        if args.output is not None:
            # Original image
            cv2.imwrite('{0}/input_{1:08d}.png'.format(args.output, frame), im)
            # Output image
            cv2.imwrite('{0}/output_{1:08d}.png'.format(args.output, frame), im_draw)

            # Keypoints
            with open('{0}/keypoints_{1:08d}.csv'.format(args.output, frame), 'w') as f:
                f.write('x y\n')
                np.savetxt(f, CMT.tracked_keypoints[:, :2], fmt='%.2f')

            # Outlier
            with open('{0}/outliers_{1:08d}.csv'.format(args.output, frame), 'w') as f:
                f.write('x y\n')
                np.savetxt(f, CMT.outliers, fmt='%.2f')

            # Votes
            with open('{0}/votes_{1:08d}.csv'.format(args.output, frame), 'w') as f:
                f.write('x y\n')
                np.savetxt(f, CMT.votes, fmt='%.2f')

            # Bounding box
            with open('{0}/bbox_{1:08d}.csv'.format(args.output, frame), 'w') as f:
                f.write('x y\n')
                # Duplicate entry tl is not a mistake, as it is used as a drawing instruction
                np.savetxt(f, np.array((CMT.tl, CMT.tr, CMT.br, CMT.bl, CMT.tl)), fmt='%.2f')

        if not args.quiet:
            cv2.imshow('main', im_draw)

            # Check key input
            k = cv2.waitKey(pause_time)
            key = chr(k & 255)
            if key == 'q':
                break
            if key == 'd':
                import ipdb; ipdb.set_trace()

        # Remember image
        im_prev = im_gray

        # Advance frame number
        frame += 1

        print '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT.center[0], CMT.center[1], CMT.scale_estimate, CMT.active_keypoints.shape[0], 1000 * (toc - tic), frame)
    # Here is where we save the images into a subfolder based on the name of the input video
    # If the input video is input.avi, then a folder will be created called input with sub folders
    # input/frame and input/img.  Two files will be created in input called input/full.txt and input/imgs.txt
    # these files can be used with opencv_createsample program to prep for cascade training
    # The aspect ratio of these samples will be constant and determined by the first bounding rectangle

    # first remove zero entries
    data = np.delete(data, np.where(data[0, :]==0), axis=1)
    # now remove entries where the scale differs by the norm by a more than a standard deviation
    mean = np.mean(data[1, :])
    std = np.std(data[1, ])
    data = np.delete(data, np.where( np.bitwise_or(data[1, :] > mean + std, data[1, :] < mean - std)), axis=1)

    cap = cv2.VideoCapture(args.inputpath)
    success, I = cap.read()
    frame = 1
    itr = 0
    # now go back through all of the images, saving the needed metadata for training the classifier
    while True:
        success, I = cap.read()
        if not success:
            break
        if data.shape[1] == itr:
            break
        if frame == data[0, itr]:

            # if this is a frame included in the dataset to save
            # Calculate the center and then the correct bounding rectangle to maintain the aspect ratio defined above
            center = (data[4:6, itr] + data[2:4, itr])/2
            width = data[4, itr] - data[2, itr]
            height = data[5, itr] - data[3, itr]
            AR_ = width / height
            if AR > AR_:
                # The original width / height > this width / height.... Always scale upwards
                # Thus this is skinnier than original, so widen it
                width = AR*height
            if AR < AR_:
                # This is fatter than the original so increase the height
                height = width / AR
            TL = center - np.array((width/2, height/2))
            TL = TL.astype(int)
            width = int(width)
            height = int(height)
            line = "frame/frame-" + '%04d'%frame + '.jpg 1 ' + '%4d'%TL[0] + ' ' + '%4d'%TL[1] + ' ' + '%4d'%width + ' ' + '%4d'%height + '\n'
            F_frame.write(line)
            cv2.imwrite(savePath + "/frame/frame-" + '%04d'%frame + '.jpg', I)
            line = "imgs/img-" + '%04d'%frame + '.jpg\n'
            F_img.write(line)


            #ROI = I[data[3, itr]:data[5, itr], data[2, itr]:data[4, itr]]
            if width == 0 or height == 0:
                print width + " " + height
                continue
            ROI = I[TL[1]:TL[1]+height, TL[0]:TL[0]+width]
            cv2.imwrite(savePath + "/imgs/img-" + '%04d'%frame + '.jpg', ROI)
            cv2.imshow("Frame", I)
            cv2.imshow("ROI", ROI)
            cv2.waitKey(1)
            itr += 1
        frame += 1
    F_frame.close()
    F_img.close()
    cv2.destroyAllWindows()
