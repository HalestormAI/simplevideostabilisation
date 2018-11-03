import cv2
import constants
import logging
import numpy as np
from Plotter import Plotter

class VideoStabilisationRunner(object):

    def __init__(self, args, stabiliser_class, complete_callback = None, error_callback = None):
        self.input_path = args.input_file
        self.output_file = args.output_file
        self.display_output = args.display_output
        self.stabiliser_class = stabiliser_class

        self.should_output_file = self.output_file is not None
        
        self.cap = None

        self.raw_frame = None
        self.stabilised_frame = None
        self.frame_number = 0

        self.running = False

        self.complete_callback = complete_callback
        self.error_callback = error_callback
        self.plotter = Plotter()

    def start(self):
        self.cap = cv2.VideoCapture(self.input_path)
        self.running = True

        self.stabiliser = self.stabiliser_class()

        if self.display_output:
            cv2.namedWindow(constants.DISPLAY_WINDOW_NAME)
            cv2.namedWindow(constants.STABILISED_WINDOW_NAME)

        while self.running:
            self.next()

    def complete(self):
        self.cap.release()
        cv2.destroyWindow(constants.DISPLAY_WINDOW_NAME)
        cv2.destroyWindow(constants.STABILISED_WINDOW_NAME)

        if self.complete_callback is not None:
            self.complete_callback()

    def next(self):
        success_flag, self.raw_frame = self.cap.retrieve()
        if not success_flag or self.raw_frame is None:
            logging.info("Video file complete, ending")
            self.complete()
            return
        
        self.stabilised_frame = self.stabiliser.stabilise(self.raw_frame, self.frame_number)

        if self.display_output:
            cv2.imshow(constants.DISPLAY_WINDOW_NAME, self.raw_frame)
            cv2.imshow(constants.STABILISED_WINDOW_NAME, self.stabilised_frame)
            k = cv2.waitKey(10) & 0xFF
            should_quit = self.keyHandler(k)
            if should_quit:
                return

        self.frame_number += 1

    def keyHandler(self, k):
        if k in (27, ord('q'), ord('Q')):
            self.running = False
            logging.info("User stopped execution")
            return True
        elif k == ord('p'):
            self.plotter.plot_displacements(np.array(self.stabiliser.displacement_history))
        elif k == ord('t'):
            self.plotter.plot_position(np.array(self.stabiliser.displacement_history))
        return False