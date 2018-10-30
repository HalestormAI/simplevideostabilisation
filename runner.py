import cv2
import constants
import logging

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

    def start(self):
        self.cap = cv2.VideoCapture(self.input_path)
        self.running = True

        self.stabiliser = self.stabiliser_class()

        if self.display_output:
            cv2.namedWindow(constants.DISPLAY_WINDOW_NAME)

        while self.running:
            self.next()

    def complete(self):
        self.cap.release()
        cv2.destroyWindow(constants.DISPLAY_WINDOW_NAME)

        if self.complete_callback is not None:
            self.complete_callback()

    def next(self):
        success_flag, self.raw_frame = self.cap.retrieve()
        if not success_flag or self.raw_frame is None:
            logging.info("Video file complete, ending")
            self.complete()
            return

        if self.display_output:
            cv2.imshow(constants.DISPLAY_WINDOW_NAME, self.raw_frame)
            k = cv2.waitKey(10) & 0xFF
            should_quit = self.keyHandler(k)
            if should_quit:
                return
        
        self.stabilised_frame = self.stabiliser.stabilise(self.raw_frame, self.frame_number)
        self.frame_number += 1

    def keyHandler(self, k):
        if k == 27 or k == 'q' or k == 'Q':
            self.running = False
            logging.info("User stopped execution")
            return True
        return False