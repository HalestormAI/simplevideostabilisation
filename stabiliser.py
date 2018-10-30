import logging

class AbstractStabiliser(object):
    def __init__(self):
        self.frame_number = 0
        self.current_raw_frame = None
        self.current_stabilised_frame = None

    """Abstract method to handle stabilisation of a video frame."""
    def stabilise(self, frame, frame_number ):
        raise NotImplementedError("This method should be overriden in the child class and should not be called directly")

class KalmanFlowStabiliser(AbstractStabiliser):
    def __init__(self):
        super(KalmanFlowStabiliser, self).__init__()

    def stabilise(self, frame, frame_number):
        logging.warning("Not currently implemented, returning the original frame")
        return frame.copy()