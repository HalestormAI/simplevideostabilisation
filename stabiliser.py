import logging
import cv2
from TrackingFrameState import TrackingFrameState
import numpy as np

class AbstractStabiliser(object):
    def __init__(self):
        self.frame_number = 0
        self.current_raw_frame = None
        self.previous_raw_frame = None
        self.current_stabilised_frame = None
        self.previous_stabilised_frame = None

    """Abstract method to handle stabilisation of a video frame."""

    def stabilise(self, frame, frame_number):
        raise NotImplementedError(
            "This method should be overriden in the child class and should not be called directly")

    def add_frame(self, frame, frame_number):
        self.current_raw_frame = frame
        self.frame_number = frame_number

class KalmanFlowStabiliser(AbstractStabiliser):
    def __init__(self):
        super(KalmanFlowStabiliser, self).__init__()

    def stabilise(self, frame, frame_number):
        logging.warning(
            "Not currently implemented, returning the original frame")
        return frame.copy()


class FlowOnlyStabiliser(AbstractStabiliser):
    def __init__(self):
        super(FlowOnlyStabiliser, self).__init__()

        self.current_frame_state = None
        self.previous_frame_state = None

        self.displacement_history = []

    def stabilise(self, frame, frame_number):
        self.add_frame(frame, frame_number)

        self.current_frame_state = TrackingFrameState(frame, frame_number)

        Tform = None
        displacement = (0,0)
        if self.previous_frame_state is not None and self.previous_frame_state.has_features and self.current_frame_state.has_features:
            matches = self.current_frame_state.match(self.previous_frame_state)
            # displacement = self.__get_median_motion(matches)
            displacement, Tform = self.__fit_motion_vector_ransac(matches)

        if Tform is None:
            self.current_stabilised_frame = self.current_raw_frame.copy()
        else:
            self.current_stabilised_frame = self.previous_stabilised_frame.copy()
            self.current_stabilised_frame = cv2.warpAffine(self.current_raw_frame, Tform, dsize=self.current_raw_frame.shape[1::-1], dst=self.current_stabilised_frame, borderMode=cv2.BORDER_TRANSPARENT)

        # cv2.putText(self.current_stabilised_frame, ("HAVE TFORM" if Tform is not None else "No Change"), (40,40), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0,255,0) if Tform is not None else (0,0,255))

        self.displacement_history.append(displacement)



        self.previous_frame_state = self.current_frame_state
        self.previous_stabilised_frame = self.current_stabilised_frame.copy()
        return self.current_stabilised_frame

    def __get_median_motion(self, matches):
        if len(matches) == 0:
            return (0, 0), None

        displacements = []
        for match in matches:
            prev_pt = self.previous_frame_state.keypoints[match.trainIdx].pt
            current_pt = self.current_frame_state.keypoints[match.queryIdx].pt
            displacements.append((current_pt[0] - prev_pt[0], current_pt[1] - prev_pt[1]))

        displacement = tuple(np.median(np.array(displacements), 0).astype(int))
        return displacement, None


    def __fit_motion_vector_ransac(self, matches):
        if len(matches) == 0:
            return (0, 0), None

        frame_1_pts = np.array([self.previous_frame_state.keypoints[x.trainIdx].pt for x in matches])
        frame_2_pts = np.array([self.current_frame_state.keypoints[x.queryIdx].pt for x in matches])

        T = cv2.estimateRigidTransform(frame_2_pts, frame_1_pts, False, 500, 0.5, 3)

        if T is None:
            return (0, 0), None
        displacementNp = cv2.transform(np.array([[self.current_frame_state.centre]], dtype=np.float32), T)
        return tuple(displacementNp[0][0]), T
