import cv2


class TrackingFrameState(object):
    """
        A class to contain frame state properties required for tracking, including the frame and feature detections.
    """
    flann = None
    orb = None

    def __init__(self, frame, frame_number):
        self.grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_number = frame_number

        self.keypoints, self.descriptors = self.__get_detector().detectAndCompute(self.grey, None)

        self.centre = self.grey.shape[1]/2, self.grey.shape[0]/2

    @property
    def has_features(self):
        return self.keypoints is not None and self.descriptors is not None and len(self.keypoints) > 2


    def match(self, prev_state, ratio_thresh=0.7):
        """
        Matches the current state against a previous one using K-Nearest Neighbours (FLANN).
        Performs Lowe's distance check to filter matches. The ratio defaults to 0.7 to echo the docs. If set to `None`,
        the distance check will be skipped.
        """
        knn_matches = self.__get_matcher().knnMatch(
            self.descriptors, prev_state.descriptors, 2)

        if ratio_thresh is not None:
            good_matches = []
            for match in knn_matches:
                if len(match) < 2:
                    continue

                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            return good_matches
        else:
            return [m for m, n in knn_matches]

    def __get_detector(self):
        if TrackingFrameState.orb is None:    
            TrackingFrameState.orb = cv2.ORB_create()

        return TrackingFrameState.orb

    def __get_matcher(self):
        """
        TODO: This should be parameterised better...
        """
        if TrackingFrameState.flann is None:    
            FLANN_INDEX_LSH = 6

            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2

            search_params = dict(checks=50)
            TrackingFrameState.flann = cv2.FlannBasedMatcher(index_params, search_params)

        return TrackingFrameState.flann
