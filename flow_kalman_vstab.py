import cv2
import argparse
import constants
import logging

from runner import VideoStabilisationRunner
from stabiliser import FlowOnlyStabiliser

logging.basicConfig(level=constants.LOG_LEVEL, format=constants.LOG_FORMAT)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-f', required=True, help='Input video to stabilise. Enter 0 for webcam stream')
    parser.add_argument('--output-file', '-o', help='Output file in which to store results. Mandatory unless `--display-output` is provided')
    parser.add_argument('--display-output', '-d', action='store_true', help='Display output to a UI window')

    args = parser.parse_args()
    if args.display_output is False and args.output_file is None:
        parser.error('You must provide either an output file (`--output-file`) or the flag `--display-output`')
    
    if args.input_file == "0":
        args.input_file = 0

    return args

if __name__ == '__main__':
    args = parse_args()
    logging.info("Starting")
    stab = VideoStabilisationRunner(args, FlowOnlyStabiliser)
    stab.start()