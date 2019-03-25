import numpy as np
import pandas as pd
import sys
import os
import json
import pickle
import cv2
import csv
from tqdm import tqdm
from config import cfg


class CSV():
    def __init__(self, csv_path, map_path):
        self.csv_path   = csv_path
        self.map_path   = map_path


        self.map_df = pd.read_csv(self.map_path, header=None)
        self.map = {}
        for i in range(len(self.map_df)):
            self.map[i] = self.map_df[0][i]
        self.image_names = []
        self.images_data = {}
        self.base_dir    = None

        if self.base_dir is None:
            self.base_dir = os.path.dirname(self.csv_path)

        try:
            with self._open_for_csv(map_path) as file:
                self.classes = self._read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels  = {}
        self.rlabels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
            self.rlabels[key] = value

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(csv_path) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())
        print ((self.image_names))
        print (self.labels)
#       print (self.image_data)
        self._detections = {}
        for key in(self.image_data.keys()):
            boxes = self.image_data[key]
            bboxes = []
            categories = []
            for detection in boxes:
                bbox = np.zeros((1,5))
                bbox[0][4] = self.rlabels[detection['class']]
                bbox[0][0] = detection['x1']
                bbox[0][1] = detection['y1']
                bbox[0][2] = detection['x2']
                bbox[0][3] = detection['y2']
                bboxes.append(bbox)
            bboxes = np.array(bboxes)
            self._detections[key] = bboxes






    def read_img(self, img_name):
        print (bytes.decode(img_name))
        mat = cv2.imread(os.path.join(self.base_dir, bytes.decode(img_name)))
        return mat.astype(np.float32)

    def detections(self, img_name):
        detections = self._detections[bytes.decode(img_name)]
        return detections.astype(float).copy()[0]

    def get_all_img(self):
        return self.image_names



    def _read_classes(self, csv_reader):
        """ Parse the classes file given by csv_reader.
        """
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _read_annotations(self, csv_reader, classes):
        """ Read annotations from the csv_reader.
        """
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []
            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue
            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result
    def _open_for_csv(self, path):
        """ Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')
