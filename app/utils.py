import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import imageio


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

def load_video(path:str) -> List[float]:
    cap = cv2.VideoCapture(path)
    print(cap)
    frames = []
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(),
                                           oov_token="",
                                           invert=True
                                          )



def load_alignments(path:str) -> List[str]:
    print('fasd', path)
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'),(-1)))[1:]


def load_data(path: str):
    path = bytes.decode(path.numpy())
    print('path', path)
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..', 'data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('..', 'data', 'alignments', 's1', f'{file_name}.align')
    # alignment_path = os.path.join('data', 'alignments', 's1', 'bbaf2n.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    # alignments = None
    return frames, alignments