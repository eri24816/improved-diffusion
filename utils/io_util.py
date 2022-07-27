import pydub 
import numpy as np
import pickle, json
import numpy

def read_mp3(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return np.float32(y) / 2**15
    else:
        return y

def write_mp3(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

# Copied from MuseMorphose
def pickle_load(f):
    return pickle.load(open(f, 'rb'))

def pickle_dump(obj, f):
    pickle.dump(obj, open(f, 'wb'))

def json_load(f):
    return json.load(open(f, 'rb'))

def json_dump(obj, f):
    json.dump(obj, open(f, 'wb'))