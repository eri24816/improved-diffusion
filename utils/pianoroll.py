import os
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, segment_length = 0, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = PianoRollDataset(
        data_dir,
        segment_length
    )
    print(f'Dataset size: {len(dataset)}')
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

from math import inf, ceil
import torch
import glob
import miditoolkit
from utils import io_util

class PianoRollDataset(Dataset):
    def __init__(self, data_dir, segment_len = 0, max_duration = 32*50):
        print(f'Creating dataset {segment_len}')
        self.pianorolls : list[PianoRoll] = []
        for file_path in glob.glob(os.path.join(data_dir,"*.json")):
            self.pianorolls.append(PianoRoll.load(file_path))
        self.segment_length = segment_len
        if segment_len:
            self.num_segments = [ceil(pianoroll.duration/segment_len) for pianoroll in self.pianorolls]
            
            self.segment_id_to_piece = []
            for pianoroll, num_seg in zip(self.pianorolls, self.num_segments):
                self.segment_id_to_piece += [(pianoroll,segment_len*i, segment_len*(i+1))for i in range(num_seg)]
            self.length = sum(self.num_segments)
        else:
            self.length = len(self.pianorolls)
            self.max_duration = min(max_duration,max([pianoroll.duration for pianoroll in self.pianorolls]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.segment_length:
            piece, start, end = self.segment_id_to_piece[idx]
            return piece.to_tensor(start, end, padding = True)/64-1 # [-1,1)
        else:
            return self.pianorolls[idx].to_tensor(0, self.max_duration, padding = True)/64-1 # [-1,1)

    def get_piano_roll(self, idx):
        if self.segment_length:
            piece, start, end = self.segment_id_to_piece[idx]
            return piece.slice(start, end)
        else:
            return self.pianorolls[idx].slice(0, self.max_duration)

    def get_all_piano_rolls(self):
        return [self.get_piano_roll(i) for i in range(len(self))]

class PianoRoll:
    @ staticmethod
    def load(path):
        return PianoRoll(io_util.json_load(path))

    @ staticmethod
    def from_tensor(tens, thres = 5, normalized = False):
        if normalized:
            tens = (tens+1)*64
        tens = tens.cpu().to(torch.int32).clamp(0,127)
        data = {"onset_events":[],"pedal_events":[]}
        for t in range(tens.shape[0]):
            for p in range(tens.shape[1]):
                if tens[t,p] > thres:
                    data["onset_events"].append([t,p+21,int(tens[t,p])])

        return PianoRoll(data,make_pedal=True)
    
    @ staticmethod
    def from_midi(path):
        midi = miditoolkit.midi.parser.MidiFile(path)
        data = {"onset_events":[],"pedal_events":[]}
        for note in midi.instruments[0].notes:
            note : miditoolkit.Note
            data["onset_events"].append([int(note.start*8/midi.ticks_per_beat),note.pitch,note.velocity])
        return PianoRoll(data,make_pedal=True)

    def __init__(self, data : dict, make_pedal = False):
        # [onset time, pitch, velocity]
        self.notes = data["onset_events"]
        if len(self.notes) and len(self.notes[0]) == 4:
            self.notes = [[onset,pitch,vel] for onset,pitch,vel,offset in self.notes]
        self.notes = sorted(self.notes) # ensure the event is sorted by time

        if len(self.notes):
            time, pitch, vel = self.notes[-1] # get the last note's onset
            self.duration = time + 16 # extend a bar to ensure the last note is covered completely
        else:
            self.duration = 0

        if make_pedal:
            self.pedal = list(range(0,self.duration,32))
        else:
            # Timestamps of pedal up events. (Otherwise the pedal is always down) 
            self.pedal = data["pedal_events"]
            self.pedal = sorted(self.pedal)

        self.offsets = self.get_offsets_with_pedal()

    def slice(self,start_time : int = 0, end_time : int = inf):
        length = end_time - start_time
        sliced_notes = []
        sliced_pedal = []
        for time, pitch, vel in self.notes:
            rel_time = time - start_time
            # only contain notes between start_time and end_time
            if rel_time < 0: continue
            if rel_time >= length : break
            sliced_notes.append([rel_time,pitch,vel])

        for pedal in self.pedal:
            time = pedal
            rel_time = time - start_time
            # only contain pedal between start_time and end_time
            if rel_time < 0: continue
            if rel_time >= length : break
            sliced_pedal.append(rel_time)

        return PianoRoll({"onset_events":sliced_notes,"pedal_events":sliced_pedal})


    def to_tensor(self, start_time : int = 0, end_time : int = inf, padding = False, normalized = False) -> torch.Tensor:
        if padding:
            # zero pad to end_time
            assert end_time != inf
            length = end_time - start_time
        else:
            length = min(self.duration, end_time) - start_time

        size = [length, 88]
        piano_roll = torch.zeros(size)

        for time, pitch, vel in self.notes:
            rel_time = time - start_time
            # only contain notes between start_time and end_time
            if rel_time < 0: continue
            if rel_time >= length : break
            pitch -= 21 # midi to piano
            piano_roll[rel_time,pitch] = vel

        if normalized:
            piano_roll = piano_roll/64-1
        return piano_roll

    def save(self,path):
        io_util.json_dump({"onset_events":self.notes,"pedal_events":self.pedal},path)

    def get_offsets_with_pedal(self):
        offsets = []
        next_onset = [inf]*88
        i = len(self.pedal)
        for onset, pitch, vel in reversed(self.notes):
            pitch -= 21 # midi number to piano
            while i > 0 and self.pedal[i-1] > onset:
                i -= 1
            if i == len(self.pedal):
                next_pedal_up = self.duration
            else:
                next_pedal_up = self.pedal[i]

            offset = min(next_onset[pitch], next_pedal_up)

            offsets.append(offset)
            next_onset[pitch] = onset
        offsets = list(reversed(offsets))
        return offsets

    def to_midi(self,path = None):
        midi = miditoolkit.midi.parser.MidiFile()
        midi.instruments = [miditoolkit.Instrument(program=0, is_drum=False, name='Piano')]
        notes = [note + [offset] for note, offset in zip(self.notes,self.offsets)]
        midi.tempo_changes.append(miditoolkit.TempoChange(144,0))
        for  onset, pitch, vel, offset in notes:
            midi.instruments[0].notes.append(
                miditoolkit.Note(vel, pitch, int(onset*midi.ticks_per_beat/8), int(offset*midi.ticks_per_beat/8))
            )
        if path:
            midi.dump(path)
        return midi

if __name__ == '__main__':  
    d = PianoRollDataset('/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll', 2*32)
    orig = d[150]
    file_name = '/tmp/pianoroll_test.mid'
    PianoRoll.from_tensor(orig,0,normalized=True).to_midi(file_name)
    recons = PianoRoll.from_midi(file_name).to_tensor(normalized=True)
    assert ((orig- recons[:64])**2).sum()==0