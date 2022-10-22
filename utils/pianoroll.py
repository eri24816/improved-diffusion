from __future__ import annotations
import os
import random
from torch.utils.data import DataLoader, Dataset
from mpi4py import MPI

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
        segment_length,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
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
    def __init__(self, data_dir, segment_len = 0, hop_len = 32, max_duration = 32*180, shard=0, num_shards=1, max_pieces = inf):
        print(f'Creating dataset {segment_len}')
        self.pianorolls : list[PianoRoll] = []

        file_list = list(glob.glob(os.path.join(data_dir,"*.json")))
        file_list = file_list[:max_pieces]
        for file_path in file_list:
            self.pianorolls.append(PianoRoll.load(file_path))
        
        self.segment_length = segment_len
        if segment_len:
            num_segments = [ceil(pianoroll.duration/hop_len) for pianoroll in self.pianorolls]
            
            self.segment_id_to_piece = []
            for pianoroll, num_seg in zip(self.pianorolls, num_segments):
                self.segment_id_to_piece += [(pianoroll,hop_len*i, hop_len*i + segment_len)for i in range(num_seg)]
            # slice shard
            self.segment_id_to_piece = self.segment_id_to_piece[shard:][::num_shards]
            self.length = len(self.segment_id_to_piece)
        else:
            self.length = len(self.pianorolls)
            self.max_duration = min(max_duration,max([pianoroll.duration for pianoroll in self.pianorolls]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> torch.Tensor:
        if self.segment_length:
            piece, start, end = self.segment_id_to_piece[idx]
            return piece.to_tensor(start, end, padding = True)/64-1 # [-1,1)
        else:
            return self.pianorolls[idx].to_tensor(0, self.max_duration, padding = True)/64-1 # [-1,1)

    def get_piano_roll(self, idx) -> PianoRoll:
        if self.segment_length:
            piece, start, end = self.segment_id_to_piece[idx]
            return piece.slice(start, end)
        else:
            return self.pianorolls[idx].slice(0, self.max_duration)

    def get_all_piano_rolls(self) -> list[PianoRoll]:
        return [self.get_piano_roll(i) for i in range(len(self))]

class Note:
    def __init__(self,onset,pitch,velocity,offset=None) -> None:
        self.onset = onset
        self.pitch = pitch
        self.velocity = velocity
        self.offset = offset

    def __repr__(self) -> str:
        return f'Note({self.onset},{self.pitch},{self.velocity},{self.offset})'

    def __gt__(self, other):
        if self.onset == other.onset:
            return self.pitch > other.pitch
        return self.onset > other.onset

class PianoRoll:
    '''
    
    '''
    @ staticmethod
    def load(path):
        '''
        Load a pianoroll from a json file
        '''
        return PianoRoll(io_util.json_load(path))

    @ staticmethod
    def from_tensor(tens, thres = 5, normalized = False):
        '''
        Convert a tensor to a pianoroll
        '''
        if normalized:
            tens = (tens+1)*64
        tens = tens.cpu().to(torch.int32).clamp(0,127)
        data = {"onset_events":[],"pedal_events":[]}
        for t in range(tens.shape[0]):
            for p in range(tens.shape[1]):
                if tens[t,p] > thres:
                    data["onset_events"].append([t,p+21,int(tens[t,p])])

        return PianoRoll(data)
    
    @ staticmethod
    def from_midi(path):
        '''
        Load a pianoroll from a midi file
        '''
        midi = miditoolkit.midi.parser.MidiFile(path)
        data = {"onset_events":[],"pedal_events":[]}
        for note in midi.instruments[0].notes:
            note : miditoolkit.Note
            data["onset_events"].append([int(note.start*8/midi.ticks_per_beat),note.pitch,note.velocity,int(note.end*8/midi.ticks_per_beat)])
        return PianoRoll(data)

    def __init__(self, data : dict):
        # [onset time, pitch, velocity]
        self.notes = [Note(*note) for note in data["onset_events"]]
        self.notes = sorted(self.notes) # ensure the event is sorted by time

        if "pedal_events" in data:
            # Timestamps of pedal up events. (Otherwise the pedal is always down) 
            self.pedal = data["pedal_events"]
            self.pedal = sorted(self.pedal)
        else:
            self.pedal = None

        if len(self.notes):
            self.duration = self.notes[-1].onset + 16 # extend a bar to ensure the last note is covered completely
        else:
            self.duration = 0

        self._have_offset = len(self.notes) == 0 or self.notes[0].offset is not None
    
    '''
    ==================
    Utils
    ==================
    '''
    def _note_data(self):
        '''
        generator that yields (onset, pitch, velocity, offset iterator)
        '''
        for note in self.notes:
            yield note.onset, note.pitch, note.velocity, note.offset
    
    def get_offsets_with_pedal(self,pedal) -> list[int]:
        offsets = []
        next_onset = [inf]*88
        i = len(pedal)
        for onset, pitch, vel, _ in reversed(list(self._note_data())): #TODO: handle offsets if there are ones
            pitch -= 21 # midi number to piano
            while i > 0 and pedal[i-1] > onset:
                i -= 1
            if i == len(pedal):
                next_pedal_up = self.duration
            else:
                next_pedal_up = pedal[i]

            offset = min(next_onset[pitch], next_pedal_up)

            offsets.append(offset)
            next_onset[pitch] = onset
        offsets = list(reversed(offsets))
        return offsets
    '''
    ==================
    Type conversion
    ==================
    '''

    def save(self,path):

        data = {"onset_events":[],"pedal_events":[]}
        for note in self.notes:
            data["onset_events"].append([note.onset,note.pitch,note.velocity])
        data["pedal_events"] = self.pedal if self.pedal else []

        io_util.json_dump({"onset_events":self.notes,"pedal_events":self.pedal},path)

    def to_tensor(self, start_time : int = 0, end_time : int = inf, padding = False, normalized = False) -> torch.Tensor:
        '''
        Convert the pianoroll to a tensor
        '''
        if padding:
            # zero pad to end_time
            assert end_time != inf
            length = end_time - start_time
        else:
            length = min(self.duration, end_time) - start_time

        size = [length, 88]
        piano_roll = torch.zeros(size)

        for time, pitch, vel, _ in self._note_data():
            rel_time = time - start_time
            # only contain notes between start_time and end_time
            if rel_time < 0: continue
            if rel_time >= length : break
            pitch -= 21 # midi to piano
            piano_roll[rel_time,pitch] = vel

        if normalized:
            piano_roll = piano_roll/64-1
        return piano_roll
    
    def to_midi(self,path = None,apply_pedal = True) -> miditoolkit.midi.parser.MidiFile:
        '''
        Convert the pianoroll to a midi file
        '''
        midi = miditoolkit.midi.parser.MidiFile()
        midi.instruments = [miditoolkit.Instrument(program=0, is_drum=False, name='Piano')]
        midi.tempo_changes.append(miditoolkit.TempoChange(144,0))
        if apply_pedal:
            if self.pedal:
                pedal = self.pedal
            else:
                pedal = list(range(0,self.duration,32))
            offsets = self.get_offsets_with_pedal(pedal)
            for i, (onset, pitch, vel, _) in enumerate(self._note_data()):
                offset = offsets[i]
                midi.instruments[0].notes.append(
                    miditoolkit.Note(vel, pitch, int(onset*midi.ticks_per_beat/8), int(offset*midi.ticks_per_beat/8))
                )
        if not apply_pedal:
            assert self._have_offset, "Offset not found"
            for onset, pitch, vel, offset in self._note_data():
                assert offset is not None, "Offset not found"
                midi.instruments[0].notes.append(
                    miditoolkit.Note(vel, pitch, int(onset*midi.ticks_per_beat/8), int(offset*midi.ticks_per_beat/8))
                )

        if path:
            midi.dump(path)
        return midi

    '''
    ==================
    Basic operations
    ==================
    '''

    def slice(self,start_time : int = 0, end_time : int = inf) -> PianoRoll:
        '''
        Slice a pianoroll from start_time to end_time
        '''
        length = end_time - start_time
        sliced_notes = []
        sliced_pedal = []
        for time, pitch, vel, offset in self._note_data():
            rel_time = time - start_time
            # only contain notes between start_time and end_time
            if rel_time < 0: continue
            if rel_time >= length : break
            sliced_notes.append([rel_time,pitch,vel,offset])

        if self.pedal:
            for pedal in self.pedal:
                time = pedal
                rel_time = time - start_time
                # only contain pedal between start_time and end_time
                if rel_time < 0: continue
                if rel_time >= length : break
                sliced_pedal.append(rel_time)
            return PianoRoll({"onset_events":sliced_notes,"pedal_events":sliced_pedal})
        else:
            return PianoRoll({"onset_events":sliced_notes})

    def get_random_tensor_clip(self,duration,normalized = False):
        '''
        Get a random clip of the pianoroll
        '''
        start_time = random.randint(0,(self.duration-duration)//32)*32
        return self.to_tensor(start_time, start_time+duration, normalized = normalized)


if __name__ == '__main__':  
    # test
    d = PianoRollDataset('/home/eri24816/pianoroll/', 2*32,max_pieces=3)
    orig = d[0]
    file_name = '/tmp/pianoroll_test.mid'
    PianoRoll.from_tensor(orig,0,normalized=True).to_midi(file_name)
    recons = PianoRoll.from_midi(file_name).to_tensor(normalized=True)
    assert ((orig- recons[:64])**2).sum()==0