from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import os
import random
from torch.utils.data import DataLoader, Dataset

from math import inf, ceil
import torch
import glob
import miditoolkit
from utils import io_util
import pandas

class PianoRollDataset(Dataset):
    def __init__(self, data_dir, segment_len = 0, hop_len = 32, max_duration = 32*180, shard=0, num_shards=1, max_pieces = None, metadata_file = None):
        print(f'Creating dataset segment_len = {segment_len}')
        if metadata_file is not None:
            metadata = pandas.read_csv(metadata_file)
        else:
            metadata = None
        self.pianorolls : list[PianoRoll] = []

        file_list = list(glob.glob(os.path.join(data_dir,"*.json")))
        file_list = file_list[:max_pieces]
        for file_path in file_list:
            new_pr = PianoRoll.load(file_path)
            if metadata is not None:
                song_id = int(file_path.split('/')[-1].split('.json')[0])
                meta = metadata[metadata['id'] == song_id].iloc[0]
                new_pr.set_metadata(name = meta['title'])

            self.pianorolls.append(new_pr)
        
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

        print(f'Created dataset with {self.length} data points from {len(self.pianorolls)} pieces')

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

@dataclass
class PRMetadata:
    name:str = ""
    start_time:int = 0
    end_time:int = 0

class PianoRoll:
    '''
    
    '''
    @ staticmethod
    def load(path):
        '''
        Load a pianoroll from a json file
        '''
        pr = PianoRoll(io_util.json_load(path))
        pr.set_metadata(name = path.split('/')[-1].split('.json')[0])
        return pr

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
        pr = PianoRoll(data)
        pr.set_metadata(name = path.split('/')[-1].split('.mid')[0])
        return pr

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
            if self.notes[-1].offset is None:
                self.duration = ceil((self.notes[-1].onset + 1)/32)*32
            else:
                self.duration = ceil((self.notes[-1].offset)/32)*32
        else:
            self.duration = 0

        self._have_offset = len(self.notes) == 0 or self.notes[0].offset is not None

        self.metadata = PRMetadata("",0,self.duration)

    def __repr__(self) -> str:
        return f'PianoRoll Bar {self.metadata.start_time//32:03d} - {ceil(self.metadata.end_time/32):03d} of {self.metadata.name}'
    
    '''
    ==================
    Utils
    ==================
    '''
    def set_metadata(self, name = None, start_time = None, end_time = None):
        if name is not None:
            self.metadata.name = name
        if start_time is not None:
            self.metadata.start_time = start_time
        if end_time is not None:
            self.metadata.end_time = end_time

    def iter_over_notes(self,notes = None):
        '''
        generator that yields (onset, pitch, velocity, offset iterator)
        '''
        if notes is None:
            notes = self.notes
        for note in notes:
            yield note.onset, note.pitch, note.velocity, note.offset
    
    def get_offsets_with_pedal(self,pedal) -> list[int]:
        offsets = []
        next_onset = [inf]*88
        i = len(pedal)
        for onset, pitch, vel, _ in reversed(list(self.iter_over_notes())): #TODO: handle offsets if there are ones
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

    def to_tensor(self, start_time : int = 0, end_time : int = inf, padding = False, normalized = False, chromagram = False) -> torch.Tensor:
        '''
        Convert the pianoroll to a tensor
        '''
        n_features = 88 if not chromagram else 12

        if padding:
            # zero pad to end_time
            assert end_time != inf
            length = end_time - start_time
        else:
            length = min(self.duration, end_time) - start_time

        size = [length, n_features]
        piano_roll = torch.zeros(size)

        for time, pitch, vel, _ in self.iter_over_notes():
            rel_time = time - start_time
            # only contain notes between start_time and end_time
            if rel_time < 0: continue
            if rel_time >= length : break
            pitch -= 21 # midi to piano
            if chromagram:
                pitch = (pitch + 9)%12
            piano_roll[rel_time,pitch] = vel

        if normalized:
            piano_roll = piano_roll/64-1
        return piano_roll
    
    def to_midi(self,path = None,apply_pedal = True) -> miditoolkit.midi.parser.MidiFile:
        '''
        Convert the pianoroll to a midi file
        '''
        notes = deepcopy(self.notes)
        if apply_pedal:
            if self.pedal:
                pedal = self.pedal
            else:
                pedal = list(range(0,self.duration,32))
            offsets = self.get_offsets_with_pedal(pedal)
            for i, note in enumerate(notes):
                note.offset = offsets[i]
        else:
            assert self._have_offset, "Offset not found"
        return self._save_to_midi([notes],path)


    def _save_to_midi(self,instrs,path):
        midi = miditoolkit.midi.parser.MidiFile()
        midi.instruments = [miditoolkit.Instrument(program=0, is_drum=False, name=f'Piano{i}') for i in range(len(instrs))]
        midi.tempo_changes.append(miditoolkit.TempoChange(144,0))
        for i,notes in enumerate(instrs):
            for onset, pitch, vel, offset in self.iter_over_notes(notes):
                assert offset is not None, "Offset not found"
                midi.instruments[i].notes.append(
                    miditoolkit.Note(vel, pitch, int(onset*midi.ticks_per_beat/8), int(offset*midi.ticks_per_beat/8))
                )

        if path:
            midi.dump(path)
        return midi

    def save_to_pretty_score(self,path,separate_point = 60,position_weight = 3)->list[Note]:
        notes = deepcopy(self.notes)
        # separate left and right hand
        left_hand:list[Note]  = []
        right_hand:list[Note]  = []
        def loss(note,prev_notes,which_hand,max_dist = 16,separate_point = 60,position_weight = 3):
            res = 0
            
            for prev_note in reversed(prev_notes):
                dt = note.onset - prev_note.onset
                dp = note.pitch - prev_note.pitch
                if dt > max_dist:
                    break
                loss = max(0,abs(dp)-5-8*dt)
                res += loss
                
            if which_hand == "l":
                res += (note.pitch-separate_point)*position_weight
            elif which_hand == "r":
                res -= (note.pitch-separate_point)*position_weight
            else:
                raise ValueError("which_hand must be 'l' or 'r'")
            return res
        
        # recursively search for min loss
        def cummulative_loss(past_notes_l,past_notes_r,future_notes,max_depth = 4, discount_factor = 0.9):
            future_notes = future_notes[:max_depth]
            if len(future_notes) == 0:
                return 0,'l'
            else:
                future_loss_l = cummulative_loss(past_notes_l + [future_notes[0]],past_notes_r,future_notes[1:])[0]
                future_loss_r = cummulative_loss(past_notes_l,past_notes_r + [future_notes[0]],future_notes[1:])[0]
                loss_l = future_loss_l*discount_factor + loss(future_notes[0],past_notes_l,"l",16,separate_point,position_weight)
                loss_r = future_loss_r*discount_factor + loss(future_notes[0],past_notes_r,"r",16,separate_point,position_weight)
                if loss_l < loss_r:
                    return loss_l,'l'
                else:
                    return loss_r,'r'
        
        while len(notes):
            _,hand = cummulative_loss(left_hand,right_hand,notes)
            if hand == 'l':
                left_hand.append(notes.pop(0))
            else:
                right_hand.append(notes.pop(0))

            
        def pretty_voice(voice: list[Note]):
            current = []
            for note in voice:
                if len(current) == 0:
                    current.append(note)
                else:
                    if note.onset == current[-1].onset:
                        current.append(note)
                    else:
                        stop_time = note.onset
                        for c in current:
                            c.offset = stop_time
                        current = [note]
        
        pretty_voice(left_hand)
        pretty_voice(right_hand)
        res = [right_hand,left_hand]
        print("left hand notes:",len(left_hand))
        print("right hand notes:",len(right_hand))
        self._save_to_midi(res,path)

        
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
        for time, pitch, vel, offset in self.iter_over_notes():
            rel_time = time - start_time
            if offset is not None:
                rel_offset = offset - start_time
            else:
                rel_offset = None
            # only contain notes between start_time and end_time
            if rel_time < 0: continue
            if rel_time >= length : break
            sliced_notes.append([rel_time,pitch,vel,rel_offset])

        if self.pedal:
            for pedal in self.pedal:
                time = pedal
                rel_time = time - start_time
                # only contain pedal between start_time and end_time
                if rel_time < 0: continue
                if rel_time >= length : break
                sliced_pedal.append(rel_time)
            new_pr = PianoRoll({"onset_events":sliced_notes,"pedal_events":sliced_pedal})
        else:
            new_pr = PianoRoll({"onset_events":sliced_notes})

        new_pr.set_metadata(self.metadata.name,self.metadata.start_time + start_time,self.metadata.start_time + end_time)
        return new_pr

    def random_slice(self, length : int = 128) -> PianoRoll:
        '''
        Randomly slice a pianoroll with length
        '''
        start_time = random.randint(0,(self.duration - length)//32)*32
        return self.slice(start_time,start_time+length)

    def get_random_tensor_clip(self,duration,normalized = False):
        '''
        Get a random clip of the pianoroll
        '''
        start_time = random.randint(0,(self.duration-duration)//32)*32 # snap to bar
        return self.to_tensor(start_time, start_time+duration, normalized = normalized)

if __name__ == '__main__':  
    # test
    d = PianoRollDataset('/home/eri24816/pianoroll/', 6*32,max_pieces=3)
    orig = d[100]
    file_name = 'legacy/pianoroll_test.mid'
    PianoRoll.from_tensor(orig,0,normalized=True).to_midi(file_name)
    recons = PianoRoll.from_midi(file_name).to_tensor(normalized=True)
    assert ((orig- recons[:6*32])**2).sum()==0 