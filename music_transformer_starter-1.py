# Music Transformer assignment
# (C) 2023-2025 Travis Mandel
import torch

import numpy as np
import math
import sys
import random

import matplotlib.pyplot as plt
import music21
import os

# Global constant for the maximum sequence length that the 
#  Transformer will process. 
maxSeqLen=20



class MusicDataset(torch.utils.data.Dataset):
    # Inits the dataset. train=True for training, false for testing
    def __init__(self, train=True):
    
    
        corpus = self.getCorpusFromFolder("chopin/")
        self.buildMapsFromCorpus(corpus)
        
        #Split into train vs test
        threshold = int(len(corpus)*0.8)
        if train:
            self.corpus = corpus[:threshold]
        else:
           self.corpus = corpus[threshold:]
        self.buildIndexToSongAndChordMap()
        
        self.startToken = None #Change this later if you need an explicit start token
           
        
        
        
        
    # Loads in a corpus of songs from a folder full of 
    # MIDI file
    # Takes in: Folder full of MIDI files
    # Returns: A corpus (list of songs, each song list of chords (or notes) )
    def getCorpusFromFolder(self, folder):
        corpus = []
        for fname in os.listdir(folder):
            if fname.endswith(".mid"):
                midi = music21.converter.parse(folder+fname)
                songs = music21.instrument.partitionByInstrument(midi)
                for part in songs.parts:
                    notes = []
                    pick = part.recurse()
                    for element in pick:
                        # The block below is admittedly not good style but I took this function from a music LSTM repo
                        if isinstance(element, music21.note.Note):
                            notes.append(str(element.pitch))
                        elif isinstance(element, music21.chord.Chord):
                            notes.append(".".join(str(n) for n in element.normalOrder))
                    corpus.append(notes)
        return corpus
                

            
    # This function should fill in two dictionaries.
    # self.chordToId and self.idToChord
    # Takes in: A corpus of songs
    # Returns: Nothing (just fills in self. variables)
    def buildMapsFromCorpus(self, corpus):
        #TODO: fill in 
        pass
    
    
    # Helper function: convert a list of chords to a list of ids
    # Takes in: a list of chords
    # Returns: a list of ids 
    def convertChordsToIds(self, sequence):
        newSequence = []
        for chord in sequence:
            newSequence.append(self.chordToId[chord])
        return newSequence


    # Build a dict mapping each index to a song and chord.
    # Do this so that we don't have to calculate all this 
    # on-the-fly each time we have a new song.
    # Make sure this goes over all valid positions in the song.
    # Takes in and returns nothing, just a helper function.
    def buildIndexToSongAndChordMap(self):
    
        count = 0
        self.indexMap = {}
        for i in range(len(self.corpus)):
           song = self.corpus[i]
           for j in range(0, len(song)-maxSeqLen-1, maxSeqLen):
                self.indexMap[count] = (i,j)
                count+=1
      
        
                
    # This returns the size of the vocabulary - how many unique
    # tokens/chords are there?
    def getNumTokens(self):
        #TODO: Fill in.
        pass        
    
    # Helper function that takes in a list of chord Ids and 
    # writes them to a WAV file (named wavName) so you can listen to them
    def saveSongAsWAV(self, chordList, wavName):
        music = []
        offset = 0 
        for chordId in chordList:
            if chordId == self.startToken:
                continue
            chord = self.idToChord[chordId]
            if "." in chord or chord.isdigit():
                chord_notes = chord.split(".") 
                notes = [] 
                for note in chord_notes:
                    note_snip = music21.note.Note(int(note))            
                    notes.append(note_snip)
                chordComb = music21.chord.Chord(notes)
                chordComb.offset = offset
                music.append(chordComb)
            else: # "chord" is just a single note
                note = music21.note.Note(chord)
                note.offset = offset
                music.append(note)
            offset += 1
        musicStream = music21.stream.Stream(music)
        musicStream.write('midi','musicGenerated.mid')
        os.system("timidity musicGenerated.mid -Ow -o "+wavName+".wav")
        os.system("rm musicGenerated.mid")

    # Returns the length of the dataset
    def __len__(self):
        return len(self.corpus)

    # Returns the chordList as a tensor of chords
    def chordListToTensor(self, chordList):
        tensor = torch.from_numpy(np.array(chordList)).to(torch.int64)
        return tensor

    #returns the item at index idx of the dataset
    def __getitem__(self, idx):
    
        #TODO: FIll in
       
        pass



class MusicTransformer(torch.nn.Module):

    # Takes in:
    #  ntoken - how many possible items/tokens in the  vocabulary
    #  d_model - number of features in the each input element
    #  nhead - number of heads
    #  d_hid - number of units on the hidden layer of the feedforward network inside the encodinger
    #  nlayers - number of encoder layers
    # dropout - how much dropout is appl
    # max_len - max length of a sequence
    def __init__(self, device, ntoken, d_model=200, nhead=2, d_hid=200,
                 nlayers=2, max_len=maxSeqLen):
        super().__init__()
        
        
        
        # This block of (bad style) code is copied from the official 
        #  Transformers pytorch tutorial and sets up the 
        # positional encoders (pe). You don't need to 
        # understand this part too deeply.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.to(device)

        
        #Sets up the transformer "encoder":
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, 0.0)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        print(self.transformer_encoder)
        
        #Sets up the transformer Embedding
        self.embedding = torch.nn.Embedding(ntoken, d_model)
        self.embedding.to(device)
        
        self.d_model = d_model
        
        #TODO: Fill in the rest
        # self.layers = 
        # self.optimizer = 
        # self.loss_fn = 
        # self.device =
        
 

        self.init_weights()

    #init the weights of the network before training
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #TODO: include the following lines with the appropriate variable names to initialize the weigths
        #<linear layer>.bias.data.zero_()
        #<linear layer>.weight.data.uniform_(-initrange, initrange)
    
        
    def forward(self, input) :

        # Embed the input
        src = self.embedding(input) * math.sqrt(self.d_model)
        # Add positional info
        src = src + self.pe[:src.size(0)]
        # Encode using the transformer (should not need the mask)
        output = self.transformer_encoder(src, None)
        #TODO: pass through final encoder layers
        return output

    
    def doTrain(self, dataloader):
        self.train()

        for (inputs, outputs) in dataloader:
            
            inputs = inputs.to(self.device)
            outputs = outputs.to(self.device)
            
            
            # Transpose the dimensions of the input
            # The transformer expects the sequence length of come first
            inputs = inputs.t().contiguous()
            
            #TODO: Fill in the rest...
        
    def predict(self, inpt):

        #Only 1 instance at a time
        numInputs = inpt.shape[0]
        assert (numInputs == 1)
        inputs = inpt.t().contiguous()
        
        self.eval()
        pred = self.forward(inputs)

        #TODO: Turn the output into a list of probabilities
        
        return pred


def genMusic(model, train_dataset, device, train_dataloader):
    #TODO: Fill this in (instructions in assignment)
    pass
        
            
            

  
        
    
train_data = MusicDataset(train=True)
test_data =  MusicDataset(train=False)
#print(train_data)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=100)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1)


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    
print("Using device", device)

model = MusicTransformer(device, train_data.getNumTokens()).to(device)
print(model)

epochs = 100
for e in range(epochs):
    model.doTrain(train_dataloader)
    
genMusic(model, train_data, device, train_dataloader)

#TODO: Evaluate on test data as well as train data

    
    
