from pytube import YouTube
import os
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
from pydub import AudioSegment
from YTExctract import *
from ClipperWorking import *
from chord_extractor.extractors import Chordino
from NoteExtractor import *
import streamlit as st
import time 
from tempo import *
from os import path


# class that uses the librosa library to analyze the key that an mp3 is in
# arguments:
#     waveform: an mp3 file loaded by librosa, ideally separated out from any percussive sources
#     sr: sampling rate of the mp3, which can be obtained when the file is read with librosa
#     tstart and tend: the range in seconds of the file to be analyzed; default to the beginning and end of file if not specified
class Tonal_Fragment(object):
    def __init__(self, waveform, sr, tstart=None, tend=None):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend
        
        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)
        
        # chroma_vals is the amount of each pitch class present in this time interval
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        # dictionary relating pitch names to the associated intensity in the song
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)} 
        
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m)%12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1,0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1,0], 3))

        # names of all major and minor keys
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)}, 
                         **{keys[i+12]: self.min_key_corrs[i] for i in range(12)}}
        
        # this attribute represents the key determined by the algorithm
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())
        
        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr*0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr
                
    # st.writes the relative prominence of each pitch class            
    
                
    # st.writes the correlation coefficients associated with each major/minor key
    
    
    # st.writeout of the key determined by the algorithm; if another key is close, that key is mentioned
    def print_key(self):
         output = "likely key: " + str(max(self.key_dict, key=self.key_dict.get)) + ", correlation: " + str(self.bestcorr)
         if self.altkey is not None:
                output += "\nalso possible: " + str(self.altkey) + ", correlation: " + str(self.altbestcorr)
         
         return output
    
    # st.writes a chromagram of the file, showing the intensity of each pitch class over time
    def chromagram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=sr, bins_per_octave=24)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        if title is None:
            plt.title('Chromagram')
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
def BPMcomp(lis):
  BPMlist = []
  BPMsum = 0

  for i in range(0, len(lis)):
    BPMlist.append(lis[i].get("BPM"))
    BPMsum += int(lis[i].get("BPM"))

  BPMavg = BPMsum / len(lis)
  st.write("The average BPM of all your pieces is: " + str(BPMavg))

  largo = 0
  adagio = 0
  andante = 0
  allegro = 0
  presto = 0

  for item in BPMlist:
    if int(item) <= 60:
        largo += 1
    elif int(item) <= 76:
        adagio += 1
    elif int(item) <= 110:
        andante += 1
    elif int(item) <= 180:
        allegro += 1
    else:
        presto += 1

  BPMclassifications = [largo, adagio, andante, allegro, presto]
  sorted_BPMclassifications = sorted(BPMclassifications, reverse=True)
  
  max_bpm_classification = max(sorted_BPMclassifications)
  
  if max_bpm_classification == largo:
      st.write("The majority of your pieces are classified as 'Largo'. This means a majority of the pieces have a very slow and broad pace.")
  elif max_bpm_classification == adagio:
      st.write("The majority of your pieces are classified as 'Adagio'. This means a majority of the pieces have a leisurely relaxed pace.")
  elif max_bpm_classification == andante:
      st.write("The majority of your pieces are classified as 'Andante'. This means a majority of the pieces have a moderate pace.")
  elif max_bpm_classification == allegro:
      st.write("The majority of your pieces are classified as 'Allegro'. This means a majority of the pieces have a fast and energetic pace.")
  else:
      st.write("The majority of your pieces are classified as 'Presto'. This means a majority of the pieces have a rapid and urgent pace.")


    
  
   

if __name__ == "__main__":
  
  n= int(st.number_input("Enter Number of Songs",1,10,1))
  lis= []
  for i in range(0,n):
    d1 = {}
    try:
      x=st.text_input('Enter the Song URL', key=i)
    except RegexMatchError:
      st.error("Enter a youtube link")
    audio_path = clip(YTDL(x))
    y, sr = librosa.load(audio_path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    unebarque = Tonal_Fragment(y_harmonic, sr)
    unebarque.chromagram("Une Barque sur l\'Ocean")
    unebarque_fsharp_min = Tonal_Fragment(y_harmonic, sr, tend=22)
    unebarque_fsharp_maj = Tonal_Fragment(y_harmonic, sr, tend=22)
    song_key1=unebarque_fsharp_maj.print_key()

    unebarque_e_min = Tonal_Fragment(y_harmonic, sr, tstart=22, tend=33)
    song_key2=unebarque_e_min.print_key()
    
    d1.update({'KEY': song_key1 + " "+song_key2})
  
  

    chordino = Chordino(roll_on=1)  
    chord_list = []


    chords = chordino.extract(audio_path)
    for i in range(0,len(chords)):
      chord_list.append(list(chords[i])[0])
      
    set_c= set(chord_list)
    clistfin = list(set_c)
    


    d1.update({'CHORDS':chord_list})
    notesbuffer= list(NExt(audio_path))
    note_list=[]
    for i in range(0,len(NExt(audio_path))-9):
      note_list.append(notesbuffer[i])
    set_res = set(note_list) 
    

    list_res = (list(set_res))
    d1.update({'NOTES':list_res})
    
    #st.write(*list_res, sep=" || ")
    output_file = "result.wav"
    sound = AudioSegment.from_mp3(audio_path)
    sound.export(output_file, format="wav")
    samps, fs = read_wav(output_file)
    bpm_value = str(int(bpm_detector(samps, fs)))
      
    #st.write("BPM OF SONG: "+ bpm_value))
    st.write("Song Processed")
    d1.update({"BPM":bpm_value})
    
    lis.append(d1)
    st.write()
    st.write()
    
    
    if st.button("display" , key=i):
      st.write("NOTES")
      st.write(*list_res, sep=" || ")
      st.write()
      
      st.write("CHORDS")
      st.write(*clistfin , sep="")
      st.write()
      
      st.write("key")
      st.write(song_key1 + " "+song_key2)
      
      st.write("BPM is: "+bpm_value)
      
      time.sleep(15)
  
  if st.button("Click to Compare Tempo"):
    BPMcomp(lis)

    
      
  
  
