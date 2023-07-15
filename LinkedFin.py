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
from collections import Counter
from analyser import*
import joblib
import base64

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpeg;base64,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");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
#https://media.tenor.com/jIx6LIssomUAAAAC/vinyl-anime.gif

st.header("Welcome to Sound Sage V-1.0.0")


st.markdown(f"""
<div class="intro">
<p> Welcome to SoundSage. Your guide to composing your own music. 
SoundSage will analyse the songs you give it and find out what makes those songs sound unique and help you compose similar-sounding music. 
</p>
</div>

<h3> How to Use SoundSage</h3>
<ol class="instr">
<li>First start off by uploading MP3 files of the songs you choose. A link is also provided to a website you can use to download mp3 files
from youtube videos</li>
<li>Click on the 'Start Processing' button and your songs will start to be analysed</li>
<li>As your songs are analysed the information will keep being displayed</li>
<li> <b> How to interpret your results to compose your own music</b>
         <ol>
         <li>If you want to compose similar-sounding music try using a tempo close to the average of your songs</li>
         <li>Try using the musical keys with the highest correlation factor given</li>
         <li>For your chord progression try using the common patterns of chords from within the songs. Also, try to use different types of chords
         such as minor, major the, etc.</li>
         <li>While writing your main melody try using the notes which are most commonly used in your songs, or try using the patterns found 
         found most commonly in your song</li>
         </ol>
</li>
</ol>
</div>

<div id="start"> Lets Begin!</div>

""" , unsafe_allow_html=True)






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
         output={}
         output.update({str(max(self.key_dict, key=self.key_dict.get)) : str(self.bestcorr)})
         if self.altkey is not None:
                output.update({str(self.altkey):str(self.altbestcorr)})
         
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
        
def BPMcomp(data):

  print ("FULL DATASTRUCTURE")
  print(data)
  
  

  BPMsum = 0

  bpm_list = []  # Empty list to store the BPM values

  for item in data:
    bpm = item['BPM']
    bpm_list.append(bpm)

  print(bpm_list)



  #BPMavg = BPMsum / len(BPMlist)
  #st.write("The average BPM of all your pieces is: " + str(BPMavg))

  largo = 0
  adagio = 0
  andante = 0
  allegro = 0
  presto = 0

  for item in bpm_list:
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


def analyze_chords(data):
    chord_array=[]
    for item in data:
      chords=item['CHORDS']
      chord_array.append(chords)
    
    chord_patterns = {}
    chord_types = {}
    top_chords=[]

   

    for i in range(len(chord_array) - 2):
        if chord_array[i] != 'N' and chord_array[i+1] != 'N' and chord_array[i+2] != 'N':
            chord_pattern = chord_array[i] + ' - ' + chord_array[i+1] + ' - ' + chord_array[i+2]
            if chord_pattern in chord_patterns:
                chord_patterns[chord_pattern] += 1
            else:
                chord_patterns[chord_pattern] = 1

    chord_patterns = {pattern: count for pattern, count in chord_patterns.items() if count >= 2}
    chord_patterns = dict(sorted(chord_patterns.items(), key=lambda x: x[1], reverse=True))

    for chord in chord_array:
        if chord != 'N':
            if len(chord) == 1:
                chord_type = 'maj'
            else:
                chord_type = chord[1:]  # Exclude the root note

            if 'b' in chord_type or '/' in chord_type:
                continue

            if chord_type in chord_types:
                chord_types[chord_type] += 1
            else:
                chord_types[chord_type] = 1


    # Print chord patterns
    print("Chord Patterns:")
    for chord_pattern, count in chord_patterns.items():
        print(chord_pattern + ': ' + str(count))

    # Print chord types
    print("\nChord Types:")
    for chord_type, count in chord_types.items():
        print(chord_type + ': ' + str(count))

   
        

 

    # Calculate the chord frequencies
    chord_frequencies = {}
    for chord in chord_array:
        if chord != 'N':
            if len(chord) == 1:
                chord_type = 'maj'
            else:
                chord_type = chord[1:]  # Exclude the root note

            if 'b' in chord_type or '/' in chord_type:
                continue

            if chord in chord_frequencies:
                chord_frequencies[chord] += 1
            else:
                chord_frequencies[chord] = 1

    # Sort the chord frequencies in descending order
    sorted_chords = sorted(chord_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Get the top 3 most used chords
    for i in range(min(3, len(sorted_chords))):
        chord, frequency = sorted_chords[i]
        top_chords.append(chord)

    # Print top 3 most used chords
    print("Top 3 Most Used Chords:")
    for chord in top_chords:
        print(chord + ': ' + str(chord_frequencies[chord]))

    # Rest of the code...

    return chord_patterns, chord_types

# Rest of the code...


    return chord_patterns, chord_types, relative_movements
  



if __name__ == "__main__":
  
  st.markdown(f"""
  <div class="input">Enter the Number of Songs </div>
  """ ,  unsafe_allow_html=True)
  n= int(st.number_input("",1,10,1))
  lis= []
  

  song_list=[]

  st.markdown(f"""
  <div class="input">Submit Your song MP3 </div>
  """ ,  unsafe_allow_html=True)
  for i in range(0,n):
      
    x=st.file_uploader("", key=i)
    song_list.append(x)
  st.write("Song List")
  if st.button("Start Processing !"):
  
    for k in range(0,n):
      d1 = {}

      file_var = AudioSegment.from_file(song_list[k]) 
      file_var.export('song.mp3', format='mp3')
    
    
      audio_path = clip("song.mp3")
      y, sr = librosa.load(audio_path)
      y_harmonic, y_percussive = librosa.effects.hpss(y)

     

      unebarque_e_min = Tonal_Fragment(y_harmonic, sr, tstart=22, tend=33)
      song_key=unebarque_e_min.print_key()
    
      d1.update({'KEY': song_key})
  
  

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
      d1.update({'NOTES':notesbuffer})
    
      #st.write(*list_res, sep=" || ")
      output_file = "result.wav"
      sound = AudioSegment.from_mp3(audio_path)
      sound.export(output_file, format="wav")
      samps, fs = read_wav(output_file)
      bpm_value = str(int(bpm_detector(samps, fs)))
      
      #st.write("BPM OF SONG: "+ bpm_value))
      st.write("Song "+ str(k+1)+" Processed")
      d1.update({"BPM":bpm_value})
    
      lis.append(d1)
    
    
      
  print(lis)
  time.sleep(1)
  
  st.markdown(f"""
  <div class="analysis">Analysis of BPM </div>
  """ ,  unsafe_allow_html=True)
  analyse_BPM(lis)
  time.sleep(10)
  
  tbr_chord_list=[]
  for item in lis:
   chords= item['CHORDS']
   for item in chords:
     tbr_chord_list.append(chords)
  
  print("THE NEW CHORD LIST")
  print(tbr_chord_list)
 
  st.markdown(f"""
  <div class="analysis">Analysis of Chords </div>
  """ ,  unsafe_allow_html=True)
  analyse_chords(lis)
  
  st.markdown(f"""
  <div class="analysis">Analysis of Musical Key</div>
  """ ,  unsafe_allow_html=True)
  time.sleep(5)
  analyse_key(lis)
  
  st.markdown(f"""
  <div class="analysis">Analysis of Notes </div>
  """ ,  unsafe_allow_html=True)
  time.sleep(5)
  analyse_notes(lis)
  st.balloons()
  
  
  
  


    
  
    
      
  
  
