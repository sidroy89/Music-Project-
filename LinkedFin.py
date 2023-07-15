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
             background-image: URL(""data:image/jpeg;base64,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)
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
  
  
  
  


    
  
    
      
  
  
