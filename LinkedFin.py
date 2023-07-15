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
             background-image: URL("data:image/jpeg;base64,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")
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
  
  
  
  


    
  
    
      
  
  
