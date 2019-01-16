import soundcard as sc
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

NOTES = pd.read_csv('./notes.csv').reset_index()


def play_metronome():
    BPM = 100 #beats per minute
    BPS = BPM/60 #betas per second
    SBB = 1/BPS #seconds between beats
    cmd = "echo -n '\a';sleep "+str(SBB)+";"
    cmd *= 4
    beep = os.system(cmd)
    cmd = "echo -n '\a';sleep "+str(SBB)+";"
    cmd *= 2
    beep = os.system(cmd)  

def record(mic, seconds=6, sample_rate=48000):
    """
    function that records using mic 
    
    returns
    """
    # record and play back one second of audio:
    num_frames = sample_rate*seconds
    data = mic.record(samplerate=sample_rate, numframes=num_frames)
    audio = pd.DataFrame(dict(index = list(range(0, data.shape[0])),
                left = data[:,0],
                right = data[:,1]), dtype=np.float128)
    return(audio)

def play(speaker, audio, sample_rate=48000):
    x = audio[['left', 'right']].values
    x_normed = x / x.max(axis=0)
    speaker.play(x_normed, samplerate=sample_rate)
    
def play_ch(speaker, audio, sample_rate=48000):
    speaker.play(audio/audio.max(), samplerate=sample_rate)

def get_fft(audio_channel, seconds=6, sample_rate=48000, cut_off_freq=22000):
    audio_channel = audio_channel.astype(float).values
    N = sample_rate * seconds

    # Use FFT to get the amplitude of the spectrum
    ampl = 1/N * abs(np.fft.fft(audio_channel))
    ampl = ampl[1:cut_off_freq]

    return(ampl)

def get_note(y_fft, notes, offset=2):
    """
    Function that identifies the note played
    
    offset is needed for calibration
    """
    return(pd.DataFrame(notes.iloc[notes.\
                            Frequency_Hz.\
                            apply(lambda x: \
                                  abs(x-y_fft.argmax())).\
                            values.\
                            argmin()+offset]).\
    reset_index().\
          rename(columns={'index':'key', 2:'value'}))

def save_plots(audio, y_fft,
               file_name='file_name', 
               annotate=False, 
               file_type='jpg',
               figsize=(15,7),
               fft=True,
               right=False,
               left=False):
    """
    function that saves 3 plots
        audio.left
        audio.right
        fft
    as 
        <file_name>_left.<file_type>
        <file_name>_right.<file_type>
        <file_name>_fft.<file_type>   
    """
    max_freq = y_fft.argmax()
    

    if right:
        audio_right = plt.figure(figsize=figsize)
        plt.plot(audio.right)
        audio_right.savefig(file_name+'_right.'+file_type)
        plt.show()

    if left:
        audio_left = plt.figure(figsize=figsize)
        plt.plot(audio.left)
        audio_left.savefig(file_name+'_left.'+file_type)
        plt.show()

    if fft:
        fft_plot = plt.figure(figsize=figsize)
        plt.plot(y_fft)
        if annotate:
            plt.text(max_freq, y_fft[max_freq], ' freq: %.2f Hz'%(max_freq))
            for row in get_note(y_fft, NOTES).iterrows():
                     plt.text(max_freq, \
                              y_fft[max_freq] \
                         * (0.9\
                          - row[0]\
                          / 10), ' %s: %s'%(str(row[1][0]), str(row[1][1])))
        fft_plot.savefig(file_name+'_fft.'+file_type)
        plt.show()
    
    #plt.xscale('log')
    #plt.xlim((100, CUT_OFF))
