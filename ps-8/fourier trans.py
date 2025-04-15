from numpy.fft import rfft, irfft
from pylab import *
import numpy as np

piano = np.loadtxt('piano.txt')
trumpet = np.loadtxt('trumpet.txt')

sample_rate = 44100

time = np.arange(0, len(piano)/sample_rate, 1/sample_rate, dtype=np.float64)

# plot(time, piano, label='piano recording')
# plt.xlabel('time/second')
# plt.ylabel('Piano Signal')
# plt.savefig('piano signal.png')
# # plt.show()
#
# plot(time, trumpet, label='trumpet recording')
# plt.xlabel('time/second')
# plt.ylabel('trumpet Signal')
# plt.savefig('trumpet signal.png')
# plt.show()

#FFT
pianofft = rfft(piano)
trumpetfft = rfft(trumpet)
freq = np.arange(0, 30000, sample_rate/len(piano))[:30000]

# plot(freq[:10000], np.abs(pianofft[:10000]), label='piano fft')
# plt.xlabel('frequency/ Hz')
# plt.ylabel('fft amplitude')
# plt.savefig('piano fft.png')
# plt.show()

# plot(freq, np.abs(trumpetfft[:30000]), label='trumpet fft')
# plt.xlabel('frequency/ Hz')
# plt.ylabel('fft amplitude')
# plt.savefig('trumpet fft.png')
# plt.show()

print('the maximum signal frequency of piano is at: '+ str(sample_rate*np.argmax(np.abs(pianofft))/len(piano))+ "Hz")
print('the maximum signal frequency is trumpet at: '+ str(sample_rate*np.argmax(np.abs(trumpetfft))/len(trumpet))+ "Hz")