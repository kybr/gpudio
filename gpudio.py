import taichi as ti
import pyaudio
import time

ti.init(arch=ti.gpu)

block = 4096
samplerate = 48000
tau = 6.28318530717958647692
f0 = 220
shape = (128, block)
data = ti.field(dtype=float, shape=shape)

@ti.kernel
def paint(t: float):
    for harmonic, sample in data:
        n = harmonic + 1
        if n * f0 < samplerate // 2:
            data[harmonic, sample] = ti.math.sin(tau * n * f0 * (t + sample) / samplerate) / n

gui = ti.GUI("GPU Audio", res=shape)
p = pyaudio.PyAudio()

i = 0
def callback(in_data, frame_count, time_info, status):
    global i
    paint(i)
    ti.sync()
    audio = data.to_numpy().sum(axis=0)
    i += block
    return (audio, pyaudio.paContinue)

stream = p.open(format=p.get_format_from_width(4),
                channels=1,
                rate=samplerate,
                output=True,
                stream_callback=callback)

while gui.running:
    pass

stream.close()
p.terminate()
