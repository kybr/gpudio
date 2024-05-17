import taichi as ti
import pyaudio

ti.init(arch=ti.gpu)

block = 16384
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

stream = p.open(format=p.get_format_from_width(4),
                channels=1,
                rate=samplerate,
                output=True)

i = 0
while gui.running:
    paint(i)
    gui.set_image(data)
    gui.show()
    stream.write(data.to_numpy().sum(axis=0))
    i += block

stream.close()
p.terminate()

