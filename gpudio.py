import taichi as ti
import taichi.math as tm
import pyaudio

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.func
def complex_sqr(z):  # complex square of a 2D vector
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))

def callback(in_data, frame_count, time_info, status):
    # If len(data) is less than requested frame_count, PyAudio automatically
    # assumes the stream is finished, and the stream stops.
    return (in_data, pyaudio.paContinue)

# Instantiate PyAudio and initialize PortAudio system resources (2)
p = pyaudio.PyAudio()

# Open stream using callback (3)
stream = p.open(format=p.get_format_from_width(4),
                channels=1,
                rate=48000,
                output=True,
                stream_callback=callback)

i = 0
while gui.running:
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
    i += 1

stream.close()
p.terminate()

