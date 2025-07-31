import numpy as np
import matplotlib.pyplot as plt
import sys

class ScopePlot:

    def plot_downsampled(filename, factor=100):


        with open (filename, "rb") as f:
                raw = f.read()

        data = np.frombuffer(raw, dtype=np.float32)
        data = data[::factor]

        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title("Downsampled Oscilloscope Data")
        plt.xlabel("Sample #")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('/home/mint-daq/online/custom/plots/scopefigure.png')
        plt.show()



#if __name__ == "__main__":
    #read_and_plot("722data3")
 #   plot_downsampled("722data5",100)