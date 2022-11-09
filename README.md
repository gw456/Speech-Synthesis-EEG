# Speech Synthesis Based On EEG
In this project we tried to synthesis limited sentences using EEG signal input. This code can be divided into 5 steps, that are data collection, signal filtering, classification with deep learning, arranging classified syllable into sentence and convert text into speech output. In this project, I am responsible for collecting data and making pipeline based on above steps, except the last step.

## Data Collection
Dataset is made by collecting EEG signal of 6 subjects. These subjects are asked to watch a video of syllables, visuallize and spell it (visualizing and spelling is done alternately). We extract EEG signal in F7 and T3 point (with a common reference point placed at bone behind ear) based on 10-20 system. Beside that, we also record the subject's voice to help segmenting EEG signals. EEG signal extraction and voice record are done using BITalino and Dolby mobile apps respectively. The signals are saved in HDF5 file format and the code will convert it into list.

## Signal Filtering
Then the data is passed through a filter program in the form of an IIR bandpass filter. This filter has a pass frequency range of 8-30 Hz; inhibition frequencies 1 Hz and 35 Hz; bandpass ripple of 0.4 dB; and the attenuation of the inhibition frequency is 3 dB. Furthermore, the data is segmented automatically by utilizing the patterned recorded sound. Segmentation is done by cutting one data retrieval into one syllable by one and the method (visualization or speech).

## Classification with Deep Learning
After segmenting, feature extraction is carried out with features in the eeglib library. The features in eeglib used consist of dfa (applying trend fluctuation analysis algorithm), hfd (returning higuchi fractal dimensions), lzc (returning Lempel-Ziv complexity (LZ 76)), pfd (returning petrosian fractal dimensions), hjorth Activity (returning activity Hjorth), hjorth Complexity (returns Hjorth complexity), hjorth Mobility (returns Hjorth mobility), and sampEn (returns sample entropy).

Data is divided into training and testing data, with 80% and 20% ratio respectively. The ANN used to classify those features into syllbale is using Categorical Cross entropy and adam as loss function and optimizer, respectively.

## Arranging Classified Syllable into Sentence
The output of ANN is taken as input of RNN to predict the next syllable. RNN implement bLSTM layer. The output of RNN is an arranged sentence that is taken into the next step.

## Convert Text into Speech Output
This process only implement existing package with input from previous step.

# BITalino (r)evolution Python API
The BITalino (r)evolution Python API provides the needed tools to interact with BITalino (r)evolution using Python.

## Dependencies
* [Python >2.7](https://www.python.org/downloads/) or [Anaconda](https://www.continuum.io/downloads) or [Python 3.4](https://www.python.org/downloads/)
* [NumPy](https://pypi.python.org/pypi/numpy)
* [pySerial](https://pypi.python.org/pypi/pyserial)
* [PyBluez](https://pypi.python.org/pypi/PyBluez/) (Not needed for Mac OS)

## Installation
1. Install Dependencies
* **NumPy**
~~~
pip install numpy
~~~

* **pySerial**
~~~
pip install pyserial
~~~

* **PyBluez** *\[Only on Windows]*

Before installing **PyBluez** some requirements should be fulfilled. For a straightforward installation please check the auxiliary section: [**Prepare PyBluez Installation on Windows 10**](#prepare-pybluez-installation-on-windows-10)
~~~
pip install pybluez2
~~~

2. Install **bitalino** API package
~~~
pip install bitalino
~~~

## Documentation
http://bitalino.com/pyAPI/

## Example
~~~python
import time
from bitalino import BITalino

# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
macAddress = "00:00:00:00:00:00"

# This example will collect data for 5 sec.
running_time = 5
    
batteryThreshold = 30
acqChannels = [0, 1, 2, 3, 4, 5]
samplingRate = 1000
nSamples = 10
digitalOutput = [1,1]

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print(device.version())
    
# Start Acquisition
device.start(samplingRate, acqChannels)

start = time.time()
end = time.time()
while (end - start) < running_time:
    # Read samples
    print(device.read(nSamples))
    end = time.time()

# Turn BITalino led on
device.trigger(digitalOutput)
    
# Stop acquisition
device.stop()
    
# Close connection
device.close()
~~~
## Prepare PyBluez Installation on Windows 10
For **Windows 10** the **PyBluez** installation procedure requires some particular steps that will be presented on the following topics (tested procedure on **\[Python 3.x]**):

1. Download and start the installation of Visual Studio 2015 ([https://go.microsoft.com/fwlink/?LinkId=532606&clcid=0x409](https://go.microsoft.com/fwlink/?LinkId=532606&clcid=0x409))

2. During the installation select the "Custom" option

![Selection of Custom Option](https://i.postimg.cc/vTcMxjpy/git-part1.png)

3. On the new screen, select some additional functionalities required for **PyBluez** installation, namely:
* Visual C++
* Python Tools for Visual Studio
* Windows 10 SDK

![Selection of Tools](https://i.postimg.cc/qqSrswT3/git-part2.png)

4. After ending step 3 you will be able to install **PyBluez**

* **PyBluez-win10**
~~~
pip install PyBluez-win10
~~~

## License
This project is licensed under the [GNU GPL v3](LICENSE.md)
