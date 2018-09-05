# Import stuff!
import tables
import os
import numpy as np

# Create EArrays in hdf5 file 
def create_hdf_arrays(file_name, ports, dig_in, emg_port, emg_channels):
	hf5 = tables.open_file(file_name, 'r+')
	n_electrodes = len(ports)*32
	atom = tables.IntAtom()
	
	# Create arrays for digital inputs
	for i in dig_in:
		dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))

	# Create arrays for neural electrodes, and make directories to store stuff coming out from blech_process
	for i in range(n_electrodes - len(emg_channels)):
		el = hf5.create_earray('/raw', 'electrode%i' % i, atom, (0,))
		
	# Create arrays for EMG electrodes
	for i in range(len(emg_channels)):
		el = hf5.create_earray('/raw_emg', 'emg%i' % i, atom, (0,))

	# Close the hdf5 file 
	hf5.close()	

# Read files into hdf5 arrays - the format should be 'one file per channel'
def read_files(hdf5_name, ports, dig_in, emg_port, emg_channels):
	hf5 = tables.open_file(hdf5_name, 'r+')

	# Read digital inputs, and append to the respective hdf5 arrays
	for i in dig_in:
		inputs = np.fromfile('board-DIN-%02d'%i + '.dat', dtype = np.dtype('uint16'))
		exec("hf5.root.digital_in.dig_in_"+str(i)+".append(inputs[:])")

	# Read data from amplifier channels
	emg_counter = 0
	el_counter = 0
	for port in ports:
		for channel in range(32):
			data = np.fromfile('amp-' + port + '-%03d'%channel + '.dat', dtype = np.dtype('int16'))
			if port == emg_port[0] and channel in emg_channels:
				exec("hf5.root.raw_emg.emg%i.append(data[:])" % emg_counter)
				emg_counter += 1
			else:
				exec("hf5.root.raw.electrode%i.append(data[:])" % el_counter)
				el_counter += 1
		hf5.flush()

	hf5.close()
				
	

	
