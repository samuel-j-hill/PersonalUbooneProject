import numpy as np
import matplotlib.pyplot as plt
from microboone_utils import *
from skimage.measure import block_reduce
from pynuml.io import File

def new_read_data(file_name, num_events, chunk_size):
    
    mmap_array = np.memmap("mmap.dat", dtype=np.float32, mode='w+', shape=(num_events,8256,6403))
    
    f = File(file_name)
    f.add_group('wire_table')
    
    for i in range(0, num_events, chunk_size):

        actual_chunk_size = min(chunk_size, num_events - i)
        
        f.read_data(i, actual_chunk_size)
        evts = f.build_evt()
        
        for j, event in enumerate(evts):
            mmap_array[i + j,:,:] = event['wire_table']

    mmap_array.flush()
    
    return mmap_array

def new_create_events_array(file_name, num_events, chunk_size):
    
    f = File(file_name)
    f.add_group('wire_table')
    
    evts_array = np.empty((num_events,1067,2400))
    
    for i in range(0, num_events, chunk_size):

        actual_chunk_size = min(chunk_size, num_events - i)
        
        f.read_data(i, actual_chunk_size)
        evts = f.build_evt()
    
        for j, evt in enumerate(evts):
        
            # Get wire readouts for first plane only
            wires = evt["wire_table"]
            planeadcs = wires.query("local_plane==%i"%0)[['adc_%i'%i for i in range(0,ntimeticks())]].to_numpy()
            
            if planeadcs.size != 0:
            
                # Downsample first plane data
                f_downsample = 6
                planeadcs = block_reduce(planeadcs, block_size=(1,f_downsample), func=np.sum)
                
                # Apply cutoff and saturation to first plane
                adccutoff = 10.*f_downsample/6.
                adcsaturation = 100.*f_downsample/6.
                planeadcs[planeadcs<adccutoff] = 0
                planeadcs[planeadcs>adcsaturation] = adcsaturation
                
                print(f"Planeadcs shape: {planeadcs.T.shape}")
                print(f"Evts array shape: {evts_array[i+j].shape}")
                print(i+j)
                evts_array[i+j] = planeadcs.T
    
    

'''
def read_data(file_name, num_events):
    
    f = File(file_name)
    f.add_group('wire_table')
    f.read_data(0, num_events)
    evts = f.build_evt()
    print(evts[0]['wire_table'].shape)
    
    return evts


def create_events_array(evts):
    
    evts_array = np.empty((len(evts),1067,2400))

    for idx, evt in enumerate(evts):
        
        # Get wire readouts for first plane only
        wires = evt["wire_table"]
        planeadcs = wires.query("local_plane==%i"%0)[['adc_%i'%i for i in range(0,ntimeticks())]].to_numpy()
        
        # Downsample first plane data
        f_downsample = 6
        planeadcs = block_reduce(planeadcs, block_size=(1,f_downsample), func=np.sum)
        
        # Apply cutoff and saturation to first plane
        adccutoff = 10.*f_downsample/6.
        adcsaturation = 100.*f_downsample/6.
        planeadcs[planeadcs<adccutoff] = 0
        planeadcs[planeadcs>adcsaturation] = adcsaturation
        
        evts_array[idx] = planeadcs.T   
        
    return evts_array 
'''    


def plot_events(evts_array, output_directory):
    
    for idx, array in enumerate(evts_array):
        plt.imshow(array, cmap='jet')
        plt.savefig(output_directory + str(idx) + ".png")
        
  
evts = new_create_events_array("bnb_WithWire_00.h5", 10, 3)      
#evts = read_data("bnb_WithWire_00.h5", 5)
#evts_array = create_events_array(evts)
#plot_events(evts_array, "imgs/")