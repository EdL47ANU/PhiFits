import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

def GausstoTesla(MagneticField):
    if np.max(MagneticField) > 20:
        MagneticField = MagneticField/10000
    return MagneticField

def Normalise(Intensity):
    Intensity = Intensity/np.max(Intensity)
    return Intensity

def Seperate(Data):
    MagField = Data[:,0]
    Intensities = Data[:,1:]
    return MagField, Intensities

def Flatten(Intensities):
    Intensities = np.reshape(Intensities, 1)
    Mask = np.array(len(Intensities))
    for i in range(len(Intensities)):
        if Intensities[i] ==0.0:
            Mask[i] = False
        else:
            Mask[i] = True
    Intensities = Intensities[Mask]
    return Intensities

def Trim(MagField, Intensities):
    mask = np.array(len(MagField), type = bool)
    for i in range(len(Intensities)):
        if Intensities[i] < 0.001:
            mask[i] = False
        else:
            mask[i] = True
    MagField = MagField[mask]
    Intensities = Intensities[mask]
    return MagField, Intensities
    
def FieldSwitches(MagField):
    x_gaps = np.diff(MagField)
    threshold = 0.1 #Tesla
    split_indices = np.where(x_gaps > threshold)[0]
    print(split_indices)
    if split_indices.size > 0:
        xlims = [(MagField[0], MagField[split_indices[0]])]
        for i in range(1, len(split_indices)):
            xlims.append((MagField[split_indices[i-1]+1], MagField[split_indices[i]]))
        xlims.append((MagField[split_indices[-1]+1], MagField[-1]))
    else:
        xlims = [(MagField[0], MagField[-1])]
    return xlims

def plot_data(input_files):
    return

#        all_xlims.extend(xlims)
#        all_y_columns.append(y_columns)
#        all_x.append(x)

    
def plot_data(input_files):
    all_xlims = []
    all_y_columns = []
    all_x = []
    legend_labels = []
    colors = ['blue', 'red']  # Define colors for each file

    for file_index, input_file in enumerate(input_files):
        if input_file.endswith('.out'):
            data = np.loadtxt(input_file, delimiter=' ')
        else:
            data = np.loadtxt(input_file, converters={i: lambda s: float(s.decode('utf-8').replace('.', '0.')) if s.decode('utf-8') == '.' else float(s) for i in range(5)})
        print("Shape of data: ", data.shape)    
        
        # Determine the legend label based on the file name
        file_name = os.path.splitext(os.path.basename(input_file))
        #legend_label = file_name.split('.')[0]
        #legend_labels.append(legend_label)
        #print(legend_label)
        legend_labels.append(file_name)

        # Extract columns
        x = data[:, 0]  
        y = data[:,1:]
        x = GausstoTesla(x)
        y = Normalise(y)
        
        print("Data",data[:, 0])
        y_columns = []
        for i in range(1, data.shape[1]):
            y = data[:, i]
            y = y / np.max(y)  # Normalize the data
            
            # a for loop to scan through the y data, if the absolute value is less than 0.05, set it to 0
            for j in range(len(y)):
                if abs(y[j]) > 0.0005:
                #if abs(y[j]) < 0.05:
                    #y[j] = 0
#                    y_columns.extend(y[j])
                    y_columns.append(y[j])
                #y_columns.extend(y)
        
        print("Number of y values: ", len(y_columns))
        print("Number of x values: ", len(x))
        
        # Calculate the ranges for brokenaxes based on gaps in x
        x_gaps = np.diff(np.sort(x))
        threshold = 0.1 #Tesla
        split_indices = np.where(x_gaps > threshold)[0]
        print(split_indices)
        if split_indices.size > 0:
            xlims = [(x[0], x[split_indices[0]])]
            for i in range(1, len(split_indices)):
                xlims.append((x[split_indices[i-1]+1], x[split_indices[i]]))
            xlims.append((x[split_indices[-1]+1], x[-1]))
        else:
            xlims = [(x[0], x[-1])]

        all_xlims.extend(xlims)
        all_y_columns.append(y_columns)
        all_x.append(x)
    
    # Remove duplicate xlims with a tolerance of 3 decimal places
    unique_xlims = []
    for xlim in all_xlims:
        if not any(np.isclose(xlim, unique_xlim, atol=1e-3).all() for unique_xlim in unique_xlims):
            unique_xlims.append(xlim)
    all_xlims = unique_xlims
    print(unique_xlims)
    
    # Plot the data with broken x-axis
    bax = brokenaxes(xlims=all_xlims, hspace=.05)
    for i, y_columns in enumerate(all_y_columns):
        for y in y_columns:
            bax.plot(all_x[i], y, linestyle='-', label=legend_labels[i], color=colors[i % len(colors)])
    
    bax.set_xlabel('Magnetic Field [T]', size=18)
    bax.set_ylabel('Absorption', size=18)
    bax.set_title('EPR Spectra', size=22)
    bax.legend(fontsize=16, loc='lower left')
    bax.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Plot.py <input_file1> <input_file2> ... <input_fileN>")
        sys.exit(1)
    input_files = sys.argv[1:]
    plot_data(input_files)