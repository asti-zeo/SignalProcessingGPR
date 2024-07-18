import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go



from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline
from scipy.signal import fftconvolve
from scipy.signal import hilbert

c = 3e8
time_to_pavement_surface = 0.5/c

def scientific_notation(x):
    return f"{x:.2e}"
# calculation definitions
def read_data(file):
    df = pd.read_csv(file)
    return df

def fit_template_signal(df , crop_idx, polynomial_degree):
    x = df['Time']
    y = df['Real']
    z = np.polyfit(x, y, polynomial_degree)
    return z

def get_fitted_template(z, x_vals):
    p = np.poly1d(z)
    return p(x_vals)

def hilbert_transform(df):
  x = df['Time']
  y = df['Real']
  analytical_y = hilbert(y)
  return x, abs(analytical_y)

def find_peaks_from_signal(analytical_y,time_series, height = 0):
  peak_ids, peak_height = find_peaks((analytical_y), height = 0)
  peak_times = time_series.iloc[peak_ids]

  combined_peaksid_height_array = np.hstack((peak_ids.reshape(-1, 1), peak_height['peak_heights'].reshape(-1,1)))
  sorted_peaksid_height_array = combined_peaksid_height_array[combined_peaksid_height_array[:, 1].argsort()]
  sorted_peaksid_height_array  = sorted_peaksid_height_array[::-1]

  return peak_times, peak_height, sorted_peaksid_height_array

#remove peaks before surface and add the next highest peak -- include df to find the times as well
def remove_peaks_before_surface(sorted_peaks_and_height_arr, number_of_layers, df):
  remove_idx = []
  peaks_found_before_surface_reflection = df['Time'].iloc[sorted_peaks_and_height_arr[: ,0]] < time_to_pavement_surface*2
  # peaks_found_before_surface_reflection

  for i in range(len(peaks_found_before_surface_reflection)):
      if peaks_found_before_surface_reflection.values[i] == True:
        number_of_layers += 1
        remove_idx.append(i)


  selected_peaks = sorted_peaks_and_height_arr[:number_of_layers]
  peaks_found_before_surface_reflection = df['Time'].iloc[selected_peaks[: ,0]] < time_to_pavement_surface*2

  selected_peaks = selected_peaks[~peaks_found_before_surface_reflection]
  return selected_peaks

def plot_graph(time_arr, *args):
  fig, ax = plt.subplots(figsize=(20, 6))
  ax.set_xlabel('Time')
  ax.set_ylabel('Amplitude')
  ax.vlines(time_to_pavement_surface*2, -1.5, 1.5, linestyle='--', color='k', label = 'Pavement surface - 0.5m')

  for arg in args:
    ax.plot(time_arr, arg)

  ax.legend()
  ax.grid()
  return fig, ax

def start_time_idx(selected_peaks, template_half_time, df):
  start_time_of_signal = df['Time'].iloc[selected_peaks[:,0]].values[0] - 1e-9
  start_time_of_signal_idx = np.argmin(abs(df['Time'] - start_time_of_signal))
  return start_time_of_signal_idx

def create_morlet_wave(fitted_template, start_time_idx, df, selected_peaks, current_peak_id, adjustmet_idx = 0):

  morlet_signal = np.zeros(len(df['Time']))
  morlet_signal[start_time_idx:start_time_idx+len(fitted_template)] = fitted_template*selected_peaks[:,1][current_peak_id]/max(fitted_template)

  return morlet_signal

def subtract_signal(received_signal, template_signal):
  return received_signal - template_signal


#peak id is from envelop -> return the start id of template in received signal
def get_subtract_signal(df, peak_id, template_time, template_array):
  #extract the df for given peak
  peak_df = df[(df['Time'] > df['Time'].iloc[peak_id] - template_time/2) & (df['Time'] < df['Time'].iloc[peak_id] + template_time/2)]

  #find the peak within extracted df

  peak_idx = peak_df['Real'].idxmax()
  peak_amplitude = peak_df['Real'].max()/max(template_array)
  print(peak_amplitude)
  # peak_idx  =50
  template_peak_idx = np.argmax(template_array)
  # template_peak_idx = 10
  left =template_peak_idx
  right = len(template_array) - template_peak_idx
  print(peak_idx,left, right)
  subtract_wave = np.zeros(len(df['Time']))
  subtract_wave[peak_idx - left: peak_idx+right] = template_array

  phase_shift = 1 if df['Real'].iloc[peak_idx] > 0 else -1

  return subtract_wave*peak_amplitude*phase_shift, peak_idx

def get_subtracted_signal(signal_to_be_subtracted, subtract_template):
  return signal_to_be_subtracted - subtract_template



# upload file
st.title("Enhanced Layer Detector")
a_scan = st.file_uploader("Upload a file", type=["csv"])

if a_scan is not None:
   st.session_state.df = pd.read_csv(a_scan)
   
   original_signal_plot = px.line(st.session_state.df, x='Time', y='Real', title='Original Signal')
   original_signal_plot.update_layout(xaxis_title='Time (s)', yaxis_title='Amplitude')
   original_signal_plot.update_xaxes(rangeslider_visible=True)
   original_signal_plot.data[0].line.color = 'mediumturquoise'
   original_signal_plot.data[0].name = 'Received Signal'
   st.plotly_chart(original_signal_plot)
   st.divider()

    #construct template signal
   st.session_state.constructed_template_signal = st.session_state.df[(st.session_state.df['Time'] > 0.7e-9) & (st.session_state.df['Time'] < 2.7e-9)]
   st.session_state.constructed_template_signal.reset_index(inplace =True)
   z = fit_template_signal(st.session_state.constructed_template_signal,100, 18 )
   p = get_fitted_template(z, st.session_state.constructed_template_signal['Time'])

   st.session_state_template_time = st.session_state.constructed_template_signal['Time'].iloc[-1] - st.session_state.constructed_template_signal['Time'].iloc[0]
#    st.write(st.session_state.df['Time'].iloc[-1] -st.session_state.df['Time'].iloc[0] )
   peak_idx = st.session_state.constructed_template_signal['Real'].idxmax()
#    print(peak_idx)

   st.session_state.templated_signal = p
   
   left_side = p[:peak_idx+1]
   right_side = np.flip(left_side[:-2])
#    print(len(right_side), len(left_side), len(p))

   st.session_state.templated_signal_symmetry = np.concatenate((left_side, right_side))
    #plot template signal
   template_signal_plot = go.Figure()
   template_signal_plot.add_trace(go.Scatter( x=st.session_state.constructed_template_signal['Time'], y=p,  mode='lines', name = 'Fitted Template signal'))
   template_signal_plot.add_trace(go.Scatter(x=st.session_state.constructed_template_signal['Time'], y=st.session_state.templated_signal_symmetry , mode='lines', name='Symmetry Template signal'))
   template_signal_plot.update_layout(xaxis_title='Time (s)', yaxis_title='Amplitude')
   template_signal_plot.data[1].line.color = 'darkviolet'
   template_signal_plot.data[0].line.dash= 'dash'
   template_signal_plot.update_layout(title = 'Template Signal')
   st.plotly_chart(template_signal_plot)




   # Calculate envelope
   st.session_state.time ,st.session_state.analytical_y = hilbert_transform(st.session_state.df)
#    st.write(type(st.session_state.analytical_y))
   st.session_state.peak_times, peak_height, sorted_peaksid_height_array = find_peaks_from_signal(st.session_state.analytical_y, st.session_state.time)
   original_signal_plot_with_envelope = go.Figure()
   original_signal_plot_with_envelope.add_trace(go.Scatter(x=st.session_state.df['Time'], y=st.session_state.df['Real'], mode='lines' , name='Received Signal'))
   original_signal_plot_with_envelope.add_trace(go.Scatter(x=st.session_state.time, y=st.session_state.analytical_y, mode='lines', name='Envelope'))
   original_signal_plot_with_envelope.add_trace(go.Scatter(x=st.session_state.peak_times, y=peak_height['peak_heights'], mode='markers', name='Envelope Peaks'))
#    original_signal_plot_with_envelope.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Pavement Surface Reflection'))
   original_signal_plot_with_envelope.add_trace(
    go.Scatter(
        x=[time_to_pavement_surface*2, time_to_pavement_surface*2],
        y=[min(st.session_state.df['Real']), max(st.session_state.df['Real'])],
        mode="lines",
        line=dict(color="fuchsia", width=2, dash="dash"),
        name="Pavement Surface Reflection(0.5m)"
    )
)
#    original_signal_plot_with_envelope.add_vline(x=time_to_pavement_surface*2, line_width=1, line_dash="dash", line_color="orange", name = 'Pavement Surface Reflection')
   original_signal_plot_with_envelope.update_layout(xaxis_title='Time (s)', yaxis_title='Amplitude')
   original_signal_plot_with_envelope.update_xaxes(rangeslider_visible=True)
   original_signal_plot_with_envelope.data[0].line.color = 'mediumturquoise'
   original_signal_plot_with_envelope.data[1].line.color = 'yellow'
#    original_signal_plot_with_envelope.data[1].line.dash = 'dash'
   original_signal_plot_with_envelope.data[2].line.color = 'crimson'
   original_signal_plot_with_envelope.data[0].name = 'Received Signal'
#    original_signal_plot_with_envelope.data[3].line.color = 'orange'

   original_signal_plot_with_envelope.update_layout(
    title='Original Signal with Envelop and Peaks',
    legend=dict(
        title="Legend",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
   st.plotly_chart(original_signal_plot_with_envelope)
   st.divider()
   col1, col2 = st.columns(2)
   with col1:
    st.session_state.num_layers = st.number_input('Number of Layers (User Input)', min_value=1, max_value=10, value=5)
   # radio buttons
   selected_layers = []

   st.session_state.layers_selected = st.multiselect('Select Layers', options=list(range(1, st.session_state.num_layers + 1)), default=[i for i in range(1, st.session_state.num_layers + 1)])

   st.selected_templated_signals = st.radio('Select Templat Signal', options=['Original Fit', 'Symmetry Fit'], index=0)
   
   if st.selected_templated_signals == 'Original Fit':
     st.template_signal = p
   else:
     st.template_signal = st.session_state.templated_signal_symmetry
   st.session_state.selected_peaks = remove_peaks_before_surface(sorted_peaksid_height_array, st.session_state.num_layers, st.session_state.df)
   
#    sw,_ = get_subtract_signal(st.session_state.df,st.session_state.selected_peaks[0,0].astype(int), 2.5e-9, st.template_signal )

#    st.session_state.subtracted_signal = subtract_signal(st.session_state.df['Real'], sw)
  

    # Plot the subtracted signal
#    subtracted_signal_plot = px.line(st.session_state.df, x='Time', y=st.session_state.subtracted_signal, title='Subtracted Signal')

   
   if st.button('Remove Selected Layers'):
    #  print(st.session_state.df['Time'].iloc[0,st.session_state.selected_peaks] )
    #  st.write(st.session_state.selected_peaks)
     st.session_state.peak_times_disp = []
    #  st.write(time_to_pavement_surface*2)
    #  if st.session_state.peak_times.iloc[0] < time_to_pavement_surface*2:
    #     st.write(st.session_state.peak_times.iloc[1])
    #  else:
    #    st.write(st.session_state.peak_times.iloc[0])


     st.session_state.peak_times_disp.append(st.session_state.peak_times.iloc[0])
    #  print(st.session_state.peak_times_disp, type(st.session_state.peak_times_disp))
     st.session_state.original_signal = st.session_state.df['Real']
     temp_df = copy.deepcopy(st.session_state.df)
     for layer in st.session_state.layers_selected:
       subtract_signal_, peak_idx = get_subtract_signal(temp_df,st.session_state.selected_peaks[layer-1,0].astype(int), 2e-9, st.template_signal )
       st.session_state.subtracted_signal = subtract_signal(st.session_state.original_signal, subtract_signal_)
       st.session_state.original_signal = st.session_state.subtracted_signal

       temp_df = pd.DataFrame({'Time': st.session_state.df['Time'], 'Real': st.session_state.subtracted_signal})
    #    print(temp_df.head())
       time, analytical_y = hilbert_transform(temp_df)
    #    peak_times, peak_height, sorted_peaksid_height_array = find_peaks_from_signal(analytical_y, time)
    #    analy_plot = px.line(x=time, y=analytical_y, title='Analytical Signal')
    #    st.plotly_chart(analy_plot)
    #    st.write(peak_times,analytical_y )
    #    print(peak_times.iloc[0])
       

       seleceted_peaks = remove_peaks_before_surface(sorted_peaksid_height_array, st.session_state.num_layers -(layer+1), st.session_state.df)
    #    st.write(seleceted_peaks)
       st.session_state.peak_times_disp.append(st.session_state.df['Time'].iloc[peak_idx])
     subtracted_signal_plot = go.Figure()

     subtracted_signal_plot.add_trace(go.Scatter(x=st.session_state.df['Time'], y=st.session_state.df['Real'], mode='lines', name='Original Signal'))
    #  subtracted_signal_plot.add_trace(go.Scatter(x=st.session_state.df['Time'], y=sw, mode='lines', name='Subtract Signal'))
     subtracted_signal_plot.add_trace(go.Scatter(x=st.session_state.df['Time'], y=st.session_state.subtracted_signal, mode='lines', name='Subtracted Signal'))
     subtracted_signal_plot.update_xaxes(rangeslider_visible=True)
     #    subtracted_signal_plot.data[0].name = 'Subtracted Signal'
     subtracted_signal_plot.data[1].line.color = 'lime'
     st.plotly_chart(subtracted_signal_plot)
     st.session_state.peak_times_disp = st.session_state.peak_times_disp[1:]
     st.header('Time for Given Layers')

     time_df = pd.DataFrame({'Layer No.': [f'Layer {i+1}' for i in range(st.session_state.num_layers)], 
                             'Time(s)': st.session_state.peak_times_disp,
                             'Depth(mm)': np.array(st.session_state.peak_times_disp)*3e8*1000/2,
                             
                             
                             })
     time_df['Time(s)'] = time_df['Time(s)'].apply(lambda x: f'{x:.2e}')
     st.table(time_df)

     #create 2 points with simialar heights for  all layer
     x = np.linspace(0, 100, 20)
     y = np.linspace(0, 200, 20)

     x_grid, y_grid = np.meshgrid(x,y)

     grid = np.stack((x_grid, y_grid,np.random.normal(loc=time_df['Depth(mm)'].iloc[0], scale=10, size=(20, 20))), axis = -1)
     grid_temp =[]
     for i in range (len(time_df['Depth(mm)'])):
       
        grid_temp.append(np.random.normal(loc=time_df['Depth(mm)'].iloc[i], scale=10, size=(20, 20)))
    
       
     print(grid.shape, grid)
    # Create a 3D surface plot
     x_vals = grid[:, :, 0]
     y_vals = grid[:, :, 1]
    #  z_vals = grid[:, :, 2]
     color_schemes = [
    'Viridis',
    'Cividis',
    'Bluered',
    'Jet',
    'Plasma',
    'solar',
    'spectral',
    'delta',
    'Jet',
    'Bluered'
]
     fig = go.Figure()
     for i in range(len(grid_temp)):
        fig.add_trace(go.Surface(x=x_vals, y=y_vals, z=grid_temp[i], colorscale=color_schemes[i%10], showscale=False))

     fig.update_layout(
    title='Layer levels',
    scene=dict(
        xaxis_title='X Direction',
        yaxis_title='Y Direction',
        zaxis=dict(title='Depth (mm)', autorange = 'reversed')
            # Adjust the range as needed
    ),
    width = 1200,
             height = 800 
)
     
     ##range=[min(time_df['Depth(mm)'])-min(time_df['Depth(mm)'])/2, max(time_df['Depth(mm)'])+min(time_df['Depth(mm)'])/2]

    #  fig = plt.figure()
    #  ax = fig.add_subplot(111, projection='3d')
    #  # Plot the surface. 
    #  ax.plot_surface(x_grid, y_grid, z_vals, cmap='viridis')

    #  st.pyplot(fig)
     st.plotly_chart(fig)

     
st.markdown("<br><br><hr><center>© 2024 Siththarththan Arunthavabalan. All rights reserved.</center>", unsafe_allow_html=True)


 

     