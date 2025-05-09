import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns



def load_tree_data(tree_file_path='Tree.xlsx', sheet_name='TreeI', header_row=0):
    """
    Load tree data from Excel file and return the dataframe.
    
    Parameters:
    - tree_file_path: Path to the Excel file (default: 'Tree.xlsx')
    - sheet_name: Name of the sheet to read (default: 'TreeI')
    - header_row: Row index containing column headers (default: 0)
    
    Returns:
    - DataFrame containing the tree data or None if file not found
    """
    # Initialize an empty dataframe
    df_combined = pd.DataFrame()
    
    # Check if file exists to avoid unnecessary errors
    if not os.path.exists(tree_file_path):
        print(f"Error: File not found at {tree_file_path}")
        return None
    
    try:
        # Read the tree file with headers at the specified row
        df_combined = pd.read_excel(tree_file_path, sheet_name=sheet_name, header=header_row)
        print(f"Successfully loaded {len(df_combined)} rows from: {tree_file_path}")
        print(f"Found columns: {list(df_combined.columns)}")
        return df_combined
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def create_tree_height_violin(df, height_col='Height_m', status_col='Status', 
                              colors={'Intact': '#358600', 'Degraded': '#C08552'}):
    """
    Create an interactive violin plot comparing tree heights by status using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tree data
    height_col : str
        Column name for tree height (default: 'Height_m')
    status_col : str
        Column name for status (default: 'Status')
    colors : dict
        Dictionary mapping status values to colors
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive violin plot figure
    """
     
    # Convert height column to numeric
    df[height_col] = pd.to_numeric(df[height_col], errors='coerce')
    
    # Check if we have valid height data
    if not df[height_col].notna().any():
        print(f"No valid data found in {height_col} column")
        return None
    
    # Calculate summary statistics for annotation
    stats = df.groupby(status_col)[height_col].agg(['mean', 'median']).round(2)
    
    # Create the violin plot with plotly express
    fig = px.violin(
        df, 
        y=height_col,
        x=status_col,
        color=status_col,
        
        color_discrete_map=colors,
        
        labels={height_col: "Height (m)", status_col: "Forest Condition"},
        template="plotly_white",  # Clean white background with light grid
        
    )
    
    # Customize layout for a more striking appearance
    fig.update_layout(
        
        
        title=dict(
            text='<b></b>',
            font=dict(size=24), # Increased font size
            x=0.5, # Center the title
            xanchor='center',
            y=0.95, # Adjust vertical position if needed
            yanchor='top'
        ),          
        showlegend=False,
        yaxis_title="<b>Height (meters)</b>",
        xaxis_title="<b>Forest Condition</b>",
        violingap=0.2,  # Gap between violins
          # 'overlay' for semi-transparent violins
         
    )
    
          
            
    return fig


import plotly.graph_objects as go # Make sure go is imported
# ... other imports ...

def create_tree_height_histogram(df, height_col='Height_m', status_col='Status',
                                colors={'Intact': '#358600', 'Degraded': '#C08552'}
                                ):
    """
    Create an interactive overlaid histogram comparing tree heights by status using Plotly (raw counts).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tree data
    height_col : str
        Column name for tree height (default: 'Height_m')
    status_col : str
        Column name for status (default: 'Status')
    colors : dict
        Dictionary mapping status values to colors
    title : str
        Title for the plot

    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive histogram figure
    """


    # Convert height column to numeric
    df[height_col] = pd.to_numeric(df[height_col], errors='coerce')

    # Check if we have valid height data
    if not df[height_col].notna().any():
        print(f"No valid data found in {height_col} column")
        return None

    # Extract data for each group
    status_values = df[status_col].unique()
    valid_statuses = [s for s in status_values if s in colors]

    # Create the figure object
    fig = go.Figure()

    # Add a histogram trace for each status
    for status in valid_statuses:
        status_heights = df[df[status_col] == status][height_col].dropna()

        if len(status_heights) > 0:
            fig.add_trace(go.Histogram(
                x=status_heights,
                name=status, # This will appear in the legend
                marker_color=colors.get(status, '#888888'),
                opacity=0.75, # Make bars semi-transparent for overlap visibility
                histnorm=None, # Use raw counts
                xbins=dict( # Optional: Define binning explicitly if needed
                    # start=all_heights.min(),
                    # end=all_heights.max(),
                    # size=bin_size # Calculated earlier, or let Plotly decide
                )
            ))

    # Set y-axis title for raw counts
    yaxis_title_text = "<b>Count</b>"


    # Customize layout
    fig.update_layout(
        title={
            'text': '<b><span style="color:green;">Intact</span> areas have more species with different <span style="color:green;">heights</span></b>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white'}
        },
        xaxis_title={
            'text': "<b>Tree Height (meters)</b>",
            'font': {'size': 16}
        },
        yaxis_title={
            'text': yaxis_title_text, # Use fixed title for counts
            'font': {'size': 16}
        },
        barmode='overlay', # Overlay the histograms
        template='plotly_white',
        legend_title_text='Forest Condition', # Add legend title
        margin=dict(l=50, r=50, t=100, b=50),

        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
    )

    return fig


def create_bird_activity_heatmap(df, date_col='Date', time_col='Time_Stamp', 
                                 
                                title='Bird Activity:  <span style="color:steelblue;">The dawn chorus</span> and  <span style="color:steelblue;">The evening chorus</span>'):
    """
    Create an interactive heatmap of bird activity by day of week and hour using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the bird sightings data
    date_col : str
        Column name for date (default: 'Date')
    time_col : str
        Column name for time (default: 'Time_Stamp')
    colors : list
        List of colors for the heatmap gradient (default: blues)
    title : str
        Title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive heatmap figure
    """
    data = df.copy()
    
    # Process the date column
    if date_col in data.columns:
        # Try to convert the date column to datetime
        try:
            if data[date_col].dtype != 'datetime64[ns]':
                # Try common date formats
                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                    try:
                        data[date_col] = pd.to_datetime(data[date_col], format=date_format, errors='raise')
                        break
                    except:
                        continue
                
                # If still not converted, try pandas auto-detection
                if data[date_col].dtype != 'datetime64[ns]':
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            
            # Extract day of week
            data['Day_of_Week'] = data[date_col].dt.day_name()
        except Exception as e:
            print(f"Error processing date data: {e}")
            return None
    else:
        print(f"Date column '{date_col}' not found in the dataset")
        return None
    
    # Process the time column
    if time_col in data.columns:
        try:
            # Handle different time formats
            if data[time_col].dtype == 'object':
                # Try parsing as time strings
                try:
                    data['Hour'] = pd.to_datetime(data[time_col], format='%H:%M:%S', errors='coerce').dt.hour
                except:
                    # Try parsing as datetime strings
                    try:
                        data['Hour'] = pd.to_datetime(data[time_col], errors='coerce').dt.hour
                    except:
                        # Last resort: manual extraction from string
                        data['Hour'] = data[time_col].str.extract(r'(\d+):', expand=False).astype(float)
            elif pd.api.types.is_datetime64_dtype(data[time_col]):
                # Already datetime type
                data['Hour'] = data[time_col].dt.hour
            else:
                # Numeric type - assume it's already the hour
                data['Hour'] = data[time_col]
            
            # Ensure hours are numeric and in the correct range
            data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
            data['Hour'] = data['Hour'].fillna(0).astype(int) % 24
        except Exception as e:
            print(f"Error processing time data: {e}")
            return None
    else:
        print(f"Time column '{time_col}' not found in the dataset")
        return None
    
    # Remove any rows with missing processed data
    data = data.dropna(subset=['Day_of_Week', 'Hour'])
    
    if len(data) == 0:
        print("No valid data after processing dates and times")
        return None
    
    # Order days of week properly
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Count sightings by hour and day of week
    heatmap_data = data.groupby(['Day_of_Week', 'Hour']).size().reset_index(name='Count')
    
    # Pivot data for the heatmap
    pivot_data = heatmap_data.pivot(
        index='Day_of_Week', 
        columns='Hour', 
        values='Count'
    ).fillna(0)
    
    # Reorder days
    pivot_data = pivot_data.reindex(days_order)
    
    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f"{hour:02d}:00" for hour in range(24)],
        y=pivot_data.index,
        colorscale='ylgnbu',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Bird sightings: %{z}<extra></extra>',
        colorbar=dict(
            title='Number of<br>Bird Sightings',
          
        )
    ))
    
    # Calculate peak activity times for annotations
    total_by_hour = data.groupby('Hour').size()
    peak_hour = total_by_hour.idxmax()
    peak_hour_count = total_by_hour.max()
    
    total_by_day = data.groupby('Day_of_Week').size()
    day_order_dict = {day: i for i, day in enumerate(days_order)}
    sorted_days = sorted(total_by_day.index, key=lambda x: day_order_dict.get(x, 99))
    peak_day = total_by_day.idxmax()
    peak_day_count = total_by_day.max()
    
    # Customize layout
    fig.update_layout(
        title={
            'text': f"<b></b>",
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#333333'}
        },
        xaxis={
            'title': "<b>Hour of Day</b>",
            
            'tickfont': {'size': 12},
            'tickangle': -30,
            'dtick': 2,  # Show every 2 hours for clarity
        },
        yaxis={
            'title': "<b>Day of Week</b>",
            
            'tickfont': {'size': 14},
            'categoryorder': 'array',
            'categoryarray': days_order
        },
        margin=dict(l=80, r=50, t=100, b=80),
        plot_bgcolor='rgba(248, 250, 255, 0.9)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
        
    )
    
    return fig


def create_bird_activity_polar(df, date_col='Date', time_col='Time_Stamp', 
                             title='Bird Activity:  <span style="color:#FF3C8E;">The dawn chorus</span> and  <span style="color:#FF3C8E;">The evening chorus</span>',
                             colorscale=None, time_filter=None):
    """
    Create an interactive polar chart showing bird activity by day of week and hour.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the bird sightings data
    date_col : str
        Column name for date (default: 'Date')
    time_col : str
        Column name for time (default: 'Time_Stamp')
    title : str
        Title for the plot
    colorscale : list
        List of colors for the intensity scale (default is yellow to purple)
    time_filter : str
        Optional filter for time of day: 'morning' (1-12) or 'evening' (13-24)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive polar chart figure
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Process the date column to extract day of week
    if date_col in data.columns:
        try:
            if data[date_col].dtype != 'datetime64[ns]':
                # Try common date formats
                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                    try:
                        data[date_col] = pd.to_datetime(data[date_col], format=date_format, errors='raise')
                        break
                    except:
                        continue
                
                # If still not converted, try pandas auto-detection
                if data[date_col].dtype != 'datetime64[ns]':
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            
            # Extract day of week
            data['Day_of_Week'] = data[date_col].dt.day_name()
        except Exception as e:
            print(f"Error processing date data: {e}")
            return None
    else:
        print(f"Date column '{date_col}' not found in the dataset")
        return None
    
    # Process the time column to extract hour
    if time_col in data.columns:
        try:
            # Handle different time formats
            if data[time_col].dtype == 'object':
                # Try parsing as time strings
                try:
                    data['Hour'] = pd.to_datetime(data[time_col], format='%H:%M:%S', errors='coerce').dt.hour
                except:
                    # Try parsing as datetime strings
                    try:
                        data['Hour'] = pd.to_datetime(data[time_col], errors='coerce').dt.hour
                    except:
                        # Last resort: manual extraction from string
                        data['Hour'] = data[time_col].str.extract(r'(\d+):', expand=False).astype(float)
            elif pd.api.types.is_datetime64_dtype(data[time_col]):
                # Already datetime type
                data['Hour'] = data[time_col].dt.hour
            else:
                # Numeric type - assume it's already the hour
                data['Hour'] = data[time_col]
            
            # Ensure hours are numeric and in the correct range
            data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
            data['Hour'] = data['Hour'].fillna(0).astype(int) % 24
        except Exception as e:
            print(f"Error processing time data: {e}")
            return None
    else:
        print(f"Time column '{time_col}' not found in the dataset")
        return None
    
    # Remove any rows with missing processed data
    data = data.dropna(subset=['Day_of_Week', 'Hour'])
    
    if len(data) == 0:
        print("No valid data after processing dates and times")
        return None
    
    # Apply time filtering if specified
    filtered_title = title
    if time_filter == 'morning':
        data = data[(data['Hour'] >= 1) & (data['Hour'] <= 12)]
        filtered_title = '<span style="color:#FF3C8E;">The dawn chorus</span>'
        if len(data) == 0:
            print("No data available for morning hours (1-12)")
            return None
    elif time_filter == 'evening':
        data = data[(data['Hour'] >= 13) & (data['Hour'] <= 24)]
        filtered_title = '<span style="color:#FF3C8E;">The evening chorus</span>'
        if len(data) == 0:
            print("No data available for evening hours (13-24)")
            return None
    
    # Define days of week and hour ranges
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Define hour ranges based on filtering
    if time_filter == 'morning':
        hours = list(range(1, 13))  # 1-12
    elif time_filter == 'evening':
        hours = list(range(13, 25))  # 13-24
    else:
        hours = list(range(24))  # 0-23
    
    # Count sightings by day and hour
    activity_counts = data.groupby(['Day_of_Week', 'Hour']).size().reset_index(name='Count')
    
    # Prepare data for the polar chart
    chart_data = []
    for day in days:
        for hour in hours:
            # Find count for this day and hour
            filtered = activity_counts[(activity_counts['Day_of_Week'] == day) & 
                                      (activity_counts['Hour'] == hour)]
            
            count = filtered['Count'].values[0] if not filtered.empty else 0
            
            chart_data.append({
                'day': day,
                'hour': hour,
                'count': count
            })
    
    # Create a DataFrame with the prepared data
    df_polar = pd.DataFrame(chart_data)
    
    # Map days to radius values (Monday=1, Tuesday=2, etc.)
    day_to_radius = {day: i+1 for i, day in enumerate(days)}
    df_polar['r1'] = df_polar['day'].map(day_to_radius)
    df_polar['r2'] = df_polar['r1'] + 1
    
    # Adjust the hour-to-angle mapping based on filtering
    if time_filter == 'morning':
        # Map hours 1-12 to 0-360 degrees
        hour_to_angle_start = {hour: (hour-1) * 30 for hour in hours}  # 30 degrees per hour
        hour_to_angle_end = {hour: hour * 30 for hour in hours}
    elif time_filter == 'evening':
        # Map hours 13-24 to 0-360 degrees
        hour_to_angle_start = {hour: (hour-13) * 30 for hour in hours}  # 30 degrees per hour
        hour_to_angle_end = {hour: (hour-12) * 30 for hour in hours}
    else:
        # Original mapping: 24 hours to 360 degrees (15 degrees per hour)
        hour_to_angle_start = {hour: hour * 15 for hour in hours}
        hour_to_angle_end = {hour: (hour + 1) * 15 for hour in hours}
    
    df_polar['theta_start'] = df_polar['hour'].map(hour_to_angle_start)
    df_polar['theta_end'] = df_polar['hour'].map(hour_to_angle_end)
    
    # Find max count for normalization
    max_count = df_polar['count'].max()
    if max_count > 0:
        df_polar['normalized_count'] = df_polar['count'] / max_count
    else:
        df_polar['normalized_count'] = 0
    
    # Define a colorscale (yellow to purple)
    if colorscale is None:
        if time_filter == 'evening':
            # Blue colorscale for evening
            colorscale = [
                   '#61A1FF', 
    '#408CFF', '#2575F7', '#1F5DD0',  '#152D83', 
    '#10155C', '#24004A', '#3D0055', '#56005F', 
    '#7A0071', '#A3008A'
            ]
        else:
            # Default yellow to purple colorscale for morning or no filter
            colorscale = [
                '#FFFF99', '#FFED6F', '#FFE14D', '#FFD52C', '#FFC914', '#FFBA10', 
                '#FF9D32', '#FF7E4D', '#FF5E68', '#FF3C8E', '#F033C3', '#C340E0',
                '#9D51F3', '#6872FE', '#00A6FF', '#00CCFF'
            ]
    
    # Function to map normalized count to the colorscale
    def get_color(normalized_value, colors):
        index = int(normalized_value * (len(colors) - 1))
        return colors[index]
    
    # Create the figure
    fig = go.Figure()
    
    for index, row in df_polar.iterrows():
        r1 = row['r1']
        r2 = row['r2']
        t1 = row['theta_start']
        t2 = row['theta_end']
        
        color = get_color(row['normalized_count'], colorscale)
        hover_text = f"Day: {row['day']}<br>Hour: {row['hour']:02d}:00<br>Sightings: {row['count']}"
        
        fig.add_trace(go.Scatterpolar(
            theta=[t1, t2, t2, t1, t1],
            r=[r1, r1, r2, r2, r1],
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(color='white', width=0.7),
            name=f"{row['day']} {row['hour']}:00",
            showlegend=False,
            hoverinfo='text',
            text=hover_text
        ))
    
    # Adjust tick labels based on the time filter
    if time_filter == 'morning':
        tick_vals = [(i - 0.5) * 30 for i in range(1, 13)]  # 12 ticks for morning hours
        tick_text = [f"{i:02d}:00" for i in range(1, 13)]
    elif time_filter == 'evening':
        tick_vals = [(i - 0.5) * 30 for i in range(1, 13)]  # 12 ticks for evening hours
        tick_text = [f"{i+12:02d}:00" for i in range(1, 13)]
    else:
        tick_vals = [(i + 0.5) * 15 for i in range(24)]  # 24 ticks for all hours
        tick_text = [f"{i:02d}:00" for i in range(24)]
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                showticklabels=False,
                showgrid=False,
                showline=False
            ),
            angularaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_text,
                direction="clockwise",
                period=360,
                rotation=90,
                showgrid=False,
                showline=False,
                visible=True
            )
        ),
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        title={
            'text': f"<b></b>",
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 30, 'color': '#333333'}
        },
        showlegend=False,
        template="plotly_white",
        margin=dict(t=100, b=50, l=50, r=60)
    )
    
    # Add colorbar to show intensity scale
    z = np.linspace(0, 1, 100)
    colorbar_colors = [get_color(val, colorscale) for val in z]
    
    # Add a hidden trace to show the colorbar
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorbar_colors,
            showscale=True,
            cmin=0,
            cmax=max_count,
            colorbar=dict(
                title="Bird Sightings"
            )
        ),
        hoverinfo='none'
    ))
    
    return fig


def create_bird_conservation_plot(df, iucn_col='IUCN_Status', status_col='Status', 
                                 title='All <span style="color:#F9E814;">vulnerable</span> <span style="color:green;">species of birds</span> are only seen in <span style="color:green;">Intact areas</span>   ',
                                 colors=None):
    """
    Create an interactive stacked bar chart showing bird conservation status distribution
    in intact vs degraded areas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the bird data
    iucn_col : str
        Column name for IUCN conservation status (default: 'IUCN_Status')
    status_col : str
        Column name for habitat status (default: 'Status')
    title : str
        Title for the plot
    colors : dict
        Dictionary mapping status values to colors
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive stacked bar chart figure
    """
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np
    
    # Create a copy to avoid modifying the original dataframe
    birds_df = df.copy()
    
    # Check for required columns
    required_columns = [iucn_col, status_col]
    if not all(col in birds_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in birds_df.columns]
        print(f"Error: Missing required columns: {missing}")
        return None
    
    # Fill missing values with 'Unknown' to avoid NaN issues
    birds_df[iucn_col] = birds_df[iucn_col].fillna('Unknown')
    birds_df[status_col] = birds_df[status_col].fillna('Unknown')
    
    # Filter for just Intact and Degraded statuses
    valid_statuses = ['Intact', 'Degraded']
    birds_df = birds_df[birds_df[status_col].isin(valid_statuses)]
    
    if len(birds_df) == 0:
        print(f"No data after filtering for {valid_statuses}")
        return None
    
    # Group by IUCN_Status and Status, then count
    iucn_counts = birds_df.groupby([iucn_col, status_col]).size().reset_index(name='Count')
    
    # Pivot the data for easier processing
    pivot_counts = iucn_counts.pivot(index=iucn_col, columns=status_col, values='Count').fillna(0)
    
    # Ensure both status columns exist
    for status in valid_statuses:
        if status not in pivot_counts.columns:
            pivot_counts[status] = 0
    
    # Calculate total counts per IUCN status for sorting
    pivot_counts['Total'] = pivot_counts['Intact'] + pivot_counts['Degraded']
    
    # Define IUCN status order from least to most concern
    iucn_order = ['Unknown', 'Least Concern', 'Near Threatened', 'Vulnerable', 
                 'Endangered', 'Critically Endangered', 'Extinct in the Wild', 'Extinct']
    
    # Filter for statuses that exist in our data
    iucn_order = [status for status in iucn_order if status in pivot_counts.index]
    
    # Reindex using the order
    pivot_counts = pivot_counts.reindex(iucn_order)
    
    # Calculate percentages for each IUCN status
    percentages = pivot_counts.copy()
    for idx in percentages.index:
        total = percentages.loc[idx, 'Total']
        if total > 0:
            percentages.loc[idx, 'Intact'] = (percentages.loc[idx, 'Intact'] / total) * 100
            percentages.loc[idx, 'Degraded'] = (percentages.loc[idx, 'Degraded'] / total) * 100
    
    # Define vibrant colors if not provided
    if colors is None:
        colors = {
            'Intact': '#358600',  # Bright green
            'Degraded': '#C08552'  # Vibrant orange/red
        }
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each status (in reverse order for proper stacking)
    for status in reversed(valid_statuses):
        fig.add_trace(go.Bar(
            y=percentages.index,
            x=percentages[status],
            name=status,
            
            orientation='h',
            marker=dict(
                color=colors.get(status, '#999999'),
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.1f}%" if val > 5 else "" for val in percentages[status]],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white', size=12, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>' +
                          f'{status}: %{{x:.1f}}%<br>' +
                          'Count: %{customdata}<extra></extra>',
            customdata=pivot_counts[status]
        ))
    
    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis={
            'title': "<b>Percentage of Bird Sightings (%)</b>",
            
            'tickfont': {'size': 14},
            'range': [0, 100],
            'gridcolor': 'rgba(0, 0, 0, 0)',
        },
        yaxis={
            'title': "<b>IUCN Conservation Status</b>",
            
            'tickfont': {'size': 14},
            'categoryorder': 'array',
            'categoryarray': iucn_order,
        },
        barmode='stack',
        
        margin=dict(l=10, r=20, t=80, b=20),
        height=600,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
    )
   
  
   
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=-0.5,
        x1=100,
        y1=-0.5,
        line=dict(color="#CCCCCC", width=2)
    )
    
    return fig

# ...existing code...

def create_tree_volume_comparison(df, common_name_col='Common_Name', height_col='Height_m', 
                               diameter_col='Diameter_cm', status_col='Status', 
                               top_n=15):
    """
    Create interactive treemaps comparing tree species volume between intact and degraded areas.
    The relative size of the subplot areas is fixed, but the rectangles within each treemap
    are proportional to the volume of the species within that area.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tree data
    common_name_col : str
        Column name for tree common name
    height_col : str
        Column name for tree height in meters
    diameter_col : str
        Column name for tree diameter in centimeters
    status_col : str
        Column name for tree status (Intact/Degraded)
    top_n : int
        Number of top species to display in each treemap

    Returns:
    --------
    dict
        Dictionary containing the Plotly figure, summary statistics, and top species data
    """
    
    # Check if we have the required columns
    required_cols = [common_name_col, height_col, diameter_col, status_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    result = {
        "figure": None,
        "statistics": {},
        "top_species": {"intact": [], "degraded": []},
        "error": None,
    }
    
    if missing_cols:
        result["error"] = f"Missing required columns: {missing_cols}"
        return result
    
    try:
        # Make a copy to avoid modifying the original dataframe
        df_trees = df.copy()
        
        # Convert height and diameter to numeric, handling errors
        df_trees[height_col] = pd.to_numeric(df_trees[height_col], errors='coerce')
        df_trees[diameter_col] = pd.to_numeric(df_trees[diameter_col], errors='coerce')
        
        # Drop rows with missing values in key columns
        df_trees = df_trees.dropna(subset=[common_name_col, height_col, diameter_col, status_col])
        
        # Fill any missing Common_Name values with 'Unknown'
        df_trees[common_name_col] = df_trees[common_name_col].fillna('Unknown')
        
        # Calculate tree volume (approximated as a cylinder)
        # Volume = π * (radius_m)² * height_m
        # Radius_m = (diameter_cm / 100) / 2
        df_trees['Volume_m3'] = (np.pi * ((df_trees[diameter_col] / 100 / 2)**2) * 
                                  df_trees[height_col])
        
        # Filter for valid statuses
        valid_statuses = ['Intact', 'Degraded']
        df_trees = df_trees[df_trees[status_col].isin(valid_statuses)]
        
        # Group by species and status to get total volume
        volume_by_species = df_trees.groupby([common_name_col, status_col])['Volume_m3'].sum().reset_index()
        
        # Create separate dataframes for intact and degraded
        intact_df = volume_by_species[volume_by_species[status_col] == 'Intact']
        degraded_df = volume_by_species[volume_by_species[status_col] == 'Degraded']
        
        # Sort by volume and limit to top N species for readability
        intact_df = intact_df.sort_values('Volume_m3', ascending=False).head(top_n)
        degraded_df = degraded_df.sort_values('Volume_m3', ascending=False).head(top_n)

        # Calculate total volumes for potential proportional sizing (though not directly applied to subplot size)
        total_intact_volume = intact_df['Volume_m3'].sum()
        total_degraded_volume = degraded_df['Volume_m3'].sum()
        grand_total_volume = total_intact_volume + total_degraded_volume

        # Determine column widths - Note: This sets subplot area width, not data-driven treemap size.
        # For true proportional visual size based on total volume, separate figures or different plot types might be needed.
       
        if grand_total_volume > 0:
             intact_width = total_intact_volume / grand_total_volume
             degraded_width = total_degraded_volume / grand_total_volume
        else:
             intact_width = 0.5
             degraded_width = 0.5

        # Create a figure with two treemaps side by side
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[intact_width, degraded_width], # Sets relative width of subplot areas
            subplot_titles=(
                f'Tree Species Volume in Intact Areas (Top {len(intact_df)})',
                f'Tree Species Volume in Degraded Areas (Top {len(degraded_df)})'
            ),
            specs=[[{"type": "treemap"}, {"type": "treemap"}]],
            horizontal_spacing=0.000001,
            
        )
        
        # Define color scales for the two treemaps
        intact_colorscale = [
            [0, '#c5e8c5'],          # Light green
            [0.5, '#5cb85c'],         # Medium green
            [1, '#2e7d32']           # Dark green
        ]
        
        degraded_colorscale = [
            [0, '#ffe0b2'],          # Light orange/tan
            [0.5, '#ff9800'],         # Medium orange
            [1, '#e65100']           # Dark orange/brown
        ]
        
        # Create treemap for intact trees if data exists
        if not intact_df.empty:
            # Format hover text
            intact_df['text'] = intact_df.apply(lambda row: 
                    f"{row[common_name_col]}<br>Volume: {row['Volume_m3']:.2f} m³", axis=1)
            
            # Calculate percentages of total volume for labeling
            intact_df['percent'] = (intact_df['Volume_m3'] / total_intact_volume * 100).round(1)
            
            # Add treemap trace for intact trees
            fig.add_trace(go.Treemap(
                labels=intact_df[common_name_col],
                values=intact_df['Volume_m3'],
                parents=[""] * len(intact_df),
                root_color="lightblue",
                text=intact_df['text'],
                hovertemplate='<b>%{label}</b><br>Volume: %{value:.2f} m³<br>Percentage: %{percentRoot:.1f}%<extra></extra>',
                texttemplate='<b>%{label}</b><br>',
                marker=dict(
                    colorscale=intact_colorscale,
                    colors=intact_df['Volume_m3'],
                    line=dict(width=3, color='white'),
                    
                ),
                
                branchvalues='total',
                textposition="middle center",
                name="Intact",
                tiling=dict(pad=0)
            ), row=1, col=1)
        
        # Create treemap for degraded trees if data exists
        if not degraded_df.empty:
            # Format hover text
            degraded_df['text'] = degraded_df.apply(lambda row: 
                    f"{row[common_name_col]}<br>Volume: {row['Volume_m3']:.2f} m³", axis=1)
            
            # Calculate percentages of total volume for labeling
            degraded_df['percent'] = (degraded_df['Volume_m3'] / total_degraded_volume * 100).round(1)
            
            # Add treemap trace for degraded trees
            fig.add_trace(go.Treemap(
                labels=degraded_df[common_name_col],
                values=degraded_df['Volume_m3'],
                parents=[""] * len(degraded_df),
                text=degraded_df['text'],
                hovertemplate='<b>%{label}</b><br>Volume: %{value:.2f} m³<br>Percentage: %{percentRoot:.1f}%<extra></extra>',
                texttemplate='<b>%{label}</b><br>',
                marker=dict(
                    colorscale=degraded_colorscale,
                    colors=degraded_df['Volume_m3'],
                    line=dict(width=3, color='white')
                ),
                branchvalues='total',
                textposition="middle center",
                name="Degraded",
                tiling=dict(pad=0)
            ), row=1, col=2)
        
        # Calculate summary statistics
        # Recalculate totals based on the top_n species shown
        total_intact_volume_top_n = intact_df['Volume_m3'].sum()
        total_degraded_volume_top_n = degraded_df['Volume_m3'].sum()
        
        # Calculate percent difference using top N totals
        if total_intact_volume_top_n > 0 and total_degraded_volume_top_n > 0:
            pct_diff = ((total_intact_volume_top_n - total_degraded_volume_top_n) / 
                       ((total_intact_volume_top_n + total_degraded_volume_top_n) / 2)) * 100
            pct_diff_abs = abs(pct_diff)
            diff_direction = 'higher in intact areas' if pct_diff > 0 else 'higher in degraded areas'
            title_pct_diff = round(pct_diff_abs, 1)
            title_direction = 'more' if pct_diff > 0 else 'less'
            title_area = 'Intact areas' if pct_diff > 0 else 'Degraded areas'
            title_color = 'green' if pct_diff > 0 else 'brown'

        elif total_intact_volume_top_n > 0:
             pct_diff_abs = 100
             diff_direction = 'higher in intact areas'
             title_pct_diff = 100
             title_direction = 'more'
             title_area = 'Intact areas'
             title_color = 'green'
        elif total_degraded_volume_top_n > 0:
             pct_diff_abs = 100
             diff_direction = 'higher in degraded areas'
             title_pct_diff = 100
             title_direction = 'more'
             title_area = 'Degraded areas'
             title_color = 'brown'
        else:
            pct_diff_abs = 0
            diff_direction = 'N/A'
            title_pct_diff = 0
            title_direction = ''
            title_area = 'areas'
            title_color = 'black'


        # Prepare summary statistics for return
        result["statistics"] = {
            "total_intact_volume": round(total_intact_volume_top_n, 2),
            "total_degraded_volume": round(total_degraded_volume_top_n, 2),
            "percent_difference": round(pct_diff_abs, 1),
            "difference_direction": diff_direction
        }
        
        # Get top 5 species for each area
        for i, row in intact_df.head(5).iterrows():
            result["top_species"]["intact"].append({
                "name": row[common_name_col],
                "volume": round(row['Volume_m3'], 2)
            })
            
        for i, row in degraded_df.head(5).iterrows():
            result["top_species"]["degraded"].append({
                "name": row[common_name_col],
                "volume": round(row['Volume_m3'], 2)
            })
        
        # Update layout
        fig_title = f'There is {title_pct_diff}% {title_direction} tree volume in <span style="color:{title_color};">{title_area}</span>' if title_pct_diff > 0 else 'Tree Volume Comparison: Intact vs Degraded Areas'

        fig.update_layout(
            title={
                'text': fig_title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'color': 'white', 'family': 'open sans, sans-serif'}
            },
            margin=dict(t=90, l=10, r=10, b=50), # Increased bottom margin for annotation
            height=700,
            font=dict(
                family="open sans, sans-serif",
                size=13,
                color="white"
            ),
            
            
            annotations=[
                dict(
                    x=0.5, y=-0.07, # Adjusted y position
                    xref='paper', yref='paper',
                    text=f'Note: Tree volume calculated using formula: π × (diameter_cm/200)² × height_m<br>' +
                         f'Showing top {top_n} species by volume in each area.',
                    showarrow=False,
                    font=dict(size=11, color='gray', style='italic'),
                    align='center'
                )
            ]
        )
        
        # Store the figure in the result dictionary
        result["figure"] = fig
        
        return result
    
    except Exception as e:
        result["error"] = str(e)
        return result


    



def create_plotly_waffle_chart(birds_df):
    """
    Creates a Plotly waffle chart visualizing bird sightings by status (Intact vs. Degraded).

    Args:
        birds_df (pd.DataFrame): DataFrame containing bird sighting data with a 'Status' column.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object, or None if data is invalid.
    """
    try:
        # Check for required columns
        if 'Status' not in birds_df.columns:
            print(f"Error: Required column 'Status' not found in the dataset.")
            print(f"Available columns: {list(birds_df.columns)}")
            return None

        # --- Data Processing (similar to Matplotlib version) ---
        # Fill missing values with 'Unknown'
        birds_df_copy = birds_df.copy() # Work on a copy to avoid modifying original df
        birds_df_copy['Status'] = birds_df_copy['Status'].fillna('Unknown')

        # Filter for just Intact and Degraded statuses
        valid_statuses = ['Intact', 'Degraded']
        birds_df_copy = birds_df_copy[birds_df_copy['Status'].isin(valid_statuses)]

        # Count sightings by status
        status_counts = birds_df_copy['Status'].value_counts()

        # Ensure both statuses exist
        for status in valid_statuses:
            if status not in status_counts.index:
                status_counts[status] = 0

        # Get total sightings for each area
        intact_total = status_counts.get('Intact', 0)
        degraded_total = status_counts.get('Degraded', 0)
        total_sightings = intact_total + degraded_total

        if total_sightings == 0:
            print("No sightings found for Intact or Degraded status. Cannot create chart.")
            return None

        # Calculate proportions
        intact_proportion = intact_total / total_sightings
        degraded_proportion = degraded_total / total_sightings

        print(f"\nBird sightings by area:")
        print(f"Intact areas: {intact_total} sightings ({intact_proportion*100:.1f}% of total)")
        print(f"Degraded areas: {degraded_total} sightings ({degraded_proportion*100:.1f}% of total)")
        print(f"Total sightings: {total_sightings}")

        # --- Prepare data for Plotly Waffle ---
        total_squares = 100
        intact_squares = int(round(intact_proportion * total_squares))
        degraded_squares = total_squares - intact_squares

        # Create lists for x, y coordinates and status labels
        x_coords = []
        y_coords = []
        status_labels = []
        colors = []
        color_map = {'Intact': '#358600', 'Degraded': '#C08552'}

        waffle_width = 10
        waffle_height = 10
        count = 0
        for r in range(waffle_height):
            for c in range(waffle_width):
                x_coords.append(c)
                y_coords.append(waffle_height - 1 - r) # Plotly y-axis goes upwards
                if count < intact_squares:
                    status_labels.append('Intact')
                    colors.append(color_map['Intact'])
                else:
                    status_labels.append('Degraded')
                    colors.append(color_map['Degraded'])
                count += 1

        # --- Create Plotly Figure ---
        fig = go.Figure()

        # Add squares as scatter points
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                color=colors,
                size=48, # Adjust size as needed
                symbol='square',
                line=dict(width=0., color='white') # Grid lines via marker outline
            ),
            
            showlegend=False # We will create a custom legend annotation
        ))

        # --- Configure Layout ---
        fig.update_layout(
            title=dict(
                text='<b>Bird Populations Show <span style="color:#D2B48C;"> Only 30% Activity in Degraded Areas</span></b>',
                font=dict(size=20),
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, waffle_width - 0.5] # Ensure squares are centered
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, waffle_height - 0.5],
                scaleanchor="x", # Make squares square
                scaleratio=1
            ),
            width=700, # Adjust figure size
            height=700,
            plot_bgcolor='black',
            margin=dict(t=50, b=50, l=54, r=54), # Adjust margins for title/legend
            # Add border
            xaxis_showline=False, yaxis_showline=False,
            
        )

        # Add custom legend-like annotations
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=-0.05, # Position below the chart
            showarrow=False,
            text=(f"<span style='color:forestgreen; font-size: 15px;'>■</span> Intact: {intact_total} sightings ({intact_proportion*100:.1f}%)   "
                  f"<span style='color:#D2B48C; font-size: 15px;'>■</span> Degraded: {degraded_total} sightings ({degraded_proportion*100:.1f}%)"),
            align="center",
            font=dict(size=14)
        )

        # Add explanation text (optional, similar to matplotlib version)
        # fig.add_annotation(
        #     xref="paper", yref="paper",
        #     x=0.5, y=-0.18,
        #     showarrow=False,
        #     text="<i>Each square represents 1% of all bird sightings.</i>",
        #     align="center",
        #     font=dict(size=12)
        # )

        return fig

    except Exception as e:
        print(f"An error occurred while creating the Plotly waffle chart: {e}")
        return None
    


def create_bird_arrival_duration_plot(df_birds):
    """
    Creates a scatter plot showing mean bird arrival time vs. mean duration,
    grouping birds by proximity in time and duration.

    Args:
        df_birds (pd.DataFrame): DataFrame containing bird sighting data with
                                 'Latin_Name', 'Date', 'Time_Stamp', 'Duration_sec'.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object containing the plot,
                                  or None if an error occurs.
    """
    try:
        df = df_birds.copy() # Work on a copy

        # --- Data Preprocessing ---
        # Standardize date format (assuming input format is 'd/m/yyyy')
        def standardize_date_format(date_str):
            try:
                # Handle potential non-string or incorrectly formatted dates
                if not isinstance(date_str, str) or '/' not in date_str:
                    return pd.NaT # Return Not a Time for invalid formats
                day, month, year = date_str.split('/')
                # Pad day and month if necessary (though int() handles it)
                return f"{int(day):02d}/{int(month):02d}/{year}"
            except Exception:
                return pd.NaT # Return NaT on any parsing error

        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(standardize_date_format)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce') # Convert to datetime

        # Convert Time_Stamp to datetime, handling potential errors
        if 'Time_Stamp' in df.columns:
            df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'], format='%H:%M:%S', errors='coerce')
        else:
            print("Error: 'Time_Stamp' column not found.")
            return None

        # Calculate Time_Seconds, handling NaT from previous step
        df['Time_Seconds'] = df['Time_Stamp'].dt.hour * 3600 + df['Time_Stamp'].dt.minute * 60 + df['Time_Stamp'].dt.second
        df['Time_Seconds'] = df['Time_Seconds'].fillna(-1).astype(int) # Fill NaT times with -1 or handle appropriately

        # Round Duration to nearest second
        if 'Duration_sec' in df.columns:
            df['Duration_sec'] = pd.to_numeric(df['Duration_sec'], errors='coerce').round().fillna(-1).astype(int) # Handle non-numeric/NaN
        else:
            print("Error: 'Duration_sec' column not found.")
            return None

        # Drop rows where essential time/duration info is missing
        df = df[(df['Time_Seconds'] >= 0) & (df['Duration_sec'] >= 0)].copy()
        if df.empty:
            print("Error: No valid data after cleaning Time_Stamp and Duration_sec.")
            return None

        # --- Grouping Logic ---
        time_tolerance_seconds = 3 * 60
        duration_tolerance_seconds = 5

        # Function to assign IDs based on time and duration proximity
        def assign_ids(df_to_process, time_tolerance, duration_tolerance):
            df_to_process = df_to_process.sort_values(by='Time_Seconds').reset_index(drop=True) # Sort for efficiency
            df_to_process['ID'] = -1 # Initialize with -1 instead of None for easier type handling
            next_id = 1
            processed = [False] * len(df_to_process)

            for i in range(len(df_to_process)):
                if not processed[i]:
                    current_id = next_id
                    df_to_process.loc[i, 'ID'] = current_id
                    processed[i] = True
                    time_i = df_to_process.loc[i, 'Time_Seconds']
                    duration_i = df_to_process.loc[i, 'Duration_sec']

                    # Look for neighbors within the tolerance window
                    for j in range(i + 1, len(df_to_process)):
                        if not processed[j]:
                            time_j = df_to_process.loc[j, 'Time_Seconds']
                            duration_j = df_to_process.loc[j, 'Duration_sec']

                            # Optimization: If time_j is already too far, break inner loop
                            if time_j - time_i > time_tolerance:
                                break

                            if (abs(time_i - time_j) <= time_tolerance) and \
                               (abs(duration_i - duration_j) <= duration_tolerance):
                                df_to_process.loc[j, 'ID'] = current_id
                                processed[j] = True
                    next_id += 1
            return df_to_process

        # Apply the ID assignment function
        df = assign_ids(df, time_tolerance_seconds, duration_tolerance_seconds)

        # --- Summarization ---
        # Group by Latin_Name and calculate mean, keeping the first assigned ID
        if 'Latin_Name' not in df.columns:
            print("Error: 'Latin_Name' column not found.")
            return None

        summary_df = df.groupby('Latin_Name').agg(
            Time_Seconds=('Time_Seconds', 'mean'),
            Duration_sec=('Duration_sec', 'mean'),
            ID=('ID', 'first')  # Keep the first assigned ID for the group
        ).reset_index()

        # Convert mean seconds back to rounded-up HH:MM format for display (optional here, done later for axis)
        def seconds_to_time_rounded_up(seconds):
            if pd.isna(seconds): return "00:00"
            seconds = int(math.ceil(seconds / 60.0)) * 60  # round up to full minute
            time = timedelta(seconds=seconds)
            hours, remainder = divmod(time.seconds, 3600)
            minutes = remainder // 60
            return f"{hours:02d}:{minutes:02d}"

        # summary_df['Mean_Time_Stamp'] = summary_df['Time_Seconds'].apply(seconds_to_time_rounded_up) # Not strictly needed for plot coords
        summary_df['Mean_Duration_sec'] = summary_df['Duration_sec'].round().astype(int)
        summary_df['Mean_Time_Seconds'] = summary_df['Time_Seconds'] # Keep mean seconds for plotting

        # --- Filtering for Groups with Multiple Species ---
        id_counts = summary_df['ID'].value_counts()
        unique_ids = id_counts[id_counts == 1].index.tolist() # IDs that appear only once
        filtered_summary_df = summary_df[~summary_df['ID'].isin(unique_ids)] # Keep only IDs with >1 species

        if filtered_summary_df.empty:
            print("Warning: No groups found with more than one species after filtering.")
            # Decide if you want to plot single-species groups or return None
            # Plotting single species groups might be messy. Returning None for now.
            return None

        # --- Grouping by ID for Plotting ---
        grouped_df = (
            filtered_summary_df.groupby('ID')
            .agg(
                Mean_Time_Seconds=('Mean_Time_Seconds', 'mean'),
                Mean_Duration_sec=('Mean_Duration_sec', 'mean'),
                Latin_Names=('Latin_Name', list),
            )
            .reset_index()
        )

        # --- Plotting ---
        plt.figure(figsize=(14, 8)) # Adjusted figure size
        sns.scatterplot(
            data=grouped_df,
            x='Mean_Time_Seconds',
            y='Mean_Duration_sec',
            hue='ID',
            palette='tab20', # Using a palette suitable for many categories
            s=200,          # Increased size slightly
            alpha=0.8,
            legend=False    # No legend needed as text labels are added
        )

        # Add text labels for species within each group
        for index, row in grouped_df.iterrows():
            species_text = '\n'.join(row['Latin_Names'])
            plt.text(
                row['Mean_Time_Seconds'],
                row['Mean_Duration_sec'],
                species_text,
                ha='center',
                va='center', # Adjust vertical alignment if needed
                color='black',
                fontsize=9, # Adjusted fontsize
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.6, ec='none') # Add background to text
            )

        # Format x-axis ticks as HH:MM
        def format_time_ticks(seconds, pos=None): # pos argument needed for FuncFormatter
            """Formats seconds since midnight to HH:MM."""
            if seconds < 0: return "" # Avoid formatting negative ticks if axis extends
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours:02d}:{minutes:02d}"

        # Use FuncFormatter for better control over tick labels
        from matplotlib.ticker import FuncFormatter
        ax = plt.gca() # Get current axes
        ax.xaxis.set_major_formatter(FuncFormatter(format_time_ticks))

        plt.xlabel('Mean Arrival Time (HH:MM)')
        plt.ylabel('Mean Duration (seconds)')
        plt.title('Bird Species Groups by Arrival Time and Call Duration')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout() # Adjust layout

        # Return the Matplotlib figure object
        fig = plt.gcf()
        # Do NOT call plt.show() here
        return fig

    except Exception as e:
        print(f"An error occurred in create_bird_arrival_duration_plot: {e}")
        # Optionally close the figure if created during error
        try:
            plt.close()
        except:
            pass # Ignore if no figure exists
        return None

# --- Example Usage (keep commented out or remove from final script) ---
# if __name__ == '__main__':
#     # Load your data
#     try:
#         df_birds_data = pd.read_csv('birds_entact.csv')
#         print("Data loaded successfully.")
#
#         # Create the plot
#         bird_arrival_fig = create_bird_arrival_duration_plot(df_birds_data)
#
#         # In a real script, you wouldn't show it here, but for testing:
#         if bird_arrival_fig:
#             print("Figure created. Displaying for testing...")
#             plt.show()
#         else:
#             print("Figure creation failed.")
#
#     except FileNotFoundError:
#         print("Error: birds_entact.csv not found.")
#     except Exception as main_e:
#         print(f"An error occurred during example usage: {main_e}")











from plotly.colors import hex_to_rgb


def create_sankey_diagram_reversed(df_birds, status_col='Status', iucn_col='IUCN_Status', name_col='Common_Name'):
    """
    Creates a Sankey diagram showing the flow of unique bird species counts
    FROM IUCN conservation categories TO Intact/Degraded status, using a single DataFrame.
    Includes a horizontal legend at the bottom for IUCN status colors. Nodes are unlabeled.

    Args:
        df_birds (pd.DataFrame): DataFrame containing bird data with Status, IUCN, and species name columns.
        status_col (str): Column name for habitat status (e.g., 'Intact', 'Degraded'). Defaults to 'Status'.
        iucn_col (str): Column name for IUCN status. Defaults to 'IUCN_Status'.
        name_col (str): Column name for the bird species identifier (e.g., Common_Name). Defaults to 'Common_Name'.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object containing the Sankey diagram,
                                     or None if an error occurs.
    """
    try:
        # --- Input Validation ---
        if not isinstance(df_birds, pd.DataFrame) or df_birds.empty:
            print("Error: Invalid or empty DataFrame provided for 'df_birds'.")
            return None

        required_cols = [status_col, iucn_col, name_col]
        if not all(col in df_birds.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_birds.columns]
            print(f"Error: Missing columns in 'df_birds': {missing}")
            return None

        # --- Data Preparation ---
        df_processed = df_birds.copy()
        df_processed[status_col] = df_processed[status_col].fillna('Unknown_Status')
        df_processed[iucn_col] = df_processed[iucn_col].fillna('Unknown_IUCN')

        birds_entact = df_processed[df_processed[status_col] == 'Intact']
        birds_degraded = df_processed[df_processed[status_col] == 'Degraded']

        if birds_entact.empty and birds_degraded.empty:
             print("Error: No data found for 'Intact' or 'Degraded' status.")
             return None

        # --- Data Aggregation ---
        entact_counts = birds_entact.groupby(iucn_col)[name_col].nunique()
        degraded_counts = birds_degraded.groupby(iucn_col)[name_col].nunique()

        categories = ['Least Concern', 'Near Threatened', 'Vulnerable', 'Endangered', 'Critically Endangered', 'Unknown_IUCN']
        present_categories_entact = entact_counts.index.unique().tolist()
        present_categories_degraded = degraded_counts.index.unique().tolist()
        all_present_categories = sorted(list(set(present_categories_entact + present_categories_degraded)),
                                        key=lambda x: categories.index(x) if x in categories else 99)

        entact_counts_values = [entact_counts.get(cat, 0) for cat in all_present_categories]
        degraded_counts_values = [degraded_counts.get(cat, 0) for cat in all_present_categories]

        # --- Sankey Configuration ---
        display_categories = [cat.replace('Unknown_IUCN', 'Unknown') for cat in all_present_categories]
        original_labels = ['Intact', 'Degraded'] + display_categories
        plot_labels = [""] * len(original_labels)
        label_indices = {label: i for i, label in enumerate(original_labels)}

        original_source_indices = [label_indices['Intact']] * len(all_present_categories) + \
                                  [label_indices['Degraded']] * len(all_present_categories)
        original_target_indices = [label_indices[cat] for cat in display_categories] * 2
        values = entact_counts_values + degraded_counts_values

        # --- Define Node Colors ---
        node_color_map = {
            'Intact': '#358600',
            'Degraded': '#C08552',
            'Least Concern': '#60C659',
            'Near Threatened': '#CCE226',
            'Vulnerable': '#F9E814',
            'Endangered': '#FC7F3F',
            'Critically Endangered': '#D81E05',
            'Unknown': '#CCCCCC'
        }
        node_colors = [node_color_map.get(label, '#888888') for label in original_labels]

        # --- Define Link Colors based on NEW source (IUCN category) ---
        def to_rgba(color, alpha=0.8):
            if color.startswith('#'):
                rgb = hex_to_rgb(color)
                return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
            return f'rgba(128, 128, 128, {alpha})'

        reversed_link_colors_map = {
            cat: to_rgba(node_color_map.get(cat, '#CCCCCC')) for cat in display_categories
        }
        reversed_link_colors = [reversed_link_colors_map.get(original_labels[idx], to_rgba('#CCCCCC')) for idx in original_target_indices]

        # --- Filter out zero-value links ---
        non_zero_indices = [i for i, v in enumerate(values) if v > 0]
        final_source_indices = [original_target_indices[i] for i in non_zero_indices]
        final_target_indices = [original_source_indices[i] for i in non_zero_indices]
        final_values = [values[i] for i in non_zero_indices]
        final_link_colors = [reversed_link_colors[i] for i in non_zero_indices]

        # --- Create Figure ---
        fig = go.Figure()

        # Add the Sankey trace
        fig.add_trace(go.Sankey(
            arrangement='perpendicular',
            node=dict(
                pad=35,
                thickness=15,
                line=dict(color="white", width=1.5),
                label=plot_labels,
                color=node_colors,
                hovertemplate='Node: %{label}<extra></extra>'
            ),
            link=dict(
                source=final_source_indices,
                target=final_target_indices,
                value=final_values,
                color=final_link_colors,
                hovertemplate='From %{source.label} to %{target.label}:<br><b>%{value}</b> unique species<extra></extra>',
                customdata=original_labels
            )
        ))

        # --- Add Dummy Traces for IUCN Legend ---
        present_iucn_for_legend = [cat for cat in categories if cat.replace('Unknown_IUCN', 'Unknown') in display_categories]

        for category in present_iucn_for_legend:
            display_name = category.replace('Unknown_IUCN', 'Unknown')
            color = node_color_map.get(display_name, '#CCCCCC')
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color=color, size=10),
                name=display_name,
                showlegend=True
            ))

        # --- Update Layout ---
        fig.update_layout(
             title=dict(
            text="<b><i>The flow of birds</i></b>",
            font=dict(size=20, color="white"),
            x=0.5,                 # Set this to 0.5 to center the title
            xanchor="center",      # This ensures the title is centered
            y=0.95                 # Adjust vertical position as needed
        ),
            font=dict(
                size=12,
                color="white"
            ),
            height=800, # Keep height or adjust as needed
            margin=dict(t=50, b=100, l=50, r=50), # Increase bottom margin for legend
            showlegend=True,
            plot_bgcolor='black',
            legend=dict(
                title_text='IUCN Status (left side)',
                orientation="h",      # Horizontal legend
                yanchor="bottom",
                y=-0.12,              # Position below the plot area (adjust as needed)
                xanchor="center",
                x=0.5                 # Center horizontally
            ),
            xaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False
            )
        )


        return fig

    except Exception as e:
        print(f"An error occurred while creating the reversed Sankey diagram: {e}")
        # import traceback
        # print(traceback.format_exc())
        return None

# ... (rest of the file) ...