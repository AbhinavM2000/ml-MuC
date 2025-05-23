import re
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objects as go
import numpy as np
import re
import pandas as pd

def process_file(input_file, output_file):
    def process_line(line):
        line = re.sub(r'[^\w\s.,+-]', ',', line)
        line = ' '.join(line.split())
        return line

    def remove_extra_commas(lines):
        cleaned_lines = []
        for line in lines:
            if line[:2] == '-,':
                line = line[2:]
            parts = line.split(',')
            cleaned_parts = [part.strip() for part in parts if part.strip()]
            cleaned_line = ','.join(cleaned_parts)
            cleaned_lines.append(cleaned_line)
        return cleaned_lines

    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = [process_line(line) for line in lines]
    cleaned_lines = remove_extra_commas(processed_lines)

    with open(output_file, 'w') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    # Print number of lines in final_result.txt
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(len(lines))

    # Open final_result.txt and if any line ends with +n, where n greater than 3, calculate sum of all such n = count. Then print % = count / (sum of all n including those less than 3)
    with open(output_file, 'r') as f:
        lines = f.readlines()

    sum_n = 0
    count = 0
    for line in lines:
        match = re.search(r'\+(\d+)$', line.strip())  # Check if the line ends with '+n'
        if match:
            n = int(match.group(1))
            if n > 3:  # Ensure n is greater than 3
                sum_n += n
                count += 1

    total_sum = sum_n
    for line in lines:
        match = re.search(r'\+(\d+)$', line.strip())  # Check if the line ends with '+n'
        if match:
            n = int(match.group(1))
            total_sum += n


    #avoid division by zero
    if total_sum == 0:
        print("No points excluded.")
    else:
        percentage = (count / total_sum) * 100
        print(f"{percentage:.2f}% of points(+4 and above decays) excluded due to difficulties in parsing.")

    with open(output_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        skip_next = 0
        for line in lines:
            if skip_next > 0:
                skip_next -= 1
                continue
            match = re.search(r'\+(\d+)$', line.strip())  # Check if the line ends with '+n'
            if match:
                n = int(match.group(1))
                if n > 3:  # Ensure n is greater than 2
                    skip_next = n
                    continue
            f.write(line)

    # Open final_result.txt and if any line ends with +2, duplicate that line below it
    with open(output_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line)
            if line.strip().endswith('+2'):
                f.write(line)

    # Print number of lines in final_result.txt
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(len(lines))

    # Open final_result.txt, if any line repeats more than once, swap its position with the line below it
    def swap_repeated_lines(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        prev_line = None
        for i in range(len(lines)):
            if prev_line == lines[i]:
                lines[i], lines[i + 1] = lines[i + 1], lines[i]
                prev_line = None
            else:
                prev_line = lines[i]

        with open(file_path, 'w') as file:
            file.writelines(lines)

    # Usage example
    file_path = output_file
    swap_repeated_lines(file_path)

    # Print number of lines in final_result.txt
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(len(lines))

    # Duplicate lines ending with +3 two times below them
    with open(output_file, 'r') as f:
        lines = f.readlines()

    with open(output_file + "_modified.txt", 'w') as f:
        for line in lines:
            f.write(line)
            if line.strip().endswith('+3'):
                f.write(line)
                f.write(line)

    # Swap lines i+1 and i+4 if line i ends with +3 and start scanning from i+6
    with open(output_file + "_modified.txt", 'r') as f:
        lines = f.readlines()

    with open(output_file + "_final.txt", 'w') as f:
        skip_next = 0
        for i in range(len(lines)):
            if skip_next > 0:
                skip_next -= 1
                continue
            f.write(lines[i])
            if lines[i].strip().endswith('+3'):
                if i + 4 < len(lines):  # Make sure there are enough lines to swap
                    lines[i + 1], lines[i + 4] = lines[i + 4], lines[i + 1]
                    skip_next = 5
                else:
                    f.write("Error: Not enough lines to perform swap.\n")

    with open(output_file, "r") as file:
        lines = file.readlines()

    combined_lines = []
    for i in range(0, len(lines), 2):
        combined_lines.append(lines[i].strip() + "," + lines[i + 1].strip())

    with open(output_file, "w") as file:
        file.write("\n".join(combined_lines))

    # Print number of lines in final_result.txt
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(len(lines))

    # Open final_result.txt and load into a pandas dataframe
    df = pd.read_csv(output_file, header=None)

    # Name the columns: id, cellId0, cellId1, energy, position (x,y,z), nMCParticles MC contribution: prim. PDG, energy_part, time, length, sec. PDG and stepPosition (x,y,z)
    df.columns = ['id', 'cellId0', 'cellId1', 'energy', 'position x', 'position y', 'position z', 'nMCParticles',
                  'mc contri prim. PDG', 'energy_part', 'time', 'length', 'sec_PDG', 'stepPosition x',
                  'stepPosition y', 'stepPosition z']

    # Convert all the columns to the numeric type
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

# Example usage:
# Assuming 'data.txt' is the input file and 'final_result.txt' is the output file


# Create histogram trace
def plot_histogram(data, x_label='Energy', y_label='Count (log scale)', nbins=20, color='royalblue', title='Energy Distribution', plot_bgcolor='rgb(51, 51, 51)', paper_bgcolor='rgb(51, 51, 51)', font_color='white'):
    # Create histogram trace
    histogram = go.Histogram(
        x=data,
        nbinsx=nbins,
        marker=dict(color=color)
    )

    # Create layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_label, color=font_color),
        yaxis=dict(title=y_label, type='log', color=font_color),
        bargap=0.05,
        bargroupgap=0.1,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=dict(color=font_color),
        autosize=True,
    )

    # Create figure
    fig = go.Figure(data=[histogram], layout=layout)

    # Update color for dark theme
    fig.update_layout(
        xaxis=dict(linecolor=font_color),
        yaxis=dict(linecolor=font_color),
        bargap=0.05,
        bargroupgap=0.1,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=dict(color=font_color)
    )

    # Show plot
    fig.show()

# Example usage:
# Assuming df['energy'] contains your energy data
#plot_histogram(df['energy'])


import plotly.graph_objects as go
import numpy as np

def plot_unrolled_and_reconstructed(df, column_name, angle):
    # Extract data
    x = df['position x']
    y = df['position y']
    z = df['position z']
    c = df['time']

    # Convert cylindrical coordinates to rectangular coordinates

    # print points with lowest z coordinate value
    min_z = df[df['position z'] == df['position z'].min()]
    O = [0, 0, min_z['position z'].values[0]]
    A = [min_z['position x'].values[0], min_z['position y'].values[0], min_z['position z'].values[0]]

    # find angle between OA and every other point, check cross product of OA and OB, if positive angle = angle otherwise angle = -angle
    def angle_between_points(O, A, B):
        OA = np.array(A) - np.array(O)
        OB = np.array(B) - np.array(O)
        angle = np.arccos(np.dot(OA, OB) / (np.linalg.norm(OA) * np.linalg.norm(OB)))
        cross_product = np.cross(OA, OB)
        if cross_product[2] > 0:
            return angle
        else:
            return -angle

    angles = []
    zvalues = []
    for i in range(len(df)):
        B = [df['position x'].iloc[i], df['position y'].iloc[i], -2203.0]
        if (B == O):
            continue
        angles.append(angle_between_points(O, A, B))
        zvalues.append(df['position z'].iloc[i])

    # For the z-coordinate, it remains the same as the z of the cylinder
    z_rect = z

    # Create scatter plot trace for unrolled plot
    scatter_unrolled = go.Scatter(
        x=angles,
        y=z_rect,
        mode='markers',
        marker=dict(
            size=8,
            color=c,
            colorscale='Rainbow',
            colorbar=dict(title='Time'),
            opacity=0.8
        )
    )

    # Create layout for unrolled plot
    layout_unrolled = go.Layout(
        title='ECalBarrelCollection Position vs Time (Event ID:{angle}/999) UNROLLED',
        xaxis=dict(title='Theta (degrees)'),
        yaxis=dict(title='Z'),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Create figure for unrolled plot
    fig_unrolled = go.Figure(data=[scatter_unrolled], layout=layout_unrolled)

    # Save unrolled plot as PNG with resolution set to 1080p
    file_name = f"images/unrolled/plot_{column_name}_{angle}.png"
    fig_unrolled.write_image(file_name, width=1920, height=1080)


    # Calculate radius
    r = np.sqrt(df['position x'] ** 2 + df['position y'] ** 2)

    # Create scatter plot trace for reconstructed plot
    scatter_reconstructed = go.Scatter3d(
        x=np.cos(angles) * r[0],
        y=np.sin(angles) * r[0],
        z=df['position z'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['time'],
            colorscale='Rainbow',
            colorbar=dict(title='Time'),
            line=dict(color='white', width=0),
            opacity=0.8
        )
    )

    # Create layout for reconstructed plot
    layout_reconstructed = go.Layout(
        title='ECalBarrelCollection Position vs Time (Event ID:000/999) RECONSTRUCTED',
        scene=dict(
            xaxis=dict(showbackground=False, showline=True, linecolor='white', title='X', showgrid=False,
                       showticklabels=True),
            yaxis=dict(showbackground=False, showline=True, linecolor='white', title='Y', showgrid=False,
                       showticklabels=True),
            zaxis=dict(showbackground=False, showline=True, linecolor='white', title='Z', showgrid=False,
                       showticklabels=True),
            bgcolor='black'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Create figure for reconstructed plot
    #fig_reconstructed = go.Figure(data=[scatter_reconstructed], layout=layout_reconstructed)

    # Show reconstructed plot
    #fig_reconstructed.show()
    # Create figure for reconstructed plot
    #fig_reconstructed = go.Figure(data=[scatter_reconstructed], layout=layout_reconstructed)
    #file_name = f"plot_ree_{column_name}_{angle}.png"
    #fig_reconstructed.write_image(file_name)
    # Show reconstructed plot
    #fig_reconstructed.show()

# Example usage
# Assuming df is your DataFrame
#plot_unrolled_and_reconstructed(df)

import plotly.graph_objects as go
import numpy as np
import os

def plot_3d_heatmap_at_angle(df, column_name, angle, save_image=False):
    # Generate data for cylinder surface
    def cylinder(x, y, z, radius, height):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, height, 10)
        U, V = np.meshgrid(u, v)
        X = x + (radius * np.cos(U))
        Y = y + (radius * np.sin(U))
        Z = z + V
        return X, Y, Z

    # Create scatter plot trace
    scatter = go.Scatter3d(
        x=df['position x'],
        y=df['position y'],
        z=df['position z'],
        mode='markers',
        marker=dict(
            size=5,
            color=df[column_name],
            colorscale='Rainbow',
            colorbar=dict(title=column_name),
            line=dict(color='white', width=0),
            opacity=0.8
        )
    )

    # Generate cylinder surface
    cylinder_x, cylinder_y, cylinder_z = cylinder(x=0, y=0, z=-2000, radius=1500, height=4000)

    # Create cylindrical outline trace
    dark_color = 'rgb(0, 0, 0)'
    light_color = 'rgb(255, 255, 255)'
    light_color2='rgb(45, 45, 45)'

    colorscale = [
        [0, light_color],
        [0.5, dark_color],
        [1, light_color]
    ]

    cylinder_outline = go.Surface(
        x=cylinder_x,
        y=cylinder_y,
        z=cylinder_z,
        opacity=0.1,
        showscale=False,
        colorscale=colorscale
    )

    n_points = 50
    theta = np.linspace(0, 2*np.pi, n_points)
    x = 1500 * np.cos(theta)
    y = 1500 * np.sin(theta)
    z = np.linspace(-2000, 2000, n_points)

    lines_circumference = []
    for i in range(n_points):
        lines_circumference.append(go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[-2000, 2000], mode='lines', line=dict(color=light_color2)))

    # Create layout
    layout = go.Layout(
        title=f'ECalBarrelCollection Position vs {column_name} (Event ID:000/999) ORIGINAL',
        scene=dict(
            xaxis=dict(showbackground=False, showline=True, linecolor='white', title='X', showgrid=False, showticklabels=True),
            yaxis=dict(showbackground=False, showline=True, linecolor='white', title='Y', showgrid=False, showticklabels=True),
            zaxis=dict(showbackground=False, showline=True, linecolor='white', title='Z', showgrid=False, showticklabels=True),
            bgcolor='black',
            camera=dict(
                up=dict(x=0, y=1, z=0),  # Adjusted for y-axis rotation
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.5, y=0.5, z=1.5)  # Zoom out by increasing z value
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Create figure
    fig = go.Figure(layout=layout)

    # Update camera viewpoint for rotation and zoom
    eye_x = np.cos(np.deg2rad(angle))
    eye_z = np.sin(np.deg2rad(angle))
    fig.update_layout(scene_camera=dict(eye=dict(x=0.5*eye_x, y=0.5, z=1.5)))  # Adjusted for y-axis rotation and zoom

    # Append scatter plot, cylinder, and lines to figure
    fig.add_trace(scatter)
    fig.add_trace(cylinder_outline)
    fig.add_traces(lines_circumference)
    fig.update_traces(showlegend=False)

    # Save plot as image if save_image is True
    if save_image:
        if not os.path.exists("images"):
            os.makedirs("images")
        #fig.write_image(f"images/plot_{column_name}_{angle}.png", width=1200, height=800, scale=3)

    # Show plot
    #fig.show()

# Example usage:
# Assuming df is your DataFrame and 'time' is the column you want to use for heatmap

import argparse
def main():
    parser = argparse.ArgumentParser(description='Process a file and plot 3D heatmap at a specified angle.')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('output_file', help='Path to the output text file')
    parser.add_argument('column_name', help='Name of the column to be plotted')
    parser.add_argument('--angle', type=float, default=0, help='Angle for the 3D plot (default: 45)')
    parser.add_argument('--save_image', action='store_true', help='Save the plot as an image')

    args = parser.parse_args()

    # Process file
    df = process_file(args.input_file, args.output_file)
    plot_unrolled_and_reconstructed(df,args.column_name, args.angle)
    # Plot 3D heatmap
    #plot_3d_heatmap_at_angle(df, args.column_name, args.angle, save_image=args.save_image)
    #os.remove(args.output_file)

if __name__ == "__main__":
    main()
