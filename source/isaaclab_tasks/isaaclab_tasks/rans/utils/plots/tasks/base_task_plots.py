import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import numpy as np
from matplotlib.ticker import MaxNLocator
import random
from matplotlib.collections import LineCollection
import matplotlib as mpl

class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)

        # Unique for each subclass + inherit from parent class
        cls._registry = getattr(super(cls, cls), '_registry', {}).copy()

        for name, value in cls.__dict__.items():
            # If an attribute is a function and has our marker, register it.
            if callable(value) and getattr(value, '_auto_register', False):
                cls._registry[name] = value

    @staticmethod
    def register(func: callable) -> callable:
        """Decorator that simply marks a function so that __init_subclass__
        knows it should be placed in the registry.
        """
        func._auto_register = True
        return func

    @classmethod
    def get_registered_methods(cls) -> dict[str, callable]:
        """Retrieve registered methods."""
        return cls._registry


class BaseTaskPlots(AutoRegister):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info: dict, folder_path: list, plot_cfg:dict) -> None:
        self._dfs = dfs
        self._trajectories_dfs = trajectories_dfs
        self._labels = labels
        self._env_info = env_info
        self._save_plots_folder_path = folder_path
        self._plot_cfg = plot_cfg

        dfs_to_concat = []
        trajectory_offset = 0

        if len(self._trajectories_dfs) > 0:
            for group_key, group_dfs in self._trajectories_dfs.items():
                for df in group_dfs:
                    df = df.copy()
                    df['trajectory'] += trajectory_offset
                    dfs_to_concat.append(df)
                    max_traj = df['trajectory'].max()
                    trajectory_offset = max_traj + 1

            self.trajectories_to_plot = pd.concat(dfs_to_concat, ignore_index=True)
            trajectory_names = self.trajectories_to_plot['trajectory'].unique()
            self.trajectory_color_map_hex = {name: "#%06x" % random.randint(0, 0xFFFFFF) for name in trajectory_names}

        self.ALPHA_VALUE = 0.8

        


    def plot(self):
        raise NotImplementedError("Subclasses should implement this method.")
    

    def plot_trajectory_completed_successfully(self, key_to_plot: str):
        key_name, units = key_to_plot.split(".")

        fig, ax = plt.subplots(figsize=(14, 8))

        data_to_plot = []
        label_names = []

        
        colors_indx = []
        for color_idx, (group_key, group_dfs) in enumerate(self._dfs.items()):
            try:
                group_values = pd.concat([df[key_to_plot] for df in group_dfs], ignore_index=True)
                group_values = group_values.dropna()
                total_vals = len(group_values)
                false_vals = group_values.sum()
                true_vals = total_vals - false_vals
                data_to_plot.append([true_vals, false_vals])

                # Use the enumeration index, cycling through available colors if needed
                colors_indx.append(color_idx % len(self._plot_cfg["box_colors"]))
                label_names.append(group_key.split("_")[-1])

            except KeyError as e:
                print(f"KeyError: {e}. The key '{key_to_plot}' does not exist in the DataFrames.")
                print(self.__class__.__name__)
                return
            
        # Use colors from plot_cfg based on the order of groups
        outer_colors = [self._plot_cfg["box_colors"][index] for index in colors_indx]
        
        # Create inner colors by making variations of the outer colors
        inner_colors = []
        for i, outer_color in enumerate(outer_colors):
            # Convert hex to RGB
            outer_rgb = mcolors.hex2color(outer_color)
            
            # Create two variations: lighter and darker
            # For true values (successful): make it lighter
            lighter = tuple(min(1.0, c + 0.2) for c in outer_rgb)
            # For false values (unsuccessful): make it darker  
            darker = tuple(max(0.0, c - 0.2) for c in outer_rgb)
            
            inner_colors.extend([mcolors.rgb2hex(lighter), mcolors.rgb2hex(darker)])

        vals = np.array(data_to_plot)
        size = 0.3
        
        # Create custom autopct function for outer pie chart
        def make_autopct_outer(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n(Total Num Envs {val})'
            return my_autopct
        
        # Outer pie chart with group labels
        outer_wedges, outer_texts, outer_autotexts = ax.pie(
            vals.sum(axis=1), 
            radius=1, 
            colors=outer_colors,
            labels=label_names,
            wedgeprops=dict(width=size, edgecolor='w'),
            autopct=make_autopct_outer(vals.sum(axis=1)),
            pctdistance=0.85,
            labeldistance=1.1
        )

        # Inner pie chart with success/failure labels
        inner_labels = []
        for i, (success, failure) in enumerate(data_to_plot):
            inner_labels.extend(['Success', 'DNF'])

        # Create custom autopct function for inner pie chart
        def make_autopct_inner(values, labels_list):
            counter = 0
            def my_autopct(pct):
                nonlocal counter
                total = sum(values)
                val = int(round(pct*total/100.0))
                label_type = "Success" if counter % 2 == 0 else "DNF"
                counter += 1
                return f'{label_type}\n{val}'
            return my_autopct

        inner_wedges, inner_texts, inner_autotexts = ax.pie(
            vals.flatten(), 
            radius=1-size, 
            colors=inner_colors,
            # labels=inner_labels,
            wedgeprops=dict(width=size, edgecolor='w'),
            autopct=make_autopct_inner(vals.flatten(), inner_labels),
            pctdistance=0.7,
            textprops={'fontsize': 8}
        )

        ax.set(aspect="equal", title='Trajectory Completion Success Rate')
        plt.tight_layout()
        if key_name == "faild_trajectories_mask":
            key_name = "trajectory_completition_comparison"
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{key_name}.png")
        plt.savefig(save_path)
        plt.close()

    def boxplot(self, key_to_plot: str):
        key_name, units = key_to_plot.split(".")

        fig, ax = plt.subplots(figsize=(14, 6))

        data_to_plot = []
        label_names = []

        
        colors_indx = []
        for color_idx, (group_key, group_dfs) in enumerate(self._dfs.items()):
            # Concatenate all values of the key across the group's dataframes
            # for group_idx, df in enumerate(group_dfs):
            #     print("#" * 20)
            #     print(f"Group {group_idx}: {df.columns}")

            try:
                group_values = pd.concat([df[key_to_plot] for df in group_dfs], ignore_index=True)
                group_values = group_values.dropna()
            except KeyError as e:
                print(f"KeyError: {e}. The key '{key_to_plot}' does not exist in the DataFrames.")
                print(self.__class__.__name__)
                return
            
            # Use the enumeration index, cycling through available colors if needed
            colors_indx.append(color_idx % len(self._plot_cfg["box_colors"]))
            
            
            data_to_plot.append(group_values)
            label_names.append(group_key)

        box = ax.boxplot(
            data_to_plot,
            labels=label_names,
            patch_artist=True,
            boxprops=dict(facecolor='skyblue'),
            medianprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5),
            showfliers=False,
            widths=0.6,
        )

        if self._plot_cfg["zoom_in"]:
            all_values = pd.concat(data_to_plot)
            ax.set_ylim(top=all_values.quantile(0.95) )

        # Set the color of the boxes
        colors = [self._plot_cfg["box_colors"][index] for index in colors_indx]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        y_label, units = key_to_plot.split(".")
        y_label = y_label.replace("_", " ").capitalize()
        ax.set_title(f"{y_label}")
        ax.set_ylabel(f"{y_label} ({units})")
        ax.set_xticks(range(1, len(label_names) + 1))
        ax.set_xticklabels(label_names, rotation=20, ha='right')
        ax.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{key_name}.png")
        plt.savefig(save_path)
        plt.close()

        # Append mean and sd to a summary file
        summary_file_path = os.path.join(self._save_plots_folder_path, "metrics_summary.txt")
        with open(summary_file_path, "a") as f:
            f.write(f"Task: {self.task_name}, Key: {key_to_plot}, Title: {y_label}\n")
            for i, (group_key, group_values) in enumerate(zip(label_names, data_to_plot)):
                mean_val = group_values.mean()
                std_val = group_values.std()
                median_val = group_values.median()
                print(f"Group: {group_key}, Mean: {mean_val}, SD: {std_val}, Median: {median_val}")
                f.write(f"Group: {group_key} \nMean, SD, Median\n${mean_val:.5f} \\pm {std_val:.5f}, {median_val:.5f}$\n")
            f.write("\n")

    def dotplot(self, key_to_plot: str):
        key_name, units = key_to_plot.split(".")

        fig, ax = plt.subplots(figsize=(14, 6))

        data_to_plot = []
        label_names = []

        colors_indx = []
        for color_idx, (group_key, group_dfs) in enumerate(self._dfs.items()):
            try:
                group_values = pd.concat([df[key_to_plot] for df in group_dfs], ignore_index=True)
            except KeyError as e:
                print(f"KeyError: {e}. The key '{key_to_plot}' does not exist in the DataFrames.")
                print(self.__class__.__name__)
                return
            
            # Use the enumeration index, cycling through available colors if needed
            colors_indx.append(color_idx % len(self._plot_cfg["box_colors"]))
            
            data_to_plot.append(group_values)
            label_names.append(group_key.split("_")[-1])

        # Create a dot plot
        for i, (values, label) in enumerate(zip(data_to_plot, label_names)):
            ax.plot([i + 1] * len(values), values, 'o', color=self._plot_cfg["box_colors"][colors_indx[i]], markersize=5)

        if self._plot_cfg["zoom_in"]:
            all_values = pd.concat(data_to_plot)
            ax.set_ylim(top=all_values.quantile(0.95) )

        y_label, units = key_to_plot.split(".")
        y_label = y_label.replace("_", " ").capitalize()
        ax.set_title(f"{y_label}")
        ax.set_ylabel(f"{y_label} ({units})")
        ax.set_xticks(range(1, len(label_names) + 1))
        ax.set_xticklabels(label_names, rotation=20, ha='right')
        ax.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{key_name}.png")
        plt.savefig(save_path)
        plt.close()

    def plot_mean_sd(self, data_df, y_col, fname, ylabel=None):
        grouped = data_df.groupby('step')[y_col]
        mean = grouped.mean()
        std = grouped.std()
        steps = mean.index
        plt.figure(figsize=(8,6))
        plt.plot(steps, mean, label='Mean', color='b')
        plt.fill_between(steps, mean-std, mean+std, color='b', alpha=0.2, label='±1 SD')
        plt.title(f'{y_col.replace("_", " ").title()} Mean ± SD Over Time')
        plt.xlabel('Step')
        plt.ylabel(ylabel if ylabel else y_col)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(fname)
        plt.close()

    def plot_trajectories(self, ax, data_df, x_col, y_col, title, xlabel, ylabel, x_lim=None, y_lim=None, add_target_lines=False):
        """
        Plots individual trajectories with distinct random hex colors using a global trajectory_color_map_hex.
        """
        for trajectory_name, group in data_df.groupby('trajectory'):
            # Get the pre-assigned random hex color for this trajectory
            hex_color = self.trajectory_color_map_hex[trajectory_name]
            ax.plot(group[x_col], group[y_col], color=hex_color, alpha=self.ALPHA_VALUE, label=f'Trajectory {trajectory_name}')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if add_target_lines:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_aspect('equal', adjustable='box') # Only for normalized XY plot

        if x_lim:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim:
            ax.set_ylim(y_lim[0], y_lim[1])

    def plot_trajectories_with_gradient(self, ax, data_df, x_col, y_col, step_col='step', cmap_name='viridis'):
        for trajectory_name, group in data_df.groupby('trajectory'):
            group = group.sort_values(step_col)
            x = group[x_col].values
            y = group[y_col].values
            steps = group[step_col].values

            # Normalize steps to [0, 1] for colormap
            norm = mpl.colors.Normalize(vmin=steps.min(), vmax=steps.max())
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(
                segments,
                cmap=cmap_name,
                norm=norm,
                array=steps[:-1],  # color by step
                linewidth=2,
                alpha=1.0
            )
            ax.add_collection(lc)
            ax.autoscale_view()

        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Step')

    def plot_xy_trajectories_0_centered(self):
        xy_df = self.trajectories_to_plot.copy()
        xy_df['tx'] = xy_df.groupby('trajectory')['target_position_x'].transform('first')
        xy_df['ty'] = xy_df.groupby('trajectory')['target_position_y'].transform('first')
        xy_df['norm_position_x'] = xy_df['position_x'] - xy_df['tx']
        xy_df['norm_position_y'] = xy_df['position_y'] - xy_df['ty']
        xy_df = xy_df.drop(columns=['tx', 'ty'])

        fig, ax = plt.subplots(figsize=(8,8))
        self.plot_trajectories_with_gradient(ax, xy_df, 'norm_position_x', 'norm_position_y', step_col='step', cmap_name='viridis')
        ax.set_title('XY Trajectory (Target at 0,0)')
        ax.set_xlabel('Normalized Position X')
        ax.set_ylabel('Normalized Position Y')
        # ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        # ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_xy-trajectory-normalized.png")
        plt.savefig(save_path)
        plt.close(fig)

    def plot_masses(self):
        fig2, ax2 = plt.subplots(figsize=(8,6))
        self.plot_trajectories(
            ax2, self.trajectories_to_plot, 'step', 'masses',
            'Avg Mass Over Time', 'Step', 'Mass (kg)',
        )
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_masses-over-time.png")
        plt.savefig(save_path)
        plt.close(fig2)

        # Position distance mean ± sd
        dist_col = 'masses'
        self.plot_mean_sd(self.trajectories_to_plot, dist_col, os.path.join(self._save_plots_folder_path, f"{self.task_name}_masses-over-time-mean-sd.png"), ylabel='Distance')


    def plot_position_distance_over_time(self):
        fig2, ax2 = plt.subplots(figsize=(8,6))
        self.plot_trajectories(
            ax2, self.trajectories_to_plot, 'step', 'position_distance',
            'Position Distance to Target Over Time', 'Step', 'Distance'
        )
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_position-distance-over-time.png")
        plt.savefig(save_path)
        plt.close(fig2)

        # Position distance mean ± sd
        dist_col = 'position_distance'
        self.plot_mean_sd(self.trajectories_to_plot, dist_col, os.path.join(self._save_plots_folder_path, f"{self.task_name}_position-distance-over-time-mean-sd.png"), ylabel='Distance')

    def plot_linear_velocity_over_time(self):
        fig4, axs4 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
        self.plot_trajectories(
            axs4[0], self.trajectories_to_plot, 'step', 'linear_velocity_x',
            'Linear Velocity X Over Time', 'Step', 'Linear Velocity X'
        )
        self.plot_trajectories(
            axs4[1], self.trajectories_to_plot, 'step', 'linear_velocity_y',
            'Linear Velocity Y Over Time', 'Step', 'Linear Velocity Y'
        )

        axs4[-1].set_xlabel('Step') # Set x-label only on the last subplot for shared x-axis
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_linear-velocities-over-time.png")
        plt.savefig(save_path)
        plt.close(fig4)

        # Combined mean±SD plot for linear velocities (x and y)
        fig7, axs7 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
        for i, col in enumerate(['linear_velocity_x', 'linear_velocity_y']):
            grouped = self.trajectories_to_plot.groupby('step')[col]
            mean = grouped.mean()
            std = grouped.std()
            steps = mean.index
            axs7[i].plot(steps, mean, label='Mean', color='b')
            axs7[i].fill_between(steps, mean-std, mean+std, color='b', alpha=0.2, label='±1 SD')
            axs7[i].set_title(f'{col.replace("_", " ").title()} Mean ± SD Over Time')
            axs7[i].set_ylabel(col)
            axs7[i].legend()
        axs7[-1].set_xlabel('Step')
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_linear-velocities-over-time-mean-sd.png")
        plt.savefig(save_path)
        plt.close(fig7)

    # GoToPosition, GoToPose
    def plot_angular_velocity_over_time(self):
        fig6, ax6 = plt.subplots(figsize=(8,6))
        self.plot_trajectories(
            ax6, self.trajectories_to_plot, 'step', 'angular_velocity_z',
            'Angular Velocity Z Over Time', 'Step', 'Angular Velocity Z'
        )
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_angular-velocity-over-time.png")
        plt.savefig(save_path)
        plt.close(fig6)

        self.plot_mean_sd(self.trajectories_to_plot, 'angular_velocity_z', os.path.join(self._save_plots_folder_path, f"{self.task_name}_angular-velocity-over-time-mean-sd.png"), ylabel='Angular Velocity Z')

    def plot_actions_over_time(self):
        action_cols = [col for col in self.trajectories_to_plot.columns if col.startswith('actions_')]
        fig5, axs5 = plt.subplots(len(action_cols), 1, figsize=(10, 2*len(action_cols)), sharex=True)
        if len(action_cols) == 1:
            axs5 = [axs5]

        for i, col in enumerate(action_cols):
            self.plot_trajectories(
                axs5[i], self.trajectories_to_plot, 'step', col,
                f'{col} Over Time', 'Step', col
            )
        axs5[-1].set_xlabel('Step')
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_actions-over-time.png")
        plt.savefig(save_path)
        plt.close(fig5)

    # ----------------------------- TrackVelocitiesPlots -----------------------------
    def plot_linear_velocity_error_over_time(self):
        fig4, axs4 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
        self.plot_trajectories(
            axs4[0], self.trajectories_to_plot, 'step', 'error_linear_velocity',
            'Error Linear Velocity X Over Time', 'Step', 'Error Linear Velocity X'
        )
        self.plot_trajectories(
            axs4[1], self.trajectories_to_plot, 'step', 'error_lateral_velocity',
            'Error Linear Velocity Y Over Time', 'Step', 'Error Linear Velocity Y'
        )

        axs4[-1].set_xlabel('Step') # Set x-label only on the last subplot for shared x-axis
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_linear-velocities-error-over-time.png")
        plt.savefig(save_path)
        plt.close(fig4)

    def plot_angular_velocity_error_over_time(self):
        fig6, ax6 = plt.subplots(figsize=(8,6))
        self.plot_trajectories(
            ax6, self.trajectories_to_plot, 'step', 'error_angular_velocity',
            'Error Angular Velocity Z Over Time', 'Step', 'Error Angular Velocity Z'
        )
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_angular-velocity-error-over-time.png")
        plt.savefig(save_path)
        plt.close(fig6)

    # ----------------------------- RaceGatesPlots -----------------------------

    def plot_trajectory_with_targets(self):
        """Plot robot trajectory with race gates marked for each group/run."""
        
        # Create separate plots for each group
        for group_key, group_trajectory_dfs in self._trajectories_dfs.items():
            if not group_trajectory_dfs:  # Skip if no data
                continue
                
            # Concatenate all trajectories for this group
            group_trajectories = pd.concat(group_trajectory_dfs, ignore_index=True)
            
            if group_trajectories.empty:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot robot trajectories for this group with reduced opacity
            for trajectory_name, group in group_trajectories.groupby('trajectory'):
                hex_color = self.trajectory_color_map_hex[trajectory_name]
                ax.plot(group['position_x'], group['position_y'], 
                       color=hex_color, alpha=self.ALPHA_VALUE, linewidth=2,
                       label=f'Robot Trajectory {trajectory_name}', zorder=1)
            
            # Extract and plot gates (use first trajectory from group for gate positions)
            first_trajectory = group_trajectories[group_trajectories['trajectory'] == group_trajectories['trajectory'].iloc[0]]
            if not first_trajectory.empty:
                first_row = first_trajectory.iloc[0]
                
                # Extract target positions (gates) - they come in pairs (x, y)
                # target_positions_0, target_positions_1 = gate 0 (x, y)
                # target_positions_2, target_positions_3 = gate 1 (x, y), etc.
                gate_positions = []
                gate_headings = []
                
                # Find how many gates we have by checking target_positions columns
                target_pos_cols = [col for col in group_trajectories.columns if col.startswith('target_positions_')]
                target_heading_cols = [col for col in group_trajectories.columns if col.startswith('target_headings_')]
                
                num_gates = len(target_pos_cols) // 2  # Divide by 2 since each gate has x,y
                
                for i in range(num_gates):
                    x_col = f'target_positions_{i*2}'
                    y_col = f'target_positions_{i*2+1}'
                    heading_col = f'target_headings_{i}'
                    
                    if x_col in first_row and y_col in first_row:
                        gate_x = first_row[x_col]
                        gate_y = first_row[y_col]
                        gate_heading = first_row[heading_col] if heading_col in first_row else 0
                        
                        # Only add gates with non-zero positions (assuming zero means no gate)
                        if gate_x != 0 or gate_y != 0:
                            gate_positions.append((gate_x, gate_y))
                            gate_headings.append(gate_heading)
                
                # Remove duplicate gates (e.g., if last gate is same as first gate)
                if len(gate_positions) > 1 and gate_positions[-1] == gate_positions[0]:
                    gate_positions = gate_positions[:-1]
                    gate_headings = gate_headings[:-1]
                
                # Plot gates as rotated rectangles
                if gate_positions:
                    for i, ((gx, gy), heading) in enumerate(zip(gate_positions, gate_headings)):
                        # Create rotated rectangle for gate
                        from matplotlib.patches import Rectangle
                        from matplotlib.transforms import Affine2D
                        
                        # Gate dimensions
                        gate_width = 0.2  # Width of the gate
                        gate_height = 1.0  # Thickness of the gate
                        
                        # Create rectangle centered at origin
                        rect = Rectangle((-gate_width/2, -gate_height/2), gate_width, gate_height,
                                       facecolor='red', edgecolor='black', alpha=0.9, linewidth=2)
                        
                        # Apply rotation and translation
                        t = Affine2D().rotate(heading).translate(gx, gy) + ax.transData
                        rect.set_transform(t)
                        ax.add_patch(rect)
                        
                        # Add gate numbers (starting from 1) with higher zorder to ensure they're on top
                        ax.annotate(f'{i+1}', (gx, gy), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                                   zorder=10)
                    
                    # Add legend entry for gates
                    dummy_rect = Rectangle((0, 0), 0, 0, facecolor='red', edgecolor='black', alpha=0.8, label='Race Gates')
            
            # Extract group name for title
            group_name = group_key.split("_")[-1]  # Get the run name
            ax.set_title(f'Robot Trajectories with Race Gates - {group_name}')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_trajectory-with-gates_{group_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

    def plot_trajectory_with_targets_mean_sd(self):
        """Plot mean and standard deviation of robot trajectories with race gates for each group/run."""
        
        # Create separate plots for each group
        for group_key, group_trajectory_dfs in self._trajectories_dfs.items():
            if not group_trajectory_dfs:  # Skip if no data
                continue
                
            # Concatenate all trajectories for this group
            group_trajectories = pd.concat(group_trajectory_dfs, ignore_index=True)
            
            if group_trajectories.empty:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Calculate mean and std for trajectory positions at each step
            grouped_by_step = group_trajectories.groupby('step')
            mean_x = grouped_by_step['position_x'].mean()
            mean_y = grouped_by_step['position_y'].mean()
            std_x = grouped_by_step['position_x'].std()
            std_y = grouped_by_step['position_y'].std()
            
            # Plot mean trajectory
            ax.plot(mean_x, mean_y, color='blue', linewidth=3, alpha=0.8, label='Mean Trajectory')
            
            # Plot confidence ellipses at regular intervals
            steps_to_plot = mean_x.index[::max(1, len(mean_x)//20)]  # Plot ~20 ellipses
            
            for step in steps_to_plot:
                if step in mean_x.index and step in std_x.index:
                    center_x, center_y = mean_x[step], mean_y[step]
                    std_x_val, std_y_val = std_x[step], std_y[step]
                    
                    # Skip if std is NaN (only one trajectory)
                    if pd.isna(std_x_val) or pd.isna(std_y_val):
                        continue
                    
                    # Create confidence ellipse (1 standard deviation)
                    from matplotlib.patches import Ellipse
                    ellipse = Ellipse((center_x, center_y), 
                                    width=2*std_x_val, height=2*std_y_val,
                                    alpha=0.3, facecolor='lightblue', 
                                    edgecolor='blue', linewidth=0.5)
                    ax.add_patch(ellipse)
            
            # Add a legend entry for the confidence region
            ax.scatter([], [], alpha=0.3, facecolor='lightblue', 
                      edgecolor='blue', s=100, label='±1 SD Region')
            
            # Extract and plot gates (use first trajectory from group for gate positions)
            first_trajectory = group_trajectories[group_trajectories['trajectory'] == group_trajectories['trajectory'].iloc[0]]
            if not first_trajectory.empty:
                first_row = first_trajectory.iloc[0]
                
                # Extract gate positions
                gate_positions = []
                gate_headings = []
                
                target_pos_cols = [col for col in group_trajectories.columns if col.startswith('target_positions_')]
                target_heading_cols = [col for col in group_trajectories.columns if col.startswith('target_headings_')]
                
                num_gates = len(target_pos_cols) // 2
                
                for i in range(num_gates):
                    x_col = f'target_positions_{i*2}'
                    y_col = f'target_positions_{i*2+1}'
                    heading_col = f'target_headings_{i}'
                    
                    if x_col in first_row and y_col in first_row:
                        gate_x = first_row[x_col]
                        gate_y = first_row[y_col]
                        gate_heading = first_row[heading_col] if heading_col in first_row else 0
                        
                        if gate_x != 0 or gate_y != 0:
                            gate_positions.append((gate_x, gate_y))
                            gate_headings.append(gate_heading)
                
                # Remove duplicate gates (e.g., if last gate is same as first gate)
                if len(gate_positions) > 1 and gate_positions[-1] == gate_positions[0]:
                    gate_positions = gate_positions[:-1]
                    gate_headings = gate_headings[:-1]
                
                # Plot gates as rotated rectangles
                if gate_positions:
                    for i, ((gx, gy), heading) in enumerate(zip(gate_positions, gate_headings)):
                        # Create rotated rectangle for gate
                        from matplotlib.patches import Rectangle
                        from matplotlib.transforms import Affine2D
                        
                        # Gate dimensions
                        gate_width = 0.2  # Width of the gate
                        gate_height = 1.0  # Thickness of the gate
                        
                        # Create rectangle centered at origin
                        rect = Rectangle((-gate_width/2, -gate_height/2), gate_width, gate_height,
                                       facecolor='red', edgecolor='black', alpha=0.8, linewidth=2)
                        
                        # Apply rotation and translation
                        t = Affine2D().rotate(heading).translate(gx, gy) + ax.transData
                        rect.set_transform(t)
                        ax.add_patch(rect)
                        
                        # Add gate numbers (starting from 1)
                        ax.annotate(f'{i+1}', (gx, gy), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                                   zorder=15)
                    
                    # Add legend entry for gates
                    dummy_rect = Rectangle((0, 0), 0, 0, facecolor='red', edgecolor='black', alpha=0.8, label='Race Gates')
            
            # Extract group name for title
            group_name = group_key.split("_")[-1]  # Get the run name
            ax.set_title(f'Mean Trajectory with ±1 SD - {group_name}')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_trajectory-with-gates-mean-sd_{group_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
    