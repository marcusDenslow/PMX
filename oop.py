import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import time
from mpl_toolkits.mplot3d import Axes3D
import random
import sys


class Planet:
    """Class representing a celestial body in the simulation."""
    
    def __init__(self, mass, initial_position, initial_velocity, color=None, name=None):
        """
        Initialize a planet with its physical properties.
        
        Args:
            mass: Mass of the planet
            initial_position: [x, y, z] position coordinates
            initial_velocity: [vx, vy, vz] velocity components
            color: Color for visualization (random if None)
            name: Optional name for the planet
        """
        self.mass = float(mass)
        self.initial_position = np.array(initial_position, dtype=float)
        self.initial_velocity = np.array(initial_velocity, dtype=float)
        
        # Generate a random color if none provided
        if color is None:
            self.color = f"#{random.randint(0, 0xFFFFFF):06x}"
        else:
            self.color = color
            
        self.name = name or f"Planet {id(self) % 1000}"
        
    def __str__(self):
        return self.name


class NBodySimulation:
    """Class for managing the n-body physics simulation."""
    
    def __init__(self, planets):
        """
        Initialize the simulation with a list of planets.
        
        Args:
            planets: List of Planet objects
        """
        if len(planets) < 2:
            raise ValueError("Simulation requires at least two planets")
        
        self.planets = planets
        self.num_planets = len(planets)
        self.solution = None
        self.t_sol = None
        self.positions = None
        self.num_points = 0
        
    def _system_odes(self, t, S):
        """
        Calculate the system of ODEs for the n-body problem.
        
        Args:
            t: Time (required by solve_ivp but not used)
            S: State vector containing positions and velocities
            
        Returns:
            Array of derivatives for the system
        """
        n = self.num_planets
        
        # Extract positions and velocities from state vector
        positions = []
        velocities = []
        
        for i in range(n):
            pos_start = i * 3
            vel_start = (n + i) * 3
            
            # Get position and velocity for each planet
            positions.append(S[pos_start:pos_start + 3])
            velocities.append(S[vel_start:vel_start + 3])
        
        # First derivatives of positions are velocities
        position_derivatives = velocities.copy()
        
        # Calculate second derivatives (accelerations)
        velocity_derivatives = [np.zeros(3) for _ in range(n)]
        
        # Calculate forces between each pair of planets
        for i in range(n):
            for j in range(n):
                if i != j:  # Skip self-interaction
                    # Vector from planet i to planet j
                    r_ij = positions[j] - positions[i]
                    # Distance between planets
                    r_norm = np.linalg.norm(r_ij)
                    
                    if r_norm > 1e-10:  # Avoid division by zero if planets overlap
                        # Acceleration on planet i due to planet j
                        velocity_derivatives[i] += self.planets[j].mass * r_ij / r_norm**3
        
        # Combine all derivatives into a single array
        derivatives = []
        derivatives.extend([comp for pos_deriv in position_derivatives for comp in pos_deriv])
        derivatives.extend([comp for vel_deriv in velocity_derivatives for comp in vel_deriv])
        
        return np.array(derivatives)
    
    def _get_initial_conditions(self):
        """
        Prepare initial conditions for the ODE solver.
        
        Returns:
            1D array of initial positions and velocities
        """
        initial_conditions = []
        
        # Add all positions first
        for planet in self.planets:
            initial_conditions.extend(planet.initial_position)
            
        # Then add all velocities
        for planet in self.planets:
            initial_conditions.extend(planet.initial_velocity)
            
        return np.array(initial_conditions)
    
    def run_simulation(self, time_start=0, time_end=75, num_points=5000):
        """
        Run the simulation and store the results.
        
        Args:
            time_start: Start time for simulation
            time_end: End time for simulation
            num_points: Number of time points to evaluate
            
        Returns:
            The solution object from solve_ivp
        """
        t_points = np.linspace(time_start, time_end, num_points)
        t1 = time.time()
        
        solution = solve_ivp(
            fun=self._system_odes,
            t_span=(time_start, time_end),
            y0=self._get_initial_conditions(),
            t_eval=t_points,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        
        t2 = time.time()
        print(f"Solved in: {t2-t1:.3f} [s]")
        
        self.solution = solution
        self.t_sol = solution.t
        self.num_points = len(t_points)
        
        # Extract position data for easier access
        self.positions = {}
        
        for i in range(self.num_planets):
            self.positions[i] = {
                'x': solution.y[i*3],
                'y': solution.y[i*3 + 1],
                'z': solution.y[i*3 + 2]
            }
        
        return self.solution
    
    def get_planet_positions(self, planet_index):
        """
        Get the positions of a specific planet.
        
        Args:
            planet_index: Index of the planet
            
        Returns:
            Dictionary with 'x', 'y', 'z' keys containing position arrays
        """
        if self.solution is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        
        return self.positions[planet_index]
    
    def get_total_energy(self):
        """
        Calculate the total energy of the system at each time point.
        
        Returns:
            Array of total energy values
        """
        if self.solution is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        
        n = self.num_planets
        num_times = len(self.t_sol)
        total_energy = np.zeros(num_times)
        
        for t in range(num_times):
            # Calculate kinetic energy
            kinetic = 0
            for i in range(n):
                vx = self.solution.y[(n + i)*3][t]
                vy = self.solution.y[(n + i)*3 + 1][t]
                vz = self.solution.y[(n + i)*3 + 2][t]
                v_squared = vx**2 + vy**2 + vz**2
                kinetic += 0.5 * self.planets[i].mass * v_squared
            
            # Calculate potential energy
            potential = 0
            for i in range(n):
                for j in range(i+1, n):
                    xi = self.solution.y[i*3][t]
                    yi = self.solution.y[i*3 + 1][t]
                    zi = self.solution.y[i*3 + 2][t]
                    
                    xj = self.solution.y[j*3][t]
                    yj = self.solution.y[j*3 + 1][t]
                    zj = self.solution.y[j*3 + 2][t]
                    
                    r_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                    potential -= self.planets[i].mass * self.planets[j].mass / r_ij
            
            total_energy[t] = kinetic + potential
            
        return total_energy


class NBodyVisualizer:
    """Class for visualizing the n-body simulation results."""
    
    def __init__(self, simulation):
        """
        Initialize the visualizer with a simulation.
        
        Args:
            simulation: NBodySimulation object with results
        """
        self.simulation = simulation
        self.fig = None
        self.ax = None
        self.planet_lines = []
        self.planet_dots = []
        self.trail_length = 300  # Default trail length
        
    def setup_plot(self, title="N-Body Simulation", figsize=(10, 8), zoom_factor=0.6):
        """
        Set up the 3D plot with initial data.
        
        Args:
            title: Plot title
            figsize: Figure size tuple
            zoom_factor: Factor to control zoom level (smaller = more zoomed in)
            
        Returns:
            Figure and axis objects
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.zoom_factor = zoom_factor
        
        # Create lines and dots for each planet
        for i, planet in enumerate(self.simulation.planets):
            pos = self.simulation.get_planet_positions(i)
            
            # Plot line (trajectory)
            line, = self.ax.plot(
                [], [], [],
                color=planet.color,
                label=str(planet),
                linewidth=1
            )
            self.planet_lines.append(line)
            
            # Plot dot (current position)
            dot, = self.ax.plot(
                [], [], [],
                'o', color=planet.color, markersize=10 if i == 0 else 6  # Larger marker for the first body (e.g., Sun)
            )
            self.planet_dots.append(dot)
        
        # Set labels and title
        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        
        # Auto-scale the axes based on the data
        self._update_axes_limits(self.zoom_factor)
        
        plt.grid(True)
        plt.legend(loc='upper right', fontsize='small')
        
        return self.fig, self.ax
    
    def _update_axes_limits(self, zoom_factor=1.0):
        """
        Update the axes limits to fit all trajectories.
        
        Args:
            zoom_factor: Factor to control zoom level (smaller = more zoomed in)
        """
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        
        for i in range(self.simulation.num_planets):
            pos = self.simulation.get_planet_positions(i)
            
            min_x = min(min_x, np.min(pos['x']))
            max_x = max(max_x, np.max(pos['x']))
            
            min_y = min(min_y, np.min(pos['y']))
            max_y = max(max_y, np.max(pos['y']))
            
            min_z = min(min_z, np.min(pos['z']))
            max_z = max(max_z, np.max(pos['z']))
        
        # Calculate the center of the system
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        # Calculate the range for each dimension
        range_x = (max_x - min_x) * zoom_factor
        range_y = (max_y - min_y) * zoom_factor
        range_z = (max_z - min_z) * zoom_factor
        
        # Ensure minimum range to avoid flat views
        min_range = max(range_x, range_y, range_z) * 0.1
        range_x = max(range_x, min_range)
        range_y = max(range_y, min_range)
        range_z = max(range_z, min_range)
        
        # Set the limits centered on the system center
        self.ax.set_xlim([center_x - range_x/2, center_x + range_x/2])
        self.ax.set_ylim([center_y - range_y/2, center_y + range_y/2])
        self.ax.set_zlim([center_z - range_z/2, center_z + range_z/2])
    
    def _update_animation(self, frame):
        """
        Update function for the animation.
        
        Args:
            frame: Current frame number
            
        Returns:
            Updated line and dot objects
        """
        # Show only the last N points for a better visualization
        lower_lim = max(0, frame - self.trail_length)
        print(f"Progress: {(frame+1)/self.simulation.num_points:.1%}", end='\r')
        
        # Update each planet's position
        for i, (line, dot) in enumerate(zip(self.planet_lines, self.planet_dots)):
            pos = self.simulation.get_planet_positions(i)
            
            # Current slice of data to display
            x_current = pos['x'][lower_lim:frame+1]
            y_current = pos['y'][lower_lim:frame+1]
            z_current = pos['z'][lower_lim:frame+1]
            
            # Update line
            line.set_data(x_current, y_current)
            line.set_3d_properties(z_current)
            
            # Update dot (larger size for the first body, e.g., Sun in solar system)
            marker_size = 10 if i == 0 and len(self.simulation.planets) > 3 else 6
            dot.set_data([x_current[-1]], [y_current[-1]])
            dot.set_3d_properties([z_current[-1]])
            dot.set_markersize(marker_size)
            
        # Rotate view slightly for each frame to help with 3D perception
        # Only for frames that are multiples of 10 to reduce computation
        if frame % 10 == 0:
            current_azim = self.ax.azim
            self.ax.view_init(elev=20, azim=current_azim + 0.5)
        
        return self.planet_lines + self.planet_dots
    
    def animate(self, interval=20, step=5, trail_length=300, zoom_factor=0.6):
        """
        Create and display the animation.
        
        Args:
            interval: Time between frames in milliseconds
            step: Step size for frames
            trail_length: Length of the trailing path to show
            zoom_factor: Factor to control zoom level (smaller = more zoomed in)
            
        Returns:
            Animation object
        """
        if self.fig is None or self.ax is None:
            self.setup_plot(zoom_factor=zoom_factor)
        else:
            # Update the zoom factor if it was changed
            self._update_axes_limits(zoom_factor)
            
        self.trail_length = trail_length
            
        animation = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=range(0, len(self.simulation.t_sol), step),
            interval=interval,
            blit=True
        )
        
        plt.show()
        
        return animation
    
    def plot_energy(self):
        """
        Plot the total energy of the system over time.
        
        Returns:
            Figure with energy plot
        """
        energy = self.simulation.get_total_energy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.simulation.t_sol, energy)
        ax.set_title("Total Energy over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.grid(True)
        
        # Calculate and display relative energy change
        relative_change = (energy[-1] - energy[0]) / abs(energy[0]) * 100
        ax.text(
            0.05, 0.95,
            f"Relative energy change: {relative_change:.6f}%",
            transform=ax.transAxes,
            verticalalignment='top'
        )
        
        plt.show()
        return fig


def create_random_planets(num_planets, max_mass=10.0, max_radius=5.0, max_velocity=1.0):
    """
    Create a list of random planets for the simulation.
    
    Args:
        num_planets: Number of planets to create
        max_mass: Maximum mass for any planet
        max_radius: Maximum distance from origin for initial positions
        max_velocity: Maximum initial velocity component
    
    Returns:
        List of Planet objects
    """
    planets = []
    
    for i in range(num_planets):
        # Random mass between 0.1*max_mass and max_mass
        mass = 0.1 * max_mass + random.random() * 0.9 * max_mass
        
        # Random position within a sphere of radius max_radius
        theta = random.random() * 2 * np.pi
        phi = random.random() * np.pi
        r = random.random() * max_radius
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Random velocity components between -max_velocity and max_velocity
        vx = (2 * random.random() - 1) * max_velocity
        vy = (2 * random.random() - 1) * max_velocity
        vz = (2 * random.random() - 1) * max_velocity
        
        planet = Planet(
            mass=mass,
            initial_position=[x, y, z],
            initial_velocity=[vx, vy, vz],
            name=f"Planet {i+1}"
        )
        
        planets.append(planet)
    
    return planets


def example_three_body():
    """Run the classic three-body problem as demonstrated in the original code."""
    
    # Create the planets with the same parameters as the original code
    planet1 = Planet(
        mass=1.0,
        initial_position=[1.0, 0.0, 1.0],
        initial_velocity=[0.0, 0.0, -1.0],
        color='green',
        name='Planet 1'
    )
    
    planet2 = Planet(
        mass=1.0,
        initial_position=[1.0, 1.0, 0.0],
        initial_velocity=[0.0, 0.0, 1.0],
        color='red',
        name='Planet 2'
    )
    
    planet3 = Planet(
        mass=1.0,
        initial_position=[0.0, 1.0, 1.0],
        initial_velocity=[0.0, 0.0, -0.6],
        color='blue',
        name='Planet 3'
    )
    
    # Create and run the simulation
    simulation = NBodySimulation([planet1, planet2, planet3])
    simulation.run_simulation(time_start=0, time_end=75, num_points=5000)
    
    # Visualize the results
    visualizer = NBodyVisualizer(simulation)
    visualizer.animate(zoom_factor=0.4)  # More zoomed in view
    
    return simulation, visualizer


def main():
    """Main function with user interface to set up and run the simulation."""
    print("Welcome to the N-Body Simulator!")
    print("--------------------------------")
    
    # Choose simulation type
    print("\nChoose a simulation type:")
    print("1. Random planets")
    print("2. Solar system model")
    print("3. Custom planets")
    
    choice = int(input("Enter your choice (1-3): "))
    
    if choice == 1:
        # Random simulation
        num_planets = int(input("Enter number of planets (2-50): "))
        num_planets = max(2, min(50, num_planets))  # Limit between 2 and 50
        
        planets = create_random_planets(num_planets)
        
    elif choice == 2:
        # Solar system
        num_planets = int(input("Enter number of planets to include (1-8): "))
        num_planets = max(1, min(8, num_planets))
        
        scale_factor = float(input("Enter scale factor (0.1-10, default=1.0): ") or "1.0")
        scale_factor = max(0.1, min(10, scale_factor))
        
        planets = create_solar_system(num_planets, scale_factor)
        
    else:
        # Custom planets
        num_planets = int(input("Enter number of planets (2-20): "))
        num_planets = max(2, min(20, num_planets))
        
        planets = []
        
        for i in range(num_planets):
            print(f"\nPlanet {i+1}:")
            mass = float(input("  Mass: "))
            
            print("  Initial position (x, y, z):")
            x = float(input("    x: "))
            y = float(input("    y: "))
            z = float(input("    z: "))
            
            print("  Initial velocity (vx, vy, vz):")
            vx = float(input("    vx: "))
            vy = float(input("    vy: "))
            vz = float(input("    vz: "))
            
            color = input("  Color (leave blank for random): ")
            color = None if color.strip() == "" else color
            
            name = input("  Name (leave blank for default): ")
            name = f"Planet {i+1}" if name.strip() == "" else name
            
            planet = Planet(
                mass=mass,
                initial_position=[x, y, z],
                initial_velocity=[vx, vy, vz],
                color=color,
                name=name
            )
            
            planets.append(planet)
    
    # Simulation settings
    print("\nSimulation settings:")
    time_end = float(input("  Simulation time (default=50): ") or "50")
    num_points = int(input("  Number of time points (default=2000): ") or "2000")
    
    # Create and run simulation
    simulation = NBodySimulation(planets)
    simulation.run_simulation(time_start=0, time_end=time_end, num_points=num_points)
    
    # Visualization
    visualizer = NBodyVisualizer(simulation)
    
    # Choose what to display
    print("\nVisualization options:")
    print("1. Animation only")
    print("2. Energy plot only")
    print("3. Both animation and energy plot")
    
    viz_choice = int(input("Enter your choice (1-3): "))
    
    # Visualization settings
    zoom_factor = float(input("Enter zoom factor (0.2-1.0, smaller = more zoomed in, default=0.4): ") or "0.4")
    zoom_factor = max(0.2, min(1.0, zoom_factor))
    
    trail_length = int(input("Enter trail length (50-1000, default=300): ") or "300")
    trail_length = max(50, min(1000, trail_length))
    
    if viz_choice in [1, 3]:
        visualizer.animate(zoom_factor=zoom_factor, trail_length=trail_length)
    
    if viz_choice in [2, 3]:
        visualizer.plot_energy()
    
    print("Simulation complete!")


if __name__ == "__main__":
    # Allow direct starting of the original three-body problem
    if len(sys.argv) > 1 and sys.argv[1] == "--three-body":
        example_three_body()
    else:
        main()

