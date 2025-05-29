import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import time

class Planet:
    def __init__(self, mass, initial_position, initial_velocity, color='blue', name=None):
        self.mass = mass
        self.initial_position = np.array(initial_position)
        self.initial_velocity = np.array(initial_velocity)
        self.color = color
        self.name = name if name else f"Planet {id(self) % 1000}"
        
        # These will be set after simulation
        self.x_sol = None
        self.y_sol = None
        self.z_sol = None
    
    def set_solution(self, solution, planet_index):
        """Extract this planet's solution from the overall solution"""
        pos_start = planet_index * 3
        self.x_sol = solution.y[pos_start]
        self.y_sol = solution.y[pos_start + 1]
        self.z_sol = solution.y[pos_start + 2]


class NBodySystem:
    def __init__(self):
        self.planets = []
        self.solution = None
        
    def add_planet(self, planet):
        """Add a planet to the system"""
        self.planets.append(planet)
        return self  # For method chaining
        
    def get_initial_conditions(self):
        """Combine all planets' initial conditions into a 1D array"""
        positions = []
        velocities = []
        
        for planet in self.planets:
            positions.extend(planet.initial_position)
            velocities.extend(planet.initial_velocity)
            
        return np.array(positions + velocities)
    
    def system_odes(self, t, s):
        """ODE system for N-body problem with any number of planets"""
        n_planets = len(self.planets)
        
        # Extract positions and velocities from state vector
        positions = []
        for i in range(n_planets):
            positions.append(s[i*3:(i+1)*3])
            
        velocities = []
        for i in range(n_planets):
            velocities.append(s[n_planets*3 + i*3:n_planets*3 + (i+1)*3])
        
        # Calculate accelerations
        accelerations = []
        for i in range(n_planets):
            acceleration = np.zeros(3)
            for j in range(n_planets):
                if i != j:  # Skip self-interaction
                    r_ij = positions[j] - positions[i]
                    distance = np.linalg.norm(r_ij)
                    acceleration += self.planets[j].mass * r_ij / distance**3
            
            accelerations.append(acceleration)
        
        # Combine velocities and accelerations for return
        derivatives = []
        derivatives.extend([v for v in velocities])
        derivatives.extend([a for a in accelerations])
        
        return np.array(derivatives).ravel()
    
    def simulate(self, t_span, n_points=5000):
        """Run the simulation"""
        t_start, t_end = t_span
        t_points = np.linspace(t_start, t_end, n_points)
        
        t1 = time.time()
        initial_conditions = self.get_initial_conditions()
        
        self.solution = solve_ivp(
            fun=self.system_odes,
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_points
        )
        
        t2 = time.time()
        print(f"Solved in: {t2-t1:.3f} [s]")
        
        # Update each planet with its solution data
        for i, planet in enumerate(self.planets):
            planet.set_solution(self.solution, i)
        
        return self.solution
    
    def visualize(self, tail_length=300, interval=8, stride=2, x_range=None, y_range=None, z_range=None):
        """
        Visualize the system with animation
        
        Parameters:
        - tail_length: Number of points to show in the trail
        - interval: Animation interval in milliseconds
        - stride: Step size for animation frames
        - x_range, y_range, z_range: Custom axis limits as tuples (min, max)
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        planet_lines = []
        planet_dots = []
        
        for planet in self.planets:
            line, = ax.plot([], [], [], color=planet.color, label=planet.name, linewidth=1)
            dot, = ax.plot([], [], [], 'o', color=planet.color, markersize=6)
            planet_lines.append(line)
            planet_dots.append(dot)
        
        ax.set_title("N-Body Problem Simulation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        # Set axis limits - using custom values if provided
        if x_range is not None:
            ax.set_xlim(x_range)
        else:
            # Auto scaling as before
            all_x = np.concatenate([p.x_sol for p in self.planets])
            mid_x = (np.max(all_x) + np.min(all_x)) / 2
            max_range_x = np.ptp(all_x) * 0.55
            ax.set_xlim(mid_x - max_range_x, mid_x + max_range_x)
        
        if y_range is not None:
            ax.set_ylim(y_range)
        else:
            # Auto scaling as before
            all_y = np.concatenate([p.y_sol for p in self.planets])
            mid_y = (np.max(all_y) + np.min(all_y)) / 2
            max_range_y = np.ptp(all_y) * 0.55
            ax.set_ylim(mid_y - max_range_y, mid_y + max_range_y)
        
        if z_range is not None:
            ax.set_zlim(z_range)
        else:
            # Auto scaling as before
            all_z = np.concatenate([p.z_sol for p in self.planets])
            mid_z = (np.max(all_z) + np.min(all_z)) / 2
            max_range_z = np.ptp(all_z) * 0.55
            ax.set_zlim(mid_z - max_range_z, mid_z + max_range_z)
        
        plt.grid()
        plt.legend()
        
        def update(frame):
            lower_lim = max(0, frame - tail_length)
            print(f"Progress: {(frame+1)/len(self.solution.t):.1%} | 100.0 %", end='\r')
            
            artists = []
            for i, planet in enumerate(self.planets):
                x_current = planet.x_sol[lower_lim:frame+1]
                y_current = planet.y_sol[lower_lim:frame+1]
                z_current = planet.z_sol[lower_lim:frame+1]
                
                planet_lines[i].set_data(x_current, y_current)
                planet_lines[i].set_3d_properties(z_current)
                planet_dots[i].set_data([x_current[-1]], [y_current[-1]])
                planet_dots[i].set_3d_properties([z_current[-1]])
                
                artists.extend([planet_lines[i], planet_dots[i]])
            
            return artists
        
        frames = range(0, len(self.solution.t), stride)
        animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        return animation


# Example usage
def main():
    # Create a system
    system = NBodySystem()
    
    # Add planets (same as original example)
    system.add_planet(Planet(
        mass=1.0,
        initial_position=[1.0, 0.0, 1.0],
        initial_velocity=[0.0, 0.0, -1.0],
        color='green',
        name='Planet 1'
    ))
    
    system.add_planet(Planet(
        mass=1.0,
        initial_position=[1.0, 1.0, 0.0],
        initial_velocity=[0.0, 0.0, 1.0],
        color='red',
        name='Planet 2'
    ))
    
    system.add_planet(Planet(
        mass=1.0,
        initial_position=[0.0, 1.0, 1.0],
        initial_velocity=[0.0, 0.0, -0.6],
        color='blue',
        name='Planet 3'
    ))

    system.add_planet(Planet(
        mass=1.0,
        initial_position=[-3.0, 1.0, 1.0],
        initial_velocity=[0.0, 0.0, -0.6],
        color='yellow',
        name='Planet 4'
    ))




    
    # Run simulation
    system.simulate((0, 75), n_points=5000)
    
    # Visualize results with your custom ranges
    system.visualize()

if __name__ == "__main__":
    main()
