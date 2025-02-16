import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import time

#massen til de forskjellige planetene
m1 = 1.0
m2 = 1.0
m3 = 1.0
m4 = 1.0
m5 = 1.0

#initialiserer posisjoner [x, y, z]
inital_position_1 =  [1.0,  0.0,  1.0]
inital_position_2 =  [1.0,  1.0,  0.0]
inital_position_3 =  [0.0,   1.0, 1.0]
inital_position_4 =  [1.0,   1.0, 1.0]
inital_position_5 =  [0.0,  0.0,  1.0]

# Initialiserer hastigheter [x, y, z]
inital_velocity_1 =  [0.0, 0.0, -1.0]
inital_velocity_2 =  [0.0, 0.0, 1.0]
inital_velocity_3 =  [0.0, 0.0, -0.6]
inital_velocity_4 =  [0.0, 0.0, -0.9]
inital_velocity_5 =  [0.0, 0.0, -0.2]


# henter posisjonene og hastighet og setter dem inn i en 1 dimensjonal array, bruker .ravel for å forsikre at den er 1 dimensjonal
initial_conditions = np.array([
    inital_position_1, inital_position_2, inital_position_3, inital_position_4, inital_position_5,
    inital_velocity_1, inital_velocity_2, inital_velocity_3, inital_velocity_4, inital_velocity_5
]).ravel()



#kalkulerer posisjonene av planete med dimensjonsløs formel for python
def system_odes(t, S, m1, m2, m3, m4, m5):

    #henter posisjonen av objektene fra S. p1, p2, p3 har alle x, y, z koordinater. så f.eks S[0:3] er posisjonen til planet1 (p1 = [x1, y1, z1])
    p1, p2, p3, p4, p5= S[0:3], S[3:6], S[6:9], S[9:12], S[12:15]

    #s[9:12] hastighet til planet1 (dp1_dt = [Vx1, Vy1, Vz1])
    dp1_dt, dp2_dt, dp3_dt, dp4_dt, dp5_dt = S[15:18], S[18:21], S[21:24], S[24:27], S[27:30]

    
    #f1, f2, f3 er hastigheten til dp1_dt, dp2_dt, dp3_dt, this is a test
    f1, f2, f3, f4, f5 = dp1_dt, dp2_dt, dp3_dt, dp4_dt, dp5_dt

    
    #finner akselerasjonen til planetene
    df1_dt = m3*(p3 - p1)/np.linalg.norm(p3 - p1)**3 + m2*(p2 - p1)/np.linalg.norm(p2 - p1)**3 + m4*(p4 - p1)/np.linalg.norm(p4 - p1)**3 + m5*(p5 - p1)/np.linalg.norm(p5 - p1)**3
    df2_dt = m3*(p3 - p2)/np.linalg.norm(p3 - p2)**3 + m1*(p1 - p2)/np.linalg.norm(p1 - p2)**3 + m4*(p4 - p2)/np.linalg.norm(p4 - p2)**3 + m5*(p5 - p2)/np.linalg.norm(p5 - p2)**3
    df3_dt = m1*(p1 - p3)/np.linalg.norm(p1 - p3)**3 + m2*(p2 - p3)/np.linalg.norm(p2 - p3)**3 + m4*(p4 - p3)/np.linalg.norm(p4 - p3)**3 + m5*(p5 - p3)/np.linalg.norm(p5 - p3)**3
    df4_dt = m1*(p1 - p4)/np.linalg.norm(p1 - p4)**3 + m2*(p2 - p4)/np.linalg.norm(p2 - p4)**3 + m3*(p3 - p4)/np.linalg.norm(p3 - p4)**3 + m5*(p5 - p4)/np.linalg.norm(p5 - p4)**3
    df5_dt = m5*(p1 - p5)/np.linalg.norm(p1 - p5)**3 + m2*(p2 - p5)/np.linalg.norm(p2 - p5)**3 + m3*(p3 - p5)/np.linalg.norm(p3 - p5)**3 + m4*(p4 - p5)/np.linalg.norm(p4 - p5)**3


    #gjør til en 1D array for å gjøre det lettere for ODE solver
    return np.array([f1, f2, f3, f4, f5, df1_dt, df2_dt, df3_dt, df4_dt, df5_dt]).ravel()





#definerer time start og time end
time_s, time_e = 0, 75

#definerer hvor mange tids-punkter det skal være mellom time start og time end
t_points = np.linspace(time_s, time_e, 5000)

#henter current unix time
t1 = time.time()

#bruker solve_ivp funksjon og passerer dataen vår. den vil spytte ut en tids array først og så en y array som essensielt er løsningene våre. 
solution = solve_ivp(
    fun=system_odes,
    t_span=(time_s, time_e),
    y0=initial_conditions,
    t_eval=t_points,
    args=(m1, m2, m3, m4, m5)
)
print("this is the solution", solution)
#henter unix time igjen
t2 = time.time()
#finner hvor lang tid det tok for å løse
print(f"Solved in: {t2-t1:.3f} [s]")

#henter tidsarrayen fra solution
t_sol = solution.t

#henter planet n sin x, y, z verdi fra solution's y index
p1x_sol = solution.y[0]
p1y_sol = solution.y[1]
p1z_sol = solution.y[2]

p2x_sol = solution.y[3]
p2y_sol = solution.y[4]
p2z_sol = solution.y[5]

p3x_sol = solution.y[6]
p3y_sol = solution.y[7]
p3z_sol = solution.y[8]

p4x_sol = solution.y[9]
p4y_sol = solution.y[10]
p4z_sol = solution.y[11]

p5x_sol = solution.y[12]
p5y_sol = solution.y[13]
p5z_sol = solution.y[14]



# ------------------------------------------------------------------- #

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

#tegner den tynne linjen etter posisjonen
planet1_plt, = ax.plot(p1x_sol, p1y_sol, p1z_sol, 'green', label='Planet 1', linewidth=1)
planet2_plt, = ax.plot(p2x_sol, p2y_sol, p2z_sol, 'red', label='Planet 2', linewidth=1)
planet3_plt, = ax.plot(p3x_sol, p3y_sol, p3z_sol, 'blue',label='Planet 3', linewidth=1)
planet4_plt, = ax.plot(p4x_sol, p4y_sol, p4z_sol, 'purple',label='Planet 4', linewidth=1)
planet5_plt, = ax.plot(p5x_sol, p5y_sol, p5z_sol, 'yellow',label='Planet 5', linewidth=1)

#tegner en dot på den siste posisjonen
planet1_dot, = ax.plot([p1x_sol[-1]], [p1y_sol[-1]], [p1z_sol[-1]], 'o', color='green', markersize=6)
planet2_dot, = ax.plot([p2x_sol[-1]], [p2y_sol[-1]], [p2z_sol[-1]], 'o', color='red', markersize=6)
planet3_dot, = ax.plot([p3x_sol[-1]], [p3y_sol[-1]], [p3z_sol[-1]], 'o', color='blue', markersize=6)
planet4_dot, = ax.plot([p4x_sol[-1]], [p4y_sol[-1]], [p4z_sol[-1]], 'o', color='purple', markersize=6)
planet5_dot, = ax.plot([p5x_sol[-1]], [p5y_sol[-1]], [p5z_sol[-1]], 'o', color='yellow', markersize=6)


ax.set_title("The 3-Body Problem")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()






def update(frame):

    #plotter de 300 siste punktene
    lower_lim = max(0, frame - 300)
    print(f"Progress: {(frame+1)/len(t_points):.1%} | 100.0 %", end='\r')

    #definerer x,y,z sine verdier og lagres i en array fra lower lim til current frame
    x_current_1 = p1x_sol[lower_lim:frame+1]
    y_current_1 = p1y_sol[lower_lim:frame+1]
    z_current_1 = p1z_sol[lower_lim:frame+1]

    x_current_2 = p2x_sol[lower_lim:frame+1]
    y_current_2 = p2y_sol[lower_lim:frame+1]
    z_current_2 = p2z_sol[lower_lim:frame+1]

    x_current_3 = p3x_sol[lower_lim:frame+1]
    y_current_3 = p3y_sol[lower_lim:frame+1]
    z_current_3 = p3z_sol[lower_lim:frame+1]

    x_current_4 = p4x_sol[lower_lim:frame+1]
    y_current_4 = p4y_sol[lower_lim:frame+1]
    z_current_4 = p4z_sol[lower_lim:frame+1]

    x_current_5 = p5x_sol[lower_lim:frame+1]
    y_current_5 = p5y_sol[lower_lim:frame+1]
    z_current_5 = p5z_sol[lower_lim:frame+1]



    planet1_plt.set_data(x_current_1, y_current_1)
    planet1_plt.set_3d_properties(z_current_1)

    planet1_dot.set_data([x_current_1[-1]], [y_current_1[-1]])
    planet1_dot.set_3d_properties([z_current_1[-1]])



    planet2_plt.set_data(x_current_2, y_current_2)
    planet2_plt.set_3d_properties(z_current_2)

    planet2_dot.set_data([x_current_2[-1]], [y_current_2[-1]])
    planet2_dot.set_3d_properties([z_current_2[-1]])



    planet3_plt.set_data(x_current_3, y_current_3)
    planet3_plt.set_3d_properties(z_current_3)

    planet3_dot.set_data([x_current_3[-1]], [y_current_3[-1]])
    planet3_dot.set_3d_properties([z_current_3[-1]])


    planet4_plt.set_data(x_current_4, y_current_4)
    planet4_plt.set_3d_properties(z_current_4)
    
    planet4_dot.set_data([x_current_4[-1]], [y_current_4[-1]])
    planet4_dot.set_3d_properties([z_current_4[-1]])


    planet5_plt.set_data(x_current_5, y_current_5)
    planet5_plt.set_3d_properties(z_current_5)
    
    planet5_dot.set_data([x_current_5[-1]], [y_current_5[-1]])
    planet5_dot.set_3d_properties([z_current_5[-1]])



    return planet1_plt, planet1_dot, planet2_plt, planet2_dot, planet3_plt, planet3_dot, planet4_plt, planet4_dot, planet5_plt, planet5_dot

animation = FuncAnimation(fig, update, frames=range(0, len(t_points), 2), interval=8, blit=True)
plt.show()
