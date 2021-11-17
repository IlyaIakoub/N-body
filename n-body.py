import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

G=4*np.pi**2

class planet:
    def __init__(self, x, y, u, v, mass):
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.mass = mass
        self.x_tab = np.array([])
        self.y_tab = np.array([])

    def update_position(planet, x_pos, y_pos, u_vit, v_vit):
        planet.x_tab = np.append(planet.x_tab, x_pos)
        planet.y_tab = np.append(planet.y_tab, y_pos)
        planet.x = x_pos
        planet.y = y_pos
        planet.u = u_vit
        planet.v = v_vit
        return planet.x_tab, planet.y_tab


#I added 0.01 to r so that speed cannot be infinite, consider this the radius of the planet
#The radius is proportionnal to the mass, all planets have the density of the sun
@jit
def gravity_x(x1, x2, y1, y2, M_planete ):
    f = -((G*M_planete*(x1-x2))/(((x1-x2)**2+(y1-y2)**2 + 0.00465047*M_planete)**(3/2)))#-((G*2*(x1))/(((x1)**2+(y1)**2 + 0.01)**(3/2)))#Last part is an unmoving so, remove it if you want
    return f
@jit
def gravity_y(x1, x2, y1, y2, M_planete):
    f = -((G*M_planete*(y1-y2))/(((x1-x2)**2+(y1-y2)**2 + 0.00465047*M_planete)**(3/2)))#-((G*2*(y1))/(((x1)**2+(y1)**2 + 0.01)**(3/2)))#Last part is an unmoving so, remove it if you want
    return f

def move(planets, t, h):
    
    for i in range(t.size - 1):

        #Arrays of attributes of all the planets
        x_ip1 = np.array([getattr(obj, 'x') for obj in planets]) + h*np.array([getattr(obj, 'u') for obj in planets])
        y_ip1 = np.array([getattr(obj, 'y') for obj in planets]) + h*np.array([getattr(obj, 'v') for obj in planets])
        x_i = np.array([getattr(obj, 'x') for obj in planets])
        y_i = np.array([getattr(obj, 'y') for obj in planets])
        M = np.array([getattr(obj, 'mass') for obj in planets])

        for _, j in enumerate(planets):

            # all planets but j
            index = (planets!=j).nonzero()

            u_ip1 = j.u + h*(np.sum(gravity_x(j.x, x_i[index], j.y, y_i[index], M[index])) + np.sum(gravity_x(x_ip1[_], x_ip1[index], y_ip1[_], y_ip1[index], M[index])))/2
            v_ip1 = j.v + h*(np.sum(gravity_y(j.x, x_i[index], j.y, y_i[index], M[index])) + np.sum(gravity_y(x_ip1[_], x_ip1[index], y_ip1[_], y_ip1[index], M[index])))/2
            x_ip1[_] = j.x + (h/2) * (j.u+u_ip1)
            y_ip1[_] = j.y + (h/2) * (j.v+v_ip1)

            #border #if it -u and -v the speed is too big
            if x_ip1[_] >= 5: u_ip1 =-5#-u_ip1
            if y_ip1[_] >= 5: v_ip1 =-5#-v_ip1
            if x_ip1[_] <= -5: u_ip1 =5#-u_ip1
            if y_ip1[_] <= -5: v_ip1 =5#-v_ip1
            
            j.update_position(x_ip1[_], y_ip1[_], u_ip1, v_ip1)

    return 

def spawn_planets(n_planets):
    all_planets = np.array([], dtype='object')

    #initial positions
    posx= np.linspace(-5,5,n_planets)
    posy= np.random.uniform(low=-5, high=5, size=(n_planets,))

    #Mass is a Gaussian
    mu=0.3
    sigma=0.4
    mass = np.abs(np.random.normal(mu, sigma, n_planets))

    for i in range(n_planets):
        # rdm1 = np.random.uniform(low=-4, high=4, size=(2,))
        rdm2 = np.random.uniform(low=-15, high=15, size=(2,))
        #u and v are initial speeds
        #(self, x, y, u, v, mass)
        p = planet(posx[i], posy[i], rdm2[0], rdm2[1], mass[i])
        all_planets = np.hstack((all_planets, p))
    return all_planets
            
def simulation(all_planets, speed=1, h=1/365, t_max=10):

    n_planets=all_planets.size
    t=np.arange(0, t_max, h)
    move(all_planets, t, h)

    #planets positions
    planets_x = np.array([getattr(obj, 'x_tab') for obj in all_planets])
    planets_y = np.array([getattr(obj, 'y_tab') for obj in all_planets])

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1)
    s=np.array([getattr(obj, 'mass') for obj in all_planets])
    planets_plot = ax.scatter(planets_x[:,0], planets_y[:,0], s=s*10, c=np.arange(0,n_planets))
    
    orbits = np.zeros_like(all_planets, 'object')
    for _ in range(n_planets):
        orbits[_], = ax.plot(0,0, linewidth=0.2)
    ax.set_aspect('equal')
    # ax.plot(0,0,'o',color='orange')

    def animate(k):
        data = np.vstack((planets_x[:,k*speed], planets_y[:,k*speed])).T
        planets_plot.set_offsets(data)
        # plotting orbits
        for i,j in enumerate(orbits):
            if k>=100: #i=planet
                j.set_data(planets_x[i,(k-100)*speed:k*speed], planets_y[i,(k-100)*speed:k*speed])
            else:
                j.set_data(planets_x[i,:k*speed], planets_y[i,:k*speed])
        return planets_plot,

    ax.axis('off')
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    ani = animation.FuncAnimation(fig, animate, interval=40, frames = round(t.size/speed)-1)
    #ani.save('n-body.mp4')
    plt.show()

simulation(spawn_planets(30), speed=1)