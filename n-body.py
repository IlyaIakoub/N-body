import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from numba import jit

#Constante
G=4*np.pi**2
#Masses en masse du soleil, positions en UA, temps en années

#Définition d'objets de type planet
class planet:
    def __init__(self, x, y, u, v, mass):
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.mass = mass
        #tableaux des anciennes positions de la planète, utile pour l'animation
        self.x_tab = np.array([])
        self.y_tab = np.array([])

    #fonction qui change la position et la vitesse de la planète et stocke la position dans les tableaux x_tab et y_tab
    def update_position(planet, x_pos, y_pos, u_vit, v_vit):
        planet.x_tab = np.append(planet.x_tab, x_pos)
        planet.y_tab = np.append(planet.y_tab, y_pos)
        planet.x = x_pos
        planet.y = y_pos
        planet.u = u_vit
        planet.v = v_vit
        return planet.x_tab, planet.y_tab

#On ajoute le rayon de la planète au rayon dans la formule afin d'éviter une division par 0 qui donnerait des vitesses infinies, tous les planètes ont la même densitée (M/R)
#@jit
def gravity_x(x1, x2, y1, y2, M_planete):
    f = -((G*M_planete*(x1-x2))/(((x1-x2)**2+(y1-y2)**2 + 0.00465047*M_planete)**(3/2)))
    return f
#Pas sur si @jit fonctionne ou change quelque chose mais au moins j'ai essayé...
#@jit
def gravity_y(x1, x2, y1, y2, M_planete):
    f = -((G*M_planete*(y1-y2))/(((x1-x2)**2+(y1-y2)**2 + 0.00465047*M_planete)**(3/2)))
    return f

#On calcule les positions de tous les planètes sur le temps t avec un pas de h, planets doit être un tableau d'objet de type planet
def move(planets, t, h, boundaries=True):
    
    #Boucle d'itération pour faire avancer le temps
    for i in range(t.size - 1):

        #Tableaux de tous les attributs des planètes que nous avons besoin
        x_ip1 = np.array([getattr(obj, 'x') for obj in planets]) + h*np.array([getattr(obj, 'u') for obj in planets])
        y_ip1 = np.array([getattr(obj, 'y') for obj in planets]) + h*np.array([getattr(obj, 'v') for obj in planets])
        x_i = np.array([getattr(obj, 'x') for obj in planets])
        y_i = np.array([getattr(obj, 'y') for obj in planets])
        M = np.array([getattr(obj, 'mass') for obj in planets])

        #Boucle d'itération pouor calculer la position de chaque planète
        for _, j in enumerate(planets):

            # indice de tous les planètes sauf celle pour laquelle se fait le calcul
            index = (planets!=j).nonzero()
            #En fournissant des tableaux à la fonction gravity_x et gravity_y, on nous retourne un tableau contenant l'influence de chaque planète indiviuellement, on en fait donc la somme.
            u_ip1 = j.u + h*(np.sum(gravity_x(j.x, x_i[index], j.y, y_i[index], M[index])) + np.sum(gravity_x(x_ip1[_], x_ip1[index], y_ip1[_], y_ip1[index], M[index])))/2
            v_ip1 = j.v + h*(np.sum(gravity_y(j.x, x_i[index], j.y, y_i[index], M[index])) + np.sum(gravity_y(x_ip1[_], x_ip1[index], y_ip1[_], y_ip1[index], M[index])))/2
            x_ip1[_] = j.x + (h/2) * (j.u+u_ip1)
            y_ip1[_] = j.y + (h/2) * (j.v+v_ip1)

            #Frontière de notre simulation, pas une partie essentielle du code
            if boundaries:
                if x_ip1[_] >= 5: u_ip1 =-5#-u_ip1
                if y_ip1[_] >= 5: v_ip1 =-5#-v_ip1
                if x_ip1[_] <= -5: u_ip1 =5#-u_ip1
                if y_ip1[_] <= -5: v_ip1 =5#-v_ip1
            
            j.update_position(x_ip1[_], y_ip1[_], u_ip1, v_ip1)
    return 

#Créée des objets de type «planet» avec des positions et vitesses aléatoires
def spawn_planets(n_planets):

    all_planets = np.array([], dtype='object')

    # La distributions des masses suit la loi normale
    mu=0.3
    sigma=0.4
    mass = np.abs(np.random.normal(mu, sigma, n_planets))

    for i in range(n_planets):
        rdm1 = np.random.uniform(low=-4, high=4, size=(2,))
        rdm2 = np.random.uniform(low=-15, high=15, size=(2,))
        #u and v sont les vitesses initiales en x et y
        #ordre : (x, y, u, v, mass)
        p = planet(rdm1[0], rdm1[1], rdm2[0], rdm2[1], mass[i])
        all_planets = np.hstack((all_planets, p))
    return all_planets


# Il faut fournir à la fonction un tableau des planètes à bouger et animer
def simulation(all_planets, speed=1, h=1/365, t_max=10, boundaries=True, tracing_length = 100):

    n_planets=all_planets.size
    t=np.arange(0, t_max, h)

    #On bouge les planètes
    move(all_planets, t, h, boundaries=boundaries)

    #tableaux contenant respectivement tous les positions x et tous les positions y de tous les planètes [planète : position_{x ou y}(t)]
    planets_x = np.array([getattr(obj, 'x_tab') for obj in all_planets])
    planets_y = np.array([getattr(obj, 'y_tab') for obj in all_planets])

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1)

    #Tableau de tous les masses des planètes pour pouvoir faire la grandeure du point qui représente la planète proportionnelle à sa masse
    s=np.array([getattr(obj, 'mass') for obj in all_planets])
    planets_plot = ax.scatter(planets_x[:,0], planets_y[:,0], s=s*10, c=np.arange(0,n_planets))

    #Tableau de tous les objets de type plt.Lines2D (je crois) qui représentent les orbites des planètes
    orbits = np.zeros_like(all_planets, 'object')
    for _ in range(n_planets):
        orbits[_], = ax.plot(0,0, linewidth=0.2)
    ax.set_aspect('equal')

    #Fonction à animer
    def animate(k):

        data = np.vstack((planets_x[:,k*speed], planets_y[:,k*speed])).T
        planets_plot.set_offsets(data)

        # Dessine tous les orbites en commenceant dans les derniers tracing_length pas de temps (dépendamment de «speed»)
        for i,j in enumerate(orbits):
            if k>=tracing_length: # i=planet
                j.set_data(planets_x[i,(k-tracing_length)*speed:k*speed], planets_y[i,(k-tracing_length)*speed:k*speed])
            else:
                j.set_data(planets_x[i,:k*speed], planets_y[i,:k*speed])
        return planets_plot,

    ax.axis('off')
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    ani = animation.FuncAnimation(fig, animate, interval=40, frames = round(t.size/speed)-1)
    #ani.save('n-body.mp4')
    plt.show()


# Premier argument est un tableaux numpy d'objets de type planet, ces objets sont créés soit avec la fonction spawn_planets(n_planètes) qui les
# créées à des positions, masses et vitesses aléatoires, soit manuellement avec planet(position_initiale_x, position_initiale_y, vitesse_initiale_x, vitesse_initiale_y, masse)
simulation(spawn_planets(30), speed=1)

# Un autre exemple de simulation
p1 = planet(0, 2, 2, 0, 1)
p2 = planet(2*np.cos(np.deg2rad(30)), -2*np.sin(np.deg2rad(30)), 2*np.cos(np.deg2rad(-120)), 2*np.sin(np.deg2rad(-120)), 1)
p3 = planet(-2*np.cos(np.deg2rad(30)), -2*np.sin(np.deg2rad(30)), 2*np.cos(np.deg2rad(-240)), 2*np.sin(np.deg2rad(-240)), 1)
simulation(np.array([p1, p2, p3], dtype='object'), h=1/3600, speed=50, tracing_length=1000, boundaries=False)
