'''
MTF estimation from the diffusion approximation of the Radiative Transfer Equation
Arturo Pardo
Grupo Ing. Fotónica, 2018
'''

# Standard modules
import numpy as np
import scipy.optimize as optimize
import os

'''
We need to produce at least two modules:
(1) One that generates the diffusion approximation of Rd(lambda, fx)
(2) A module that estimates if the diffusion approximation is good enough
    for this particular problem. 
'''

# (1) diffuse approximation of reflectance
def Rd_diffuse(fx, mu_s_prime, mu_a, n=1.37):
    '''
    Estimates the backscattered reflectance provided by a material of specific
    absorption and scattering at a specific spatial frequency (1/m)
    :param fx: Spatial frequency of operation (1/m)
    :param mu_s_prime: Reduced scattering coefficient (mu_s') (1/m)
    :param mu_a: Absorption coefficient (mu_a) (1/m)
    :param n: Refraction index of the medium
    :return: Returns Rd, backscattered reflectance by the diffusion approximation (a.u.)
    '''

    # (1) Reference parameters ---- ---- ---- ----

    # I have just noticed that the units aren't really that
    # important, as long as they are the same units for all the variables.

    # Transport coefficient
    mu_tr = mu_a + mu_s_prime

    # Effective absorption coefficient
    mu_eff = np.sqrt(3.0*mu_a*mu_tr)

    # Spatial wavenumber (frequency, really)
    k_x = 2.0*np.pi*fx

    # (2) Intermediate paremeters ---- ---- ---- ----

    # Reduced albedo
    a_prime = mu_s_prime / mu_tr

    # Effective reflection coefficient
    Reff = 0.0636*n + 0.668 + 0.710/n - 1.440/(n**2.0)

    # Proportionality constant
    A = 0.5*(1-Reff)/(1+Reff)

    # Effective transport due to kx
    mu_eff_prime = np.sqrt(mu_eff**2.0 + k_x**2.0)


    # (3) Diffuse reflectance ---- ---- ---- ----
    C1 = 3.0*A*a_prime
    C2 = (mu_eff_prime/mu_tr + 1.0)
    C3 = (mu_eff_prime/mu_tr + 3.0*A)
    Rd = C1/(C2*C3)

    return Rd


# (1) Sub-diffusive approximation of reflectance
def Rd_subdiffuse(fx, mu_s_prime, gamma, eta=0.003, zeta1 = 68.6, zeta2=-0.97, zeta3= 0.61, zeta4= 16.6, debug=0):
    '''
    Semi-empirical model of backscattered reflectance in Spatial Frequency Domain Imaging (SFDI)
    :param fx: Spatial frequency value/range (in 1/m)
    :param mu_s_prime: Reduced scattering coefficient (mu_s') (1/m)
    :param gamma:   Relative probability of large backscattering events
                    Proportional to the weighted ratio of 1st and 2nd Legendre moments of P(theta_s)
                    P(theta_s) is the scattering phase function of the sample
    :return:
    '''

    # (1) Reference parameters ---- ---- ---- ----
    '''
    Stephen Chad Kanick, David M. McClatchy, 
    V. Krishnaswamy, J. T. Elliott, K. D. Paulsen, and Brian W. Pogue
    From 'Sub-diffuse scattering parameter maps recovered using wide-field
    high-frequency structured light imaging'
    '''
    v = mu_s_prime*((1E-15+ fx)**(-1))

    # (2) Intermediate paremeters ---- ---- ---- ----
    A0 = zeta4*(gamma**(-2.0))
    A1 = np.power(v,(-zeta3*gamma))
    A = A0*A1

    C1 = eta*(1.0 + A)
    C2 = np.power(v, (-zeta2*gamma))
    C3 = (zeta1*(gamma**2.0) + C2)

    Rd = C1*(C2/C3)

    # (3) Final returned function ---- ---- ---- ----
    if debug == 0:
        return Rd
    else:
        return Rd, v, C1, C2, C3, C2/C3

def Rd_complete(fx, mu_s_prime, mu_a, gamma, n, debug=0, eta_mode='const', eta=1.0, thresh=0.5E3):

    # Find reflectance graphs for both models
    R_d = Rd_diffuse(fx, mu_s_prime, mu_a, n)
    R_sd = Rd_subdiffuse(fx, mu_s_prime, gamma, eta=eta)

    # Find mu_tr
    mu_tr = mu_s_prime + mu_a
    #thresh = 0.33*mu_tr

    if debug == 1: print('Rd = ' + str(R_d))
    if debug == 1: print('Rsd = ' + str(R_sd))

    # Values should meet at one point, given by fx = 0.2
    pos = np.argmin(np.abs(fx-thresh))
    if debug == 1: print('pos = ' + str(pos) + ', corresponding to fx = ' + str(fx[pos]))

    # Two weighting modes -- whichever works is the one we will use.
    if eta_mode == 'const':
        alpha = 1.0
    else:
        alpha = R_d[pos]/R_sd[pos]

    if debug == 1: print('alpha = ' + str(alpha))

    Rd_mix = np.zeros(np.size(fx))
    Rd_mix[0:pos] = R_d[0:pos]
    Rd_mix[pos:] = R_sd[pos:]*alpha

    if debug == 1: print('Rd_mix = ' + str(Rd_mix))

    return Rd_mix

def inverse_Rd_complete(Rd, fx, debug=0):
    popt, pcov = optimize.curve_fit(Rd_complete, fx, Rd,
                                    bounds = ([0.1, 0.0001, 0.5, 1.3], [40.0, 0.4, 2.0, 1.4]),
                                    p0 = [0.1, 0.0001, 1.0, 1.37])

    if debug == 1: print('popt_complete:' + str(popt))
    return popt

def inverse_Rd_diffuse(Rd, fx, debug=0):
    popt, pcov = optimize.curve_fit(Rd_diffuse, fx, Rd,
                                    bounds=([0.01, 0.01, 1.369], [40.0, 0.4, 1.37]),
                                    p0=[0.01, 0.01, 1.37],
                                    max_nfev=100000)

    if debug == 1: print('popt:' + str(popt))
    return popt

def inverse_Rd_subdiffuse(Rd, fx, debug=0):
    # eta=0.003, zeta1 = 68.6, zeta2= -0.97, zeta3= 0.61, zeta4= 16.6
    popt, pcov = optimize.curve_fit(Rd_subdiffuse, fx, Rd,
                                    bounds=([0.4, 0.90,
                                             0.1, 68.599, -0.9701, 0.6099, 16.599],
                                            [40.0, 3.0,
                                             1.0, 68.6, -0.97, 0.61, 16.6]),
                                    p0=[0.41, 0.9, 1.0, 68.6, -0.97, 0.61, 16.6],
                                    max_nfev=10000)

    if debug == 1: print('popt:' + str(popt))
    return popt

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    experimentName = 'with_sign_negative'

    ''' Pyplot parameters for compiling LaTeX strings'''
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.rc('font', size=14)
    plt.rc('legend', fontsize=10)

    test = 5# Try 0, 1, or 2.

    if test == 0:

        ''' Initial parameters for frequency response '''
        fx_range = np.linspace(0.001, 10.0, 2000)  # Spatial frequency range
        N = np.size(fx_range)  # Resolution

        ''' Material 1  '''
        n = 1.33  # Refraction index
        mu_s_prime = 10.0
        mu_a = 0.000
        gamma_range = np.linspace(1.3, 2.4, 10)

        ''' Módulo original '''
        Rd_diff = Rd_diffuse(fx_range, mu_s_prime, mu_a, n)

        fig1 = plt.figure(1, figsize=(18, 7))
        ax11 = fig1.add_subplot(1, 3, 1)
        ax21 = fig1.add_subplot(1, 3, 2)
        ax31 = fig1.add_subplot(1, 3, 3)

        plt.ion()

        diff = np.zeros(np.size(fx_range))

        eta = 1.0

        for g in range(0, np.size(gamma_range)):
            Rd_sdiff, v, C1, C2, C3, Cdiv = Rd_subdiffuse(fx_range, mu_s_prime, gamma_range[g], eta, debug=1)
            Rd_original = Rd_diffuse(fx_range, mu_s_prime, mu_a, n)

            diff += np.abs(Rd_sdiff-Rd_original*eta)

            ax11.semilogy(1/v, Rd_sdiff, color=[1.0/(0.5*g+1), 1.0/(0.5*g+1), 0.0] ,label='$\\gamma = ' + str(np.round(gamma_range[g], 2)) + '$')
            ax11.set_xlabel("$f_{x}/\\mu_{s}'$")
            ax11.set_ylabel('$R_{d}$')
            ax11.grid(True)
            ax11.set_title('Semi-empirical function for sd-SFDI')

            ax21.loglog(v, C1, color=[1.0/(0.5*g+1), 0.0, 0.0], label='$C1, \\gamma = ' + str(np.round(gamma_range[g], 2))  + '$')
            ax21.loglog(v, C2, color=[0.0, 1.0/(0.5*g+1), 0.0],  label='$C2, \\gamma = ' + str(np.round(gamma_range[g], 2))+ '$' )
            ax21.loglog(v, C3, color=[0.0, 0.0, 1.0/(0.5*g+1)], label='$C3, \\gamma = ' + str(np.round(gamma_range[g], 2)) + '$')
            ax21.loglog(v, Cdiv, 'k',  label='$C2/C3, \\gamma = ' + str(np.round(gamma_range[g], 2))  + '$')
            ax21.set_xlabel("$\\mu_{s}' f_{x}^{-1}$")
            ax21.grid(True)
            ax21.set_title('Corresponding sub-functions')

            ax31.semilogy(fx_range, Rd_sdiff, color=[0.9/(0.5*g+1), 0.9/(0.5*g+1), 0.9/(0.5*g+1)], label='$\\gamma = ' + str(np.round(gamma_range[g], 2))+ '$')

            ax31.set_xlim(left=0.0, right=5.0)

        ax31.semilogy(fx_range, diff/np.size(gamma_range), color=[0.9, 0.0, 0.0], label='Average error')
        ax31.semilogy(fx_range, eta*Rd_original, linestyle='--', color=[0, 0, 0],
                  label='$R_{d}$ (diffuse model)')
        ax31.set_title('Comparison: Diffuse vs. Sub-diffuse')
        ax31.text(0.6, 0.015, "$\\mu_{s}'= " + str(np.round(mu_s_prime)) + '$ 1/mm',
                                                            horizontalalignment='center')
        ax31.axvline(x=0.5, label='Boundary between models')
        ax31.set_xlabel("$f_{x}$ (1/mm)")
        ax31.set_ylabel("$R_{d}(f_{x})$ (\%)")


        ax11.legend()
        ax21.legend(loc='upper left')
        ax31.legend()
        fig1.canvas.draw()

        os.makedirs('../../../output/utils_sfdi', exist_ok=True)
        plt.savefig('../../../output/utils_sfdi/function_' + experimentName + '.pdf')

        plt.ioff()

    if test == 1:

        ''' Initial parameters for frequency response '''
        fx_range = np.linspace(0.1, 10.0, 2000)  # Spatial frequency range
        N = np.size(fx_range)  # Resolution

        ''' Material 1  '''
        mu_s_prime = 10
        mu_a = 0.001
        n = 1.33
        gamma_range = np.linspace(1.2, 2.4, 20)

        ''' Figuras varias '''
        fig1 = plt.figure(1)
        ax11 = fig1.add_subplot(1, 1, 1)

        ''' Plotting '''
        plt.ion()

        ax11.clear()
        for t in range(0, np.size(gamma_range)):
            Rd_sdiff = Rd_subdiffuse(fx_range, mu_s_prime, gamma_range[t], eta=0.45)
            ax11.semilogy(fx_range/mu_s_prime, Rd_sdiff, c=[t/np.size(gamma_range), 0.0, 0.0],
                          label='mu_s = ' + str(np.round(gamma_range[t], 3)))
            ax11.set_ylim(bottom=2E-3, top=1.0)
            fig1.canvas.draw()


        plt.ioff()

    if test == 2:
        ''' Figure 2 '''

        fx_range = np.linspace(1E-5, 1, 1000)
        mu_s_range = np.array([0.3, 0.5, 1.0, 3.0, 5.0, 10.0])
        gamma = np.array([1.3, 1.9])
        eta = 0.003

        fig1 = plt.figure(1)
        ax11 = fig1.add_subplot(1, 3, 1)
        ax12 = fig1.add_subplot(1, 3, 2)
        ax13 = fig1.add_subplot(1, 3, 3)

        for m in range(0, np.size(mu_s_range)):
            Rd = Rd_subdiffuse(fx_range, mu_s_range[m], gamma[0], eta)
            Rd2 = Rd_subdiffuse(fx_range, mu_s_range[m], gamma[1], eta)
            ax11.semilogy(fx_range, 100*Rd, color = [m/np.size(mu_s_range), 0.0, 0.0])
            ax12.semilogy(fx_range, 100*Rd2, color = [0.0, m/np.size(mu_s_range), 0.0])
            ax13.loglog(mu_s_range[m]/fx_range, 100*Rd, color = [m/np.size(mu_s_range), 0.0, 0.0])
            ax13.loglog(mu_s_range[m]/fx_range, 100*Rd2, color = [0.0, m/np.size(mu_s_range), 0.0])

    if test == 3:
        #fx_range = np.array([   0.,  150.,  610., 1370.])
        fx_range = np.array([0.    , 0.0465, 0.1488, 0.6053, 0.7764, 0.9157, 1.3736])
        #fx_range = np.linspace(0, 1.4, 20)
        mu_s = 1
        mu_a = 0.001

        Rd = Rd_complete(fx_range, mu_s, mu_a, 1.45, 1.3, eta_mode='const')
        Rd_ = Rd_complete(fx_range, mu_s, mu_a, 1.45, 1.3, eta_mode='adaptive')
        Rd1 = Rd_diffuse(fx_range, mu_s, mu_a, 1.3)
        Rd2 = Rd_subdiffuse(fx_range, mu_s, 1.45, eta=1.0)

        fig1 = plt.figure(1)
        ax11 = fig1.add_subplot(1, 1, 1)
        ax11.clear()
        ax11.semilogy(fx_range, Rd, marker='.', label='complete, const', alpha=0.8)
        ax11.semilogy(fx_range, Rd_, marker='.', label='complete, adaptive', alpha=0.8)
        ax11.semilogy(fx_range, Rd1, marker='.', label='diffuse', alpha=0.5)
        ax11.semilogy(fx_range, Rd2, marker='.', label='subdiffuse', alpha=0.5)

        ax11.legend(fancybox=True)

    if test == 4:
        fx_range = np.array([0., 0.0465, 0.1488, 0.6053, 0.7764, 0.9157, 1.3736]) * 1E3
        fx_range = np.array([0., 0.1488, 0.6053, 1.3736]) * 1E3
        #fx_range = np.linspace(0, 1.4E3, 20)
        mu_s = 500
        mu_a = 200
        gamma = 2.0

        Rd_1 = Rd_complete(fx_range, mu_s, mu_a, gamma, 1.37, eta_mode='const')
        Rd_2 = Rd_complete(fx_range, mu_s/2, mu_a/2, gamma, 1.37, eta_mode='const')

        fig1 = plt.figure(1)
        ax11 = fig1.add_subplot(1, 1, 1)
        #ax11.clear()
        ax11.semilogy(fx_range, Rd_1, label='full', marker='.', alpha=0.8)
        ax11.semilogy(fx_range, Rd_2, label='half', marker='.', alpha=0.8)
        ax11.legend(fancybox=True)


    if test == 5:
        fx_range = np.array([0., 0.0465, 0.1488, 0.6053, 0.7764, 0.9157, 1.3736])*1E3
        #fx_range = np.linspace(0, 1.4, 100)

        mu_s_range = np.array([0.45, 0.55])*1E3
        mu_a_range = np.array([0.1, 0.3])*1E3
        gamma_range = np.array([2.3, 2.8])

        random_ops = np.random.rand(3, 1000)
        random_ops[0, :] = mu_s_range[0] + (mu_s_range[1] - mu_s_range[0])*random_ops[0, :]
        random_ops[1, :] = mu_a_range[0] + (mu_a_range[1] - mu_a_range[0])*random_ops[1, :]
        random_ops[2, :] = gamma_range[0] + (gamma_range[1] - gamma_range[0])*random_ops[2, :]

        spectra_list = []
        for k in range(0, 1000):
            spectra_list.append(Rd_complete(fx_range, random_ops[0, k], random_ops[1, k], random_ops[2, k], 1.4, eta_mode='const'))

        spectra_list = np.asarray(spectra_list).T

        fig1 = plt.figure(1)
        ax11 = fig1.add_subplot(1, 1, 1)
        ax11.clear()
        ax11.semilogy(fx_range, spectra_list)

