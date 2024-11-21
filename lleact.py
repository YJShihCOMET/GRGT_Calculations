import decimal
import numpy as np
import pandas as pd
import liquidmodel as lm
        
class Calc_binary_rg:
    # LLE calculation by coarse-graining RG
    tol_lle_dgmix = {'NRTL':1.e-12,
                     'Wilson':1.e-12,
                     'Margules':1.e-12,
                     'COSMOSAC2002e':1.e-9,
                     'COSMOSACb':1.e-12,
                     'COSMOSAC_MIT':1.e-7}
    
    def __init__(self, liquidmodelname, mixturename=None, lhat0=None, delta=None, deltal=1.\
                 , iteration=None, iterationdata=False, nx=1001, d=3, c_rg=2., dt=1.e-4, rgtype=None):
        """
        Parameters
        ----------
        liquidmodelname : string
            The name of liquid model
        mixturename : string
            The name of this mixture
        lhat0i : float
            one of Renormalization Group parameter 
        deltai : float
            one of Renormalization Group parameter 
        iteration : int or None
            The maximum iteration number including mean-field free energy, dGmix0
        iterationi : int or None
            initial setting of total iteration time you want to calculate
        iterationdata : boolean
            Whether need to leave iteration data
        nx : int
            total number of mole fraction data
        nc : int
            The Number of components in this mixture
        d : int
            The number of dimension
        c_rg : int
            The factor by which the length of a cell is increased per iteration
        dt : float 
            differential temperature, used in the calculation of heat capacity
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
        ----------
        object
        ----------
        cblleperg_lmobject : 
            The object of class "Context" in module "liquidmodel"
            
        """
        self._liquidmodelname = liquidmodelname
        self.lhat0 = lhat0
        self.delta = delta
        self.deltal = deltal
        self.iteration = iteration
        self.iterationi = iteration # initial iteration
        self.iterationdata = iterationdata
        self._nx = nx
        self.d = d
        self.c_rg = c_rg
        self.dt = dt
        self.rgtype = rgtype
        self.cbllerg_lmobject = lm.Context(lm.create(liquidmodelname))
        if mixturename is not None:
            self._mixturename = mixturename 
            self.cbllerg_lmobject.mixture = mixturename
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        #self.z = 2 #coordinate number

    @property
    def liquidmodelname(self):
        return self._liquidmodelname
    
    @liquidmodelname.setter
    def liquidmodelname(self, new_liquidmodelname):
        self._liquidmodelname = new_liquidmodelname
        self.cbllerg_lmobject = lm.Context(lm.create(new_liquidmodelname))
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    @property       
    def mixturename(self):
        return self._mixturename
    
    @mixturename.setter
    def mixturename(self, new_mixturename):
        self._mixturename = new_mixturename
        self.cbllerg_lmobject.mixture = new_mixturename
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    @property
    def nc(self):
        return self.cbllerg_lmobject.nc
        
    @property       
    def nx(self):
        return self._nx
    
    @nx.setter
    def nx(self, new_nx):
        self._nx = new_nx
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    def set_bip(self, aij, bij, cij, dij, eij, fij):
        self.cbllerg_lmobject.set_bip(aij, bij, cij, dij, eij, fij)
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    def set_epsilon(self, epsilon12dk, epsilon11dk, epsilon22dk, Nc=4, f_bip=True):
        self.epsilon12dk = epsilon12dk
        self.epsilon11dk = epsilon11dk
        self.epsilon22dk = epsilon22dk
        self.Nc = Nc
        if f_bip is True:
            self.cbllerg_lmobject.set_epsilon(epsilon12dk, epsilon11dk, epsilon22dk, Nc)
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None

    def set_cp(self, dbname, method):
        self.dbname = dbname
        self.method = method
        self.cbllerg_lmobject.set_cp(dbname, method)
    
    def set_rgp(self, lhat0=None, delta=None, deltal=None, iteration=None, iterationdata=None, rgtype=None):
        if lhat0 is not None:
            self.lhat0 = lhat0
        if delta is not None:
            self.delta = delta
        if deltal is not None:
            self.deltal = deltal
        if iteration is not None:
            self.iteration = iteration
            self.iterationi = iteration
        if iterationdata is not None:
            self.iterationdata = iterationdata
        if rgtype is not None:
            self.rgtype = rgtype
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
            
    def set_t(self, ti):
        self.t = ti
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
    
    @property
    def nymax(self):
        return int( ( self.nx - 1 ) / 2 + 1 )
    
    @property
    def jx_tail(self):
        return np.array(self.xmin * ( self.nx - 1 ) + 1, dtype = int) # x: 0 -> 1, iy: 1 -> 501 -> 1

    @property
    def jmx_tail(self):
        return np.arange(self.nx) + 1  # x: 0 -> 1, jpx_tail: 1 -> 1001
        
    @property
    def jpx_tail(self):
        return self.nx - np.arange(self.nx)  # x: 0 -> 1, jpx_tail: 1001 -> 1

    @property
    def jpx_end(self):
        return self.jpx_tail - 1  # x: 0 -> 1, jpx_tail: 1000 -> 0
    
    @property
    def dx(self):
        return 1 / ( self.nx - 1 )
    
    @property
    def x(self):
        x1i = np.linspace(0, 1, self.nx)
        return np.array([x1i, np.flipud(x1i)])
    
    @property
    def xmin(self):
        return np.amin(self.x, axis=0) # x: 0 -> 1, xmin: 0 -> 0.5 -> 0
    
    @property
    def ymax(self):
        return self.xmin[:self.nymax] # ymax: 0 -> 0.5
    
    @property
    def nmax(self): 
        """ The maximum number of iteration due to the limitation of maximum number of floating point"""
        maxnum = np.finfo(float).max * ( 1 - 1e-13 ) # the maximum number of floating point
        return int(np.floor(np.emath.logn(self.c_rg, maxnum **( 1 / self.d ) / self.lhat0)))
    
    @property
    def u(self):
        ti = self.t
        flag_setx = hasattr(self.cbllerg_lmobject, 'x')
        if flag_setx:
            xoi = self.cbllerg_lmobject.x
        self.cbllerg_lmobject.set_tx(ti, np.array([0.5,0.5]))
        ui = self.cbllerg_lmobject.u()
        if flag_setx:
            self.cbllerg_lmobject.set_tx(ti, xoi)
        '''
        if self.liquidmodelname[:5] == 'COSMO':
            ti = self.t
            self.cbllerg_lmobject.set_tx(ti, np.array([0.5,0.5]))
            ui = self.cbllerg_lmobject.u()
        else:
            zi = self.z
            ti = self.t
            self.cbllerg_lmobject.set_tx(ti, np.array([0.5,0.5]))
            ui = self.cbllerg_lmobject.u() * zi / 2
            #ui = self.cbllerg_lmobject.calc_gex()
        '''
        return ui
    
    @property
    def dgmix_cg(self):
        ti = self.t
        xi = self.x
        dgmixi = np.zeros(self.nx, dtype=float)
        for ix, xe in enumerate(xi.T):
            self.cbllerg_lmobject.set_tx(ti, xe)
            dgmixi[ix] = self.cbllerg_lmobject.calc_dgmix()
        return dgmixi
    
    @property
    def ddgmixdx1_cg(self):
        """
        Parameters
        ----------
        ddgmixdx1_cg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            first derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_cg = self.dgmix_cg
        ddgmixdx1_cg = np.zeros(self.nx, dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                ddgmixdx1_cg[ix] = ( dgmix_cg[ix+1] - dgmix_cg[ix] ) / dxi
            elif ( ix == self.nx - 1 ):
                ddgmixdx1_cg[ix] = ( dgmix_cg[ix] - dgmix_cg[ix-1] ) / dxi
            else:
                ddgmixdx1_cg[ix] = ( dgmix_cg[ix+1] - dgmix_cg[ix-1] ) / ( 2 * dxi )
        return ddgmixdx1_cg

    @property
    def d2dgmixdx12_cg(self):
        """
        Parameters
        ----------
        ddgmixdx1_cg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            first derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_cg = self.dgmix_cg
        d2dgmixdx12_cg = np.zeros(self.nx, dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                d2dgmixdx12_cg[ix] = ( dgmix_cg[ix+2] - 2 * dgmix_cg[ix+1] + dgmix_cg[ix] ) / ( dxi **2 )
            elif ( ix == self.nx - 1 ):
                d2dgmixdx12_cg[ix] = ( dgmix_cg[ix-2] - 2 * dgmix_cg[ix-1] + dgmix_cg[ix] ) / ( dxi **2 )
            else:
                d2dgmixdx12_cg[ix] = ( dgmix_cg[ix+1] - 2 * dgmix_cg[ix] + dgmix_cg[ix-1] ) / ( dxi **2 )
        return d2dgmixdx12_cg
    
    def calc_dgmix_rg(self):
        """
        Parameters
        ----------
        xi : numpy array, float ncXnx
            mole fraction of component i  
        dxi : float
            differential mole fraction of component i  
        lhat0i : float
            one of Renormalization Group parameter 
        deltai : float
            one of Renormalization Group parameter 
        iteration : int or None
            total iteration time you want to calculate
        iterationdata : boolean
            Whether need to leave iteration data
        nx : int
            total number of mole fraction data
        nymax : int
            total number of mole fraction data for maximum varible y
        nid : int
            actual iteration data for gibbs free energy of mixing 
            (if iterationdata is True: iteration, else: 1)
        xmin : numpy array, float 1Xnx
            minimum mole fraction for all component in xi
            eg. if x[i] = np.array([0.3, 0.7]), xmin[i] = 0.3
        ymax : numpy array, float 1Xnymax
            minimum mole fraction for all component in xi
        ui : float
            one of Renormalization Group parameter (total mixing internal energy for x1=0.5)
        tol_dgmix : float
            the tolerance for Renormalization Group iteration 
        n : int
            iteration step
        d : int
            dimension
        dgmix0 : numpy array,.float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data
        dgmixn : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        dgmixnm1 : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at (n-1)th iteration
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
        deltagn : numpy array, float 1Xnx
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        deltag_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data 
            (calculated by liquid modol+RG)
            
        """
        
        if self.__dgmix_rg is not None: # dgmix_rg have been calculated
            return self.__dgmix_rg
        """Get class attributes"""
        xi = self.x
        dxi = self.dx
        lhat0i = self.lhat0
        deltai = self.delta
        deltali = self.deltal
        iteration = self.iteration
        iterationdata = self.iterationdata
        nx = self.nx
        nymax = self.nymax
        jx_tail = self.jx_tail
        xmin = self.xmin
        ymax = self.ymax
        dgmix0 = self.dgmix_cg
        d = self.d
        ui = self.u  
        c_rg = self.c_rg
        tol_dgmix = self.tol_lle_dgmix[self.liquidmodelname]

        """Declare array"""
        gsn = np.zeros((nx, nymax), dtype=float)
        gln = np.zeros((nx, nymax), dtype=float)
        overflowfactorsn = np.zeros(nx, dtype=float)
        overflowfactorln = np.zeros(nx, dtype=float)
        numeratorn = np.zeros(nx, dtype=float)
        denominatorn = np.zeros(nx, dtype=float)

        if iterationdata is True:
            gs_rg = np.zeros((1, nx, nymax), dtype=float)
            gl_rg = np.zeros((1, nx, nymax), dtype=float)
            numerator_rg = np.zeros((1, nx), dtype=float)
            denominator_rg = np.zeros((1, nx), dtype=float)
            deltag_rg = np.zeros((1, nx), dtype=float)
            dgmix_rg = dgmix0.copy().reshape(1, nx)
        
        n = 0 
        dgmixn = dgmix0.copy()
        while True:
            n += 1
            vhatn = ( c_rg **n * lhat0i ) **d
            dgmixnm1 = dgmixn.copy()
            for ix, xe in enumerate(xi.T):
                for jx, ye in enumerate(ymax[:jx_tail[ix]]):
                    flucl = ui * ye **2 * deltali
                    flucs = c_rg **( - 2 * n ) * flucl * deltai
                    gsln = ( dgmixnm1[ix+jx] + dgmixnm1[ix-jx] ) / 2 - dgmixnm1[ix]
                    gsn[ix,jx] = gsln + flucs
                    gln[ix,jx] = gsln + flucl
                overflowfactorsn[ix] = np.max( - gsn[ix,:] )
                overflowfactorln[ix] = np.max( - gln[ix,:] )
                numeratorn[ix] = ( np.sum( np.exp( ( - gsn[ix,1:jx] - overflowfactorsn[ix] ) * vhatn ) ) + \
                                   ( np.exp( ( - gsn[ix,0] - overflowfactorsn[ix] ) * vhatn ) + \
                                    np.exp( ( - gsn[ix,jx] - overflowfactorsn[ix] ) * vhatn ) ) / 2 ) * dxi
                denominatorn[ix] = ( np.sum( np.exp( ( - gln[ix,1:jx] - overflowfactorln[ix] ) * vhatn ) ) + \
                                     ( np.exp( ( - gln[ix,0] - overflowfactorln[ix] ) * vhatn ) + \
                                      np.exp( ( - gln[ix,jx] - overflowfactorln[ix] ) * vhatn ) ) / 2 ) * dxi

                '''
                 *  Add a term in the exponential (each amplitude is weighted by exp(-y))
                numeratorn[ix] = ( np.sum( np.exp( ( - gsn[ix,1:jx] - overflowfactorsn[ix] ) * vhatn + ymax[1:jx] ) ) + \
                                   ( np.exp( ( - gsn[ix,0] - overflowfactorsn[ix] ) * vhatn + ymax[0] ) + \
                                    np.exp( ( - gsn[ix,jx] - overflowfactorsn[ix] ) * vhatn + ymax[jx] ) ) / 2 ) * dxi
                denominatorn[ix] = ( np.sum( np.exp( ( - gln[ix,1:jx] - overflowfactorln[ix] ) * vhatn + ymax[1:jx] ) ) + \
                                     ( np.exp( ( - gln[ix,0] - overflowfactorln[ix] ) * vhatn + ymax[0] ) + \
                                      np.exp( ( - gln[ix,jx] - overflowfactorln[ix] ) * vhatn + ymax[jx] ) ) / 2 ) * dxi

                --------------------------------
                 * Simpson's method
                if (jx_tail[ix] <= 2):
                    numeratorn[ix] = ( np.sum( np.exp( ( - gsn[ix,1:jx] - overflowfactorsn[ix] ) * vhatn ) ) + \
                                       ( np.exp( ( - gsn[ix,0] - overflowfactorsn[ix] ) * vhatn ) + \
                                        np.exp( ( - gsn[ix,jx] - overflowfactorsn[ix] ) * vhatn ) ) / 2 ) * dxi
                    denominatorn[ix] = ( np.sum( np.exp( ( - gln[ix,1:jx] - overflowfactorln[ix] ) * vhatn ) ) + \
                                         ( np.exp( ( - gln[ix,0] - overflowfactorln[ix] ) * vhatn ) + \
                                          np.exp( ( - gln[ix,jx] - overflowfactorln[ix] ) * vhatn ) ) / 2 ) * dxi
                else:
                    funcs = np.exp( ( - gsn[ix,:jx+1] - overflowfactorsn[ix] ) * vhatn )
                    funcl = np.exp( ( - gln[ix,:jx+1] - overflowfactorln[ix] ) * vhatn )
                    numeratorn[ix] = simpson(y=funcs,  dx=dxi)
                    denominatorn[ix] = simpson(y=funcl,  dx=dxi)
                '''
            deltagn = - np.log( numeratorn / denominatorn ) / vhatn - ( overflowfactorsn - overflowfactorln )
            # print(f'numeratorn={numeratorn[501]}, denominatorn={denominatorn[501]}, vhatn={vhatn}, overflowfactorsn={overflowfactorsn}, overflowfactorln={overflowfactorln}')
            dgmixn = dgmixnm1 + deltagn
            egn = np.max(np.abs(deltagn[1:-1]))
            # egn = np.max(np.abs(deltagn[1:-1] / dgmixn[1:-1]))
            if iterationdata is True: 
                gs_rg = np.append(gs_rg, gsn.reshape(1, nx, nymax), axis=0)
                gl_rg = np.append(gl_rg, gln.reshape(1, nx, nymax), axis=0)
                numerator_rg = np.append(numerator_rg, (numeratorn*np.exp(overflowfactorsn)).reshape(1, nx), axis=0)
                denominator_rg = np.append(denominator_rg, (denominatorn*np.exp(overflowfactorln)).reshape(1, nx), axis=0)
                deltag_rg = np.append(deltag_rg, deltagn.reshape(1, nx), axis=0)
                dgmix_rg = np.append(dgmix_rg, dgmixn.reshape(1, nx), axis=0)
                print(f'n = {n} : maximum error = {egn}, sum of error = {np.sum(np.abs(deltagn[1:-1] / dgmixn[1:-1]))}')
            if (egn < tol_dgmix): # ensure all ddgmix(x) < tolerance
                if iterationdata is True: 
                    print('Converged')
                break
            elif not np.isfinite(egn):
                print(f'Diveraged (egn={egn})')
                break
            elif (egn >= tol_dgmix) and (iteration is not None):
                if (n >= iteration - 1):
                    if iterationdata is True: 
                        print('Not converged (exceed the maximum iteration number)')
                    break

        if iterationdata is True: 
            self.gs_rg = gs_rg.copy()
            self.gl_rg = gl_rg.copy()
            self.numerator_rg = numerator_rg.copy()
            self.denominator_rg = denominator_rg.copy()
            self.__dgmix_rg = dgmix_rg.copy()
            self.deltag_rg = deltag_rg.copy()
            self.nid = n + 1
        else:
            dgmix_rg = dgmixn.reshape(1, nx)
            self.__dgmix_rg = dgmix_rg.copy()
            self.nid = 1
        self.iteration = n + 1
        
        return dgmix_rg

    def calc_dgmix_rgaw(self):
        """
        Parameters
        ----------
        xi : numpy array, float ncXnx
            mole fraction of component i  
        dxi : float
            differential mole fraction of component i  
        lhat0i : float
            one of Renormalization Group parameter 
        deltai : float
            one of Renormalization Group parameter 
        iteration : int or None
            total iteration time you want to calculate
        iterationdata : boolean
            Whether need to leave iteration data
        nx : int
            total number of mole fraction data
        nymax : int
            total number of mole fraction data for maximum varible y
        nid : int
            actual iteration data for gibbs free energy of mixing 
            (if iterationdata is True: iteration, else: 1)
        xmin : numpy array, float 1Xnx
            minimum mole fraction for all component in xi
            eg. if x[i] = np.array([0.3, 0.7]), xmin[i] = 0.3
        ymax : numpy array, float 1Xnymax
            minimum mole fraction for all component in xi
        ui : float
            one of Renormalization Group parameter (total mixing internal energy for x1=0.5)
        tol_dgmix : float
            the tolerance for Renormalization Group iteration 
        n : int
            iteration step
        d : int
            dimension
        dgmix0 : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data
        dgmixn : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        dgmixnm1 : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at (n-1)th iteration
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
        deltagn : numpy array, float 1Xnx
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        deltag_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data 
            (calculated by liquid modol+RG)
            
        """
        
        if self.__dgmix_rgaw is not None: # dgmix_rg have been calculated
            return self.__dgmix_rgaw
        dgmix_rg = self.calc_dgmix_rg()
        return dgmix_rg
    
    @property
    def dgmix_rg(self):
        """
        Parameters
        ----------
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
            
        """
        rgtype = self.rgtype
        if rgtype is None:
            return self.calc_dgmix_rg()
        elif rgtype == 'A':
            return self.calc_dgmix_rgaw()
    
    @property
    def ddgmixdx1_rg(self):
        """
        Parameters
        ----------
        ddgmixdx1_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            first derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_rg = self.dgmix_rg
        nid = self.nid
        ddgmixdx1_rg = np.zeros((nid, self.nx), dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                ddgmixdx1_rg[:,ix] = ( dgmix_rg[:,ix+1] - dgmix_rg[:,ix] ) / dxi
            elif ( ix == self.nx - 1 ):
                ddgmixdx1_rg[:,ix] = ( dgmix_rg[:,ix] - dgmix_rg[:,ix-1] ) / dxi
            else:
                ddgmixdx1_rg[:,ix] = ( dgmix_rg[:,ix+1] - dgmix_rg[:,ix-1] ) / ( 2 * dxi )
        return ddgmixdx1_rg
    
    @property
    def d2dgmixdx12_rg(self):
        """
        Parameters
        ----------
        d2dgmixdx12_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            second derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_rg = self.dgmix_rg # order of dgmix_rg and nid cannot change
        nid = self.nid
        d2dgmixdx12_rg = np.zeros((nid, self.nx), dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                d2dgmixdx12_rg[:,ix] = ( dgmix_rg[:,ix+2] - 2 * dgmix_rg[:,ix+1] + dgmix_rg[:,ix] ) / ( dxi **2 )
            elif ( ix == self.nx - 1 ):
                d2dgmixdx12_rg[:,ix] = ( dgmix_rg[:,ix-2] - 2 * dgmix_rg[:,ix-1] + dgmix_rg[:,ix] ) / ( dxi **2 )
            else:
                d2dgmixdx12_rg[:,ix] = ( dgmix_rg[:,ix+1] - 2 * dgmix_rg[:,ix] + dgmix_rg[:,ix-1] ) / ( dxi **2 )
        return d2dgmixdx12_rg
    
    @property
    def d3dgmixdx13_rg(self):
        """
        Parameters
        ----------
        d3dgmixdx13_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            third derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_rg = self.dgmix_rg # order of dgmix_rg and nid cannot change
        nid = self.nid
        d3dgmixdx13_rg = np.zeros((nid, self.nx), dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+4] - 2 * dgmix_rg[:,ix+3] + 2 * dgmix_rg[:,ix+1] - dgmix_rg[:,ix] ) / ( 2 * dxi **3 )
            elif ix == 1:
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+3] - 2 * dgmix_rg[:,ix+2] + 2 * dgmix_rg[:,ix] - dgmix_rg[:,ix-1] ) / ( 2 * dxi **3 )
            elif ( ix == self.nx - 2 ):
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+1] - 2 * dgmix_rg[:,ix] + 2 * dgmix_rg[:,ix-2] - dgmix_rg[:,ix-3] ) / ( 2 * dxi **3 )
            elif ( ix == self.nx - 1 ):
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix] - 2 * dgmix_rg[:,ix-1] + 2 * dgmix_rg[:,ix-3] - dgmix_rg[:,ix-4] ) / ( 2 * dxi **3 )
            else:
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+2] - 2 * dgmix_rg[:,ix+1] + 2 * dgmix_rg[:,ix-1] - dgmix_rg[:,ix-2] ) / ( 2 * dxi **3 )
        return d3dgmixdx13_rg
    
    @property
    def g_cg(self):
        """
        Parameters
        ----------
        g_cg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy divide by RT for all mole fraction data (calculated by liquid modol)
            
        """
        ti = self.t
        xi = self.x
        epsilon11dk = self.epsilon11dk
        epsilon22dk = self.epsilon22dk
        Nc = self.Nc
        dgmix_cg = self.dgmix_cg
        g_cg = dgmix_cg + ( xi[0] * epsilon11dk + xi[1] * epsilon22dk ) * Nc / ( 2 * ti ) 
        return g_cg
    
    @property
    def g_rg(self):
        """
        Parameters
        ----------
        g_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy divide by RT for all mole fraction data (calculated by liquid modol+RG)
            
        """
        ti = self.t
        xi = self.x
        epsilon11dk = self.epsilon11dk
        epsilon22dk = self.epsilon22dk
        Nc = self.Nc
        dgmix_rg = self.dgmix_rg
        g_rg = dgmix_rg + ( xi[0] * epsilon11dk + xi[1] * epsilon22dk ) * Nc / ( 2 * ti ) 
        return g_rg
    
    def calc_gt_cg(self): # G/Nk
        ti = self.t
        g_cg = self.g_cg
        gt_cg = g_cg * ti
        return gt_cg
    
    def calc_gt_rg(self): # G/Nk
        ti = self.t
        g_rg = self.g_rg
        gt_rg = g_rg * ti
        return gt_rg
    
    @property
    def mu_rg(self):
        """
        Parameters
        ----------
        mu_rg : numpy array, float (if iterationdata is True: nXncXnx, else: 1XncXnx) 
            chemical potential for component i in the mixture divide by RT (numerical equation)
        """
        xi = self.x
        nxi = self.nx
        nci = self.nc
        dgmix_rg = self.dgmix_rg
        nid = self.nid
        ddgmixdx1_rg = self.ddgmixdx1_rg
        mu_rg = np.zeros((nid, nci, nxi), dtype=float)
        for n in range(nid):
            mu_rg[n,0,:] = dgmix_rg[n,:] + xi[1] * ddgmixdx1_rg[n,:]
            mu_rg[n,1,:] = dgmix_rg[n,:] - xi[0] * ddgmixdx1_rg[n,:]
        return mu_rg
        
    @property
    def gammam_rg(self):
        """
        Parameters
        ----------
        gammam_rg : numpy array, float (if iterationdata is True: nXncXnx, else: 1XncXnx) 
            activity coefficient for component i in the mixture (numerical equation)
        """
        xi = self.x
        nxi = self.nx
        nci = self.nc
        mu_rg = self.mu_rg
        nid = self.nid
        gammam_rg = np.zeros((nid, nci, nxi), dtype=float)
        for ic in range(nci):
            gammam_rg[:,ic,:] = np.exp(mu_rg[:,ic,:]) / xi[ic]
        return gammam_rg
    
    @property
    def cp_cg(self):
        """
        Parameters
        ----------
        cp_cg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar heat capacity divide by R for all mole fraction data (calculated by liquid modol)
            
        """
        ti = self.t
        dt = self.dt
        d2gtdt2_cg = self.calc_d2gtdt2_cg(dt)
        cp_cg = - ti * d2gtdt2_cg
        return cp_cg
    
    @property
    def cp_rg(self):
        """
        Parameters
        ----------
        cp_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar heat capacity divide by R for all mole fraction data (calculated by liquid modol+RG)
            
        """
        ti = self.t
        dt = self.dt
        d2gtdt2_rg = self.calc_d2gtdt2_rg(dt)
        cp_rg = - ti * d2gtdt2_rg
        return cp_rg
    
    def calc_xec0(self):
        # get extreme value composition(first derivative of dgmix for x1 is zero) at zero iteration 
        xi = self.x
        ddgmix0dx1 = self.ddgmixdx1_cg
        nxi = self.nx
        nxti = nxi - 1
        nxhalf = nxti // 2
        nci = self.nc
        xeca0 = np.zeros(nci, dtype=float)
        xecb0 = np.zeros(nci, dtype=float)
        flag_xeca = False
        flag_xecb = False
        for ix in range(nxi):
            if ( ddgmix0dx1[ix] >= 0 ) and ( flag_xecb is False ):
                xecb0 = xi[:,ix]
                flag_xecb = True
            if ( ddgmix0dx1[nxti-ix] <= 0 ) and ( flag_xeca is False ):
                xeca0 = xi[:,nxti-ix]
                flag_xeca = True
            if ( flag_xeca is True ) and ( flag_xecb is True ):
                break
            elif ( ix > nxhalf ) and ( flag_xeca is False ) and ( flag_xecb is False ):
                xeca0 = np.full(nci, np.nan)
                xecb0 = np.full(nci, np.nan)
                break
        else:
            xeca0 = np.full(nci, np.nan)
            xecb0 = np.full(nci, np.nan)

        if xeca0[0] - xecb0[0] < 0.: # T>Tc
            xeca0 = np.full(nci, np.nan)
            xecb0 = np.full(nci, np.nan)
        self.xeca0 = xeca0
        self.xecb0 = xecb0 

    def calc_xic(self):
        # get inflection composition (second derivative of dgmix for x1 is zero)
        xi = self.x
        d2dgmixdx12_rg = self.d2dgmixdx12_rg
        nxi = self.nx
        nxti = nxi - 1
        nxhalf = nxti // 2
        nci = self.nc
        nid = self.nid
        #nica = np.zeros(nid, dtype=int)
        #nicb = np.zeros(nid, dtype=int)
        xica = np.zeros((nci, nid), dtype=float)
        xicb = np.zeros((nci, nid), dtype=float)
        tol = 0.
        for n in range(nid):
            flag_xica = False
            flag_xicb = False
            for ix in range(nxi): 
                if ( d2dgmixdx12_rg[n,ix] <= tol ) and ( flag_xicb is False ):
                    xicb[:,n] = xi[:,ix]
                    #nicb[n] = ix
                    flag_xicb = True
                if ( d2dgmixdx12_rg[n,nxti-ix] <= tol ) and ( flag_xica is False ):
                    xica[:,n] = xi[:,nxti-ix]
                    #nica[n] = nxti-ix
                    flag_xica = True
                if ( flag_xica is True ) and ( flag_xicb is True ):
                    break
                elif ( ix > nxhalf ) and ( flag_xica is False ) and ( flag_xicb is False ):
                    xica[:,n] = np.full(nci, np.nan)
                    xicb[:,n] = np.full(nci, np.nan)
                    break
            else:
                xica[:,n] = np.full(nci, np.nan)
                xicb[:,n] = np.full(nci, np.nan)
        self.xica = xica
        self.xicb = xicb
        #self.nica = nica
        #self.nicb = nicb
        self.xa = xica[:,-1]
        self.xb = xicb[:,-1]
    
    def calc_critical_tx_objfun(self, x):
        ti = x
        self.set_t(ti)
        self.calc_xic()
        x1ai = self.xa[0]
        x1bi = self.xb[0]
        ex = x1ai - x1bi
        return ex
    
    def calc_critical_tx(self, ucp=True, lcp=False, tgui=None, tgli=None, t_stepgu=None, t_stepgl=None):
        nci = self.nc
        dxi = self.dx
        tol_lle_dgmix = self.tol_lle_dgmix[self.liquidmodelname]
        tol_tc = 1e-14
           
        def solve_cp(tsi, t_step):
            toi = tsi
            t_stepo = t_step
            t_stepn = t_step
            exo = self.calc_critical_tx_objfun(toi)
            while np.isnan(exo): # if exo == nan (single phase), change initial temperature until exo > 0 (phase separation)
                toi = toi * 0.9 if t_step > 0 else toi * 1.1
                exo = self.calc_critical_tx_objfun(toi)
            while True:
                tni = toi + t_stepn
                exn = self.calc_critical_tx_objfun(tni)
                # print(tni, exn, t_stepn)
                if ( exn >= dxi ): # T<Tc
                    toi = tni  
                    if np.isnan(exo): # the ex of previous step is nan
                        t_stepn = 0.5 * t_stepo
                    else: # the ex of previous step is not nan
                        t_stepn = t_stepo
                    if ( abs(t_stepn) <= tol_tc ):
                        print('Not found CP')
                        tci = tni
                        xci = np.full(nci, np.nan)
                        break
                elif np.isnan(exn): # T>Tc
                    t_stepn = 0.5 * t_stepo
                else: # T=Tc
                    tci = tni
                    xci = self.xa
                    break
                exo = exn
                t_stepo = t_stepn
            # print('converged', tci, xci)
            return tci, xci

        if ucp is True:
            tsu = ( tgui - 50 ) if tgui is not None else 280 # K
            t_stepu = t_stepgu if t_stepgu is not None else 100
            tcu, xcu = solve_cp(tsu, t_stepu)
        else:
            tcu, xcu = np.full(2, np.nan)
        if lcp is True:
            tgl = ( tgli + 20 ) if ( tgli is not None ) else ( tcu - 90 ) if ( ucp is True ) else 380 # K
            t_stepl = t_stepgl if t_stepgl is not None else - 50
            tcl, xcl = solve_cp(tgl, t_stepl)
        else:
            tcl, xcl = np.full(2, np.nan)
        tc = {'UCP':tcu, 'LCP':tcl}
        xc = {'UCP':xcu, 'LCP':xcl}
        return tc, xc

        
        
