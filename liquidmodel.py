import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

class Context():
    """
    The Context defines the interface of interest to clients.
    """
    
    def __init__(self, strategy):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """
        self._strategy = strategy

    @property
    def strategy(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy
        
    @property
    def mixture(self):
        """
        The Context maintains a reference to one of the mixture objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._mixture

    @mixture.setter
    def mixture(self, mixture):
        """
        Usually, the Context allows replacing a mixture object at runtime.
        """
        self._mixture = mixture
        self._strategy.set_mixture(mixture)
        
    def set_tx(self, t, x):
        """
        Parameters
        ----------
        t : float
            temperture (K)
        x : numpy array, 1Xn float
            mole fraction of component
            eg. x = np.array([x1, x2, ..., xn])
        """
        self.t = t
        self.x = x
        self._strategy.set_tx(t, x)
        
    def set_bip(self, a, b, c, d, e, f):
        """
        Parameters
        ----------
        a~f : numpy array, float nXn matrix 
            binary interaction parameters
        ----------
        Gij = exp(-alphaij * tauij)
        tauij = aij + bij / t + eij * ln(t) + fij * t
        alphaij = cij + dij * (t-273.15K)
        tauii = 0
        Gii = 1
        """  
        self._strategy.set_bip(a, b, c, d, e, f)  
        
    def set_epsilon(self, epsilon12dk, epsilon11dk, epsilon22dk, Nc=4, Nc_array=None):
        """
        Parameters
        ----------
        epsilon12dk, epsilon11dk, epsilon22dk: float
            potential energy for 1-2, 1-1, 2-2 pair (K)
        Nc : int
            coordination number
        """
        self._strategy.set_epsilon(epsilon12dk, epsilon11dk, epsilon22dk, Nc, Nc_array)  

    def set_cp(self, dbname, method):
        """ 
        Parameters
        ----------
        dbname : string
            name of database used in COSMOSAC model from MIT group (in class COSMOSAC_MIT)
            please set one of ['VT', 'UD', 'G09']
        method: string
            version of COSMOSAC model from MIT group (in class COSMOSAC_MIT)
            please set one of ['COSMOSAC-2002','COSMOSAC-2010','COSMOSAC-dsp']
            
        """ 
        self._strategy.set_cp(dbname, method)
        
    @property
    def nc(self):
        """ The number of component in the mixture """
        return self._strategy.nc
    
    @property
    def tauij(self):
        """ The NRTL and Wilson parameters for given mixture """
        return self._strategy._calc_tau()
    
    @property
    def Gij(self):
        """ The NRTL and Wilson parameters for given mixture """
        return self._strategy._calc_G()

    def u(self):
        """ The interaction potential for each given components """
        return self._strategy._get_u()
        
    def path(self):
        """ The path of COSMO file for each given components """
        return self._strategy.get_path()
    
    def tvc(self):
        """ Total volume of cavity for each components """
        return self._strategy._get_tvc()    
    
    def segnum(self):
        """ The Number of segments for each components """
        return self._strategy._get_segnum() 

    def cosmodata(self):
        """ The data of COSMO file for each components"""
        return self._strategy._get_cosmodata() 
    
    def sigmaarea(self):
        """ The sigma area of each components within 51 segments """
        return self._strategy._get_sigmaarea()
    
    def sumsigmaarea(self):
        """ The molecule area of each components """
        return self._strategy._get_sumsigmaarea()
    
    def sigmaprofile(self):
        """ The sigma profile of each components """
        return self._strategy._get_sigmaprofile()
    
    def dw(self):
        """ The exchange energy of each components """
        return self._strategy._get_dw()
    
    def seggamma(self):
        """ The segment activity coefficients of each pure components """
        return self._strategy._get_seggamma()
    
    def seggammam(self):
        """ The segment activity coefficients of given mixture """
        return self._strategy._get_seggammam()
    
    def sigmaprofilem(self):
        """ The sigma profile of given mixture """
        return self._strategy._get_sigmaprofilem()
    
    def gammaSG(self):
        """ The Staverman-Guggenheim activity coefficicent of given mixture """
        return self._strategy._get_gammaSG() 
        
    """
    The Context delegates some work to the Strategy object instead of
    implementing multiple versions of the algorithm on its own.
    """
    
    def calc_gammaminf(self):
        """ Start to calculate infintie dilution activity coefficients of each component within the given mixture """
        gammaminf = self._strategy._get_gammaminf()
        self.__gammaminf = gammaminf
        return gammaminf
    
    def calc_gammam(self):
        """ Start to calculate activity coefficients of each component within the given mixture """
        gammam = self._strategy._get_gammam()
        self.__gammam = gammam
        return gammam

    def calc_mu(self):
        """ Start to calculate chemical potentials of each component within the given mixture """
        mu = self._strategy._get_mu()
        self.__mu = mu
        return mu
    
    def calc_dmudxa(self):
        """ Start to calculate first derivatives of chemical potentials of each component (analytic) """
        dmudxa = self._strategy._get_dmudxa()
        self.__dmudxa = dmudxa
        return dmudxa
    
    def calc_dmudxn(self, dx=1.e-4):
        """ Start to calculate first derivatives of chemical potentials of each component (numerical) """
        dmudxn = self._strategy._get_dmudxn(dx)
        self.__dmudxn = dmudxn
        return dmudxn
        
    def calc_d2mudx2a(self):
        """ Start to calculate second derivatives of  chemical potentials of each component (analytic) """
        d2mudx2a = self._strategy._get_d2mudx2a()
        self.__d2mudx2a = d2mudx2a
        return d2mudx2a
    
    def calc_d2mudx2n(self, dx=1.e-4):
        """ Start to calculate first derivatives of chemical potentials of each component (numerical) """
        d2mudx2n = self._strategy._get_d2mudx2n(dx)
        self.__d2mudx2n = d2mudx2n
        return d2mudx2n
        
    def calc_gex(self):
        """ Start to calculate molar excess gibbs free energy of the given mixture """
        gex = self._strategy._get_gex()
        self.__gex = gex
        return gex
        
    def calc_hexa(self):
        """ Start to calculate molar excess enthalpy of the given mixture (analytic) """
        hexa = self._strategy._get_hexa()
        self.__hexa = hexa
        return hexa
    
    def calc_hexn(self, dt = 1.e-3):
        """ Start to calculate molar excess enthalpy of the given mixture (numerical) """
        hexn = self._strategy._get_hexn(dt)
        self.__hexn = hexn
        return hexn
    
    def calc_sexa(self):
        """ Start to calculate molar excess entropy of the given mixture (analytic) """
        sexa = self._strategy._get_sexa()
        self.__sexa = sexa
        return sexa
    
    def calc_sexn(self, dt = 1.e-3):
        """ Start to calculate molar excess entropy of the given mixture (numerical) """
        sexn = self._strategy._get_sexn(dt)
        self.__sexn = sexn
        return sexn
    
    def calc_dgmix(self):
        """ Start to calculate molar mixing gibbs free energy of the given mixture """
        dgmix = self._strategy._get_dgmix()
        self.__dgmix = dgmix
        return dgmix
    
    def calc_dsmixa(self):
        """ Start to calculate molar entropy of mixing of the given mixture (analytic) """
        dsmixa = self._strategy._get_dsmixa()
        self.__dsmixa = dsmixa
        return dsmixa
    
    def calc_dsmixn(self):
        """ Start to calculate molar entropy of mixing of the given mixture (numerical) """
        dsmixn = self._strategy._get_dsmixn()
        self.__dsmixn = dsmixn
        return dsmixn
    
    def calc_sa(self):
        """ Start to calculate molar entropy of the given mixture (analytic) """
        sa = self._strategy._get_sa()
        self.__sa = sa
        return sa
    
    def calc_sn(self):
        """ Start to calculate molar entropy of the given mixture (numerical) """
        sn = self._strategy._get_sn()
        self.__sn = sn
        return sn
        
    @property
    def get_gammam(self):
        """ Get the activity coefficient """
        return self.__gammam
    
    @property
    def get_gex(self):
        """ Get the molar excess gibbs free energy """
        return self.__gex
    
    @property
    def get_mu(self):
        """ Get the chemical potentials of each component """
        return self.__mu


def decorator_dmdtn(mi):
    """ 
    A decorator for numerically calculate first derivative of temperature of properties. 
    """
    def method(mth):
        def wrapper(self, dt):
            """
            Parameters
            ----------
            dti : float
                step size of temperature for the numerical differentiation
            ti : float
                temperture (K) 
            mi : function
                any property that is function of temperture
            mth : regular method
                the regular method you want to decorate it.
            mplusi : float or numpy array, float nX1 matrix
                property in the state of ti + dti
            mminusi : float or numpy array, float nX1 matrix
                property in the state of ti - dti
            dmdtni : float or numpy array, float nXn matrix
                first derivatives of the property (numerical equation)

            """
            dti = dt
            ti = self.t
            self.t = ti + dti
            mplusi = mi(self)
            self.t = ti - dti
            mminusi = mi(self)
            dmdtni = ( mplusi - mminusi ) / ( 2 * dti )
            self.t = ti
            return dmdtni
        return wrapper
    return method

def decorator_d2mdt2n(mi):
    """ 
    A decorator for numerically calculate second derivative of temperature of properties. 
    """
    def method(mth):
        def wrapper(self, dt):
            """
            Parameters
            ----------
            dti : float
                step size of temperature for the numerical differentiation
            ti : float
                temperture (K) 
            mi : function
                any property that is function of temperture
            mth : regular method
                the regular method you want to decorate it.
            mci : float or numpy array, float nX1 matrix
                property in the state of ti
            mplusi : float or numpy array, float nX1 matrix
                property in the state of ti + dti
            mminusi : float or numpy array, float nX1 matrix
                property in the state of ti - dti
            d2mdt2ni : float or numpy array, float nXn matrix
                second derivatives of the property (numerical equation)

            """
            dti = dt
            ti = self.t
            mci = mi(self)
            self.t = ti + dti
            mplusi = mi(self)
            self.t = ti - dti
            mminusi = mi(self)
            d2mdt2ni = ( mplusi - 2 * mci + mminusi ) / ( dti **2 )
            self.t = ti
            return d2mdt2ni
        return wrapper
    return method

def decorator_dmdx1n(mi):
    """ 
    A decorator for numerically calculate first derivative of x1 of properties. 
    """
    def method(mth):
        def wrapper(self, dx=1.e-4):
            """
            Parameters
            ----------
            dxi : float
                step size of mole fraction of component i for the numerical differentiation
            xi : numpy array, float nX1 matrix
                mole fraction of component i  
            mi : function
                any property that is function of mole fraction 
            mth : regular method
                the regular method you want to decorate it.
            mplusi : float or numpy array, float nX1 matrix
                property in the state of xi + dxi
            mminusi : float or numpy array, float nX1 matrix
                property in the state of xi - dxi
            dmdx1ni : float or numpy array, float nXn matrix
                first derivatives of the property (numerical equation)

            """
            dxi = dx
            xi = self.x
            x1i, x2i = xi
            self.x = np.array([x1i + dxi, x2i - dxi])
            mplusi = mi(self)
            self.x = np.array([x1i - dxi, x2i + dxi])
            mminusi = mi(self)
            dmdx1ni = ( mplusi - mminusi ) / ( 2 * dxi )
            self.x = xi
            return dmdx1ni
        return wrapper
    return method

def decorator_d2mdx12n(mi): 
    """ 
    A decorator for numerically calculate second derivative of x1 of properties. 
    """
    def method(mth):
        def wrapper(self, dx=1.e-4):
            """
            Parameters
            ----------
            dxi : float
                step size of mole fraction of component i for the numerical differentiation
            xi : numpy array, float nX1 matrix
                mole fraction of component i  
            mi : function
                any property that is function of mole fraction 
            mth : regular method
                the regular method you want to decorate it.
            mci : float or numpy array, float nX1 matrix
                property in the state of xi
            mplusi : float or numpy array, float nX1 matrix
                property in the state of xi + dxi
            mminusi : float or numpy array, float nX1 matrix
                property in the state of xi - dxi
            d2mdx12ni : float or numpy array, float nXn matrix
                second derivatives of the property (numerical equation)

            """
            dxi = dx
            xi = self.x
            mci = mi(self)
            x1i, x2i = xi
            self.x = np.array([x1i + dxi, x2i - dxi])
            mplusi = mi(self)
            self.x = np.array([x1i - dxi, x2i + dxi])
            mminusi = mi(self)
            d2mdx12ni = ( mplusi - 2 * mci + mminusi ) / ( dxi **2 )
            self.x = xi
            return d2mdx12ni
        return wrapper
    return method

class LiquidModelStrategy(metaclass=ABCMeta):
    
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """
    
    """
    Parameters
    ----------
    gpure : float
        molar Gibbs free energy of pure component i divide by RT (reference state)
    spure : float
        molar entropy of pure component i divide by R (reference state)
    """
    
    gpure = 0 
    spure = 0 
    dt = 1.e-3
    dx = 1.e-4
    
    def set_tx(self, t, x):
        """
        Parameters
        ----------
        t : float
            temperture (K)
        x : numpy array, float nX1 matrix
            mole fraction of component i  
            eg. x = np.array([x1, x2, ..., xn])
        """
        self.t = t
        self.x = x
    
    @abstractmethod
    def set_bip(self, a, b, c, d, e, f):
        pass
    
    @abstractmethod
    def set_mixture(self, mixture):
        pass
    
    @abstractmethod
    def _get_u(self):
        pass
    
    @abstractmethod
    def _get_gammam(self):
        pass
    
    def _get_mu(self):
        """
        Parameters
        ----------
        gpurei : float
            molar Gibbs free energy of pure component i
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        ti: float
            temperture (K)
        gammami : numpy array, float nX1 matrix
            activity coefficient of mixture of component i
        mui : numpy array, float nX1 matrix
            chemical potential of mixture of component i divide by RT

        """
        gpurei = self.gpure
        xi = self.x
        gammami = self._get_gammam()
        mui = gpurei + np.log(xi * gammami)
        return mui
    
    @abstractmethod
    def _get_dmudxa(self):
        pass
    
    @decorator_dmdx1n(mi=_get_mu)
    def _get_binary_dmudxn(self, dx=1.e-4):
        pass
    
    def _get_dmudxn(self, dx=1.e-4):
        """
        Parameters
        ----------
        nci : interger
            number of components
        dmu12dx1i : numpy array, float 1Xn matrix
            first derivatives of chemical potential of component 1 and 2 divide by RT (numerical equation)
            for binary systems: dmu12dx1i = np.array([dmu1dx1, dmu2dx1])
        dmudxi : numpy array, float nXn matrix
            first derivatives of chemical potential divide by RT (numerical equation)
            for binary systems: dmudxi = np.array([dmu1dx1, dmu2dx1], [dmu1dx2, dmu2dx2])
        """
        nci = self.nc
        if nci == 2:
            dmu12dx1i = self._get_binary_dmudxn(dx)
            dmudxi = np.array([[dmu12dx1i[0], dmu12dx1i[1]], [-dmu12dx1i[0], -dmu12dx1i[1]]])
            
        else:
            pass
        return dmudxi
    
    @abstractmethod
    def _get_d2mudx2a(self):
        pass
    
    @decorator_d2mdx12n(mi=_get_mu)
    def _get_binary_d2mudx2n(self, dx=1.e-4):
        pass
    
    def _get_d2mudx2n(self, dx=1.e-4):
        """
        Parameters
        ----------
        nci : interger
            number of components
        d2mu12dx12i : numpy array, float 1Xn matrix
            second derivatives of chemical potential of component 1 and 2 divide by RT (numerical equation)
            for binary systems: d2mu12dx12i = np.array([d2mu1dx12i, d2mu2dx12i])
        d2mudx2i : numpy array, float nXn matrix
            second derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([d2mu1dx12, d2mu2dx12], [d2mu1dx22, d2mu2dx22])

        """
        nci = self.nc
        if nci == 2:
            d2mu12dx12i = self._get_binary_d2mudx2n(dx)
            d2mudx2i = np.array([[d2mu12dx12i[0], d2mu12dx12i[1]], [d2mu12dx12i[0], d2mu12dx12i[1]]])
            
        else:
            pass
        return d2mudx2i
    
    def _get_gex(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        gammami : numpy array, float nX1 
            activity coefficient of mixture of component i
        gexi : float
            excess total molar Gibbs free energy divide by RT
        """
        xi = self.x
        if np.all(xi != 0):
            gammami = self._get_gammam()
            gexi = np.sum( xi * np.log(gammami) )
        else:
            gexi = 0
        return gexi
    
    @abstractmethod
    def _get_hexa(self):
        pass
    
    def _get_hexn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        ti: float
            temperture (K)
        dti: float
            differential temperture (K)
        hexi : float
            excess total molar enthalpy divide by RT (numerical)

        """
        dti = dt
        ti = self.t
        self.t = ti + dti
        gexplusi = self._get_gex()
        self.t = ti - dti
        gexminusi = self._get_gex()
        hexi = - ti * ( gexplusi - gexminusi ) / ( 2 * dti )
        self.t = ti
        return hexi
    
    def _get_sexa(self):
        """
        Parameters
        ----------
        gexi : float
            excess total molar Gibbs free energy divide by RT
        hexi : float
            excess total molar enthalpy divide by RT (analytical)
        sexi : float
            excess total molar entropy divide by R (analytical)
        """
        gexi = self._get_gex()
        hexi = self._get_hexa()
        sexi = hexi - gexi
        return sexi
    
    def _get_sexn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        dti: float
            differential temperture (K)
        gexi : float
            excess total molar Gibbs free energy divide by RT
        hexi : float
            excess total molar enthalpy divide by RT (numerical)
        sexi : float
            excess total molar entropy divide by R (numerical)
        """
        dti = dt
        gexi = self._get_gex()
        hexi = self._get_hexn(dti)
        sexi = hexi - gexi
        return sexi

    def _get_dgidmix(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        dgidmixi : float
            total molar Gibbs free energy of ideal mixing divide by RT
        """
        xi = self.x
        dgidmixi = sum( xi * np.log( xi ) ) if np.all(xi != 0) else 0.
        return dgidmixi
    
    def _get_dgmix(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        gexi : float
            excess total molar Gibbs free energy divide by RT
        dgidmixi : float
            total molar Gibbs free energy of ideal mixing divide by RT
        dgmixi : float
            total molar Gibbs free energy of mixing divide by RT
        """
        xi = self.x
        gexi = self._get_gex()
        dgidmixi = self._get_dgidmix()
        dgmixi = gexi + dgidmixi
        return dgmixi
    
    def _get_dsidmix(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        dsidmixi : float
            total molar entropy of ideal mixing divide by RT
        """
        xi = self.x
        dsidmixi = - sum( xi * np.log( xi ) ) if np.all(xi != 0) else 0.
        return dsidmixi
    
    def _get_dsmixa(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        sexi : float
            excess total molar entropy divide by R (analytical)
        dsidmixi : float
            total molar entropy of ideal mixing divide by RT
        dsmixi : float
            total molar entropy of mixing divide by R (analytical)
        """
        xi = self.x
        sexi = self._get_sexa()
        dsidmixi = self._get_dsidmix()
        dsmixi = sexi + dsidmixi
        return dsmixi
    
    def _get_dsmixn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        dti: float
            differential temperture (K)
        xi : numpy array, float nX1 matrix
            mole fraction of component i 
        spure : float
            molar entropy of pure component i divide by R
        sexi : float
            excess total molar entropy divide by R (numerical)
        dsidmixi : float
            total molar entropy of ideal mixing divide by RT
        dsmixi : float
            total molar entropy of mixing divide by R (numerical)
        """
        dti = dt
        xi = self.x
        sexi = self._get_sexn(dti)
        dsidmixi = self._get_dsidmix()
        dsmixi = sexi + dsidmixi
        return dsmixi
    
    def _get_sa(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        spure : float
            molar entropy of pure component i divide by R
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        dsmixi : float
            total molar entropy of mixing divide by R (analytical)
        si : float
            total molar entropy divide by R (analytical)
        """
        xi = self.x
        spurei = self.spure
        dsmixi = self._get_dsmixa()
        si = dsmixi + sum( xi * spurei )
        return si
    
    def _get_sn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        spure : float
            molar entropy of pure component i divide by R
        dsmixi : float
            total molar entropy of mixing divide by R (numerical)
        si : float
            total molar entropy divide by R (numerical)
        """
        xi = self.x
        dti = dt
        spurei = self.spure
        dsmixi = self._get_dsmixn(dti)
        si = dsmixi + sum( xi * spurei )
        return si
    
    
"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class NRTL(LiquidModelStrategy):
          
    def set_bip(self, a, b, c, d, e, f):
        """
        Parameters
        ----------
        a~f : numpy array, float nXn matrix 
            binary interaction parameters
        ----------
        Gij = exp(-alphaij * tauij)
        tauij = aij + bij / t + eij * ln(t) + fij * t
        alphaij = cij + dij * (t-273.15K)
        tauii = 0
        Gii = 1
        """
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d
        self.__e = e
        self.__f = f
        self.__check_bip()
        
    def set_mixture(self, mixture):
        self._mixture = mixture
        
    def __check_bip(self):
        """
        Parameters
        ----------
        aij~fij : numpy array, float nXn matrix
            binary interaction parameters
        nc : interger
            number of components
        """
        aij = self.__a
        bij = self.__b
        cij = self.__c
        dij = self.__d
        eij = self.__e
        fij = self.__f
        if np.shape(aij) == np.shape(bij) == np.shape(cij) == np.shape(dij) == np.shape(eij) == np.shape(fij) and \
           np.shape(aij)[0] == np.shape(aij)[1]:
            self.nc = np.shape(aij)[0]
        else:
            print('The shape of BIPs are not consistant.')
        
    def _calc_tau(self):
        """
        Parameters
        ----------
        aij~fij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        aij = self.__a
        bij = self.__b
        eij = self.__e
        fij = self.__f
        ti = self.t
        if np.all( eij == 0. ):
            tauij = aij + bij / ti + fij * ti
        else:
            tauij = aij + bij / ti + eij * np.log( ti ) + fij * ti
        return tauij
        
    def _calc_dtaudt(self):
        """
        Parameters
        ----------
        bij~fij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        dtaudtij: numpy array, float nXn matrix
            one of temperature derivative of binary interaction parameters

        """
        bij = self.__b
        eij = self.__e
        fij = self.__f
        ti = self.t
        dtaudtij = - bij / ti **2 + eij / ti + fij
        return dtaudtij
    
    def _calc_alpha(self):
        """
        Parameters
        ----------
        cij, dij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        alphaij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        cij = self.__c
        dij = self.__d
        ti = self.t
        alphaij = cij + dij * ( ti - 273.15 )
        return alphaij
        
    def _calc_G(self):
        """
        Parameters
        ----------
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters
        alphaij: numpy array, float nXn matrix
            one of binary interaction parameters
        Gij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        return Gij
        
    def _calc_dalphadt(self):
        """
        Parameters
        ----------
        dij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        dalphadtij: numpy array, float nXn matrix
            one of temperature derivative of binary interaction parameters

        """
        dij = self.__d
        dalphadtij = dij
        return dalphadtij
        
    def _get_u(self):
        """
        Parameters
        ----------
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters
        ui : float
            Interaction potential of the mixture divide by RT
        """
        tauij = self._calc_tau()
        ui = np.sum(tauij)
        return ui
        
    def _get_gammaminf(self):
        """
        Parameters
        ----------
        nci : interger
            number of components
        alphaij, tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        gammaminfi : numpy array, float nX1 matrix
            Infinite dilution activity coefficient of mixture of component i

        """
        nci = self.nc
        tauij = self._calc_tau()
        Gij = self._calc_G()
        if nci == 2:
            gammaminfi = np.exp(np.array([tauij[1,0] + tauij[0,1] * Gij[0,1], tauij[0,1] + tauij[1,0] * Gij[1,0]]))
        return gammaminfi
    
    def _get_gammam(self):
        """
        Parameters
        ----------
        n : interger
            number of component
        xi : numpy array, float nX1
            mole fraction of component i  
        Xi : numpy array, float nXn matrix
            mole fraction of component i
            eg. Xi=[[x1, x2, ...],[x1, x2, ...],...]
        tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        gammami : numpy array, float nX1 matrix
            activity coefficient of mixture of component i

        """
        n = len(self.x)
        xi = self.x.reshape(n, 1)
        Xi = np.array([self.x.T for i in range(n)])
        tauij = self._calc_tau()
        Gij = self._calc_G()
        gammami = np.exp( ( xi.T.dot( tauij * Gij ) / ( xi.T.dot( Gij ) ) ).T + \
                   np.array([np.diag( ( Gij * Xi / ( Xi.dot( Gij ) ) ).dot( ( tauij - \
                            ( Xi.dot( tauij * Gij ) )  / ( Xi.dot( Gij ) ) ).T ) ) ] ).T )
        return gammami[:,0]
    
    def _get_dmudxa(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i
        nci : interger
            number of components
        alphaij, tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        dmudxi : numpy array, float nXn matrix
            first derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([dmu1dx1, dmu2dx1], [dmu1dx2, dmu2dx2])

        """
        xi = self.x
        nci = self.nc
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        dmudxi = np.zeros((nci,nci), dtype=float)
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            tau12, tau21 = tauij[0,1], tauij[1,0]
            G12, G21 = Gij[0,1], Gij[1,0]
            dmu1dx1 = - 2 * x2 * ( tau21 * ( G21 / ( x1 + x2 * G21 ) ) **2 + G12 * tau12 / ( ( x2 + x1 * G12 ) **2) ) \
                          - 2 * x2 **2 * ( tau21 * G21 **2 * ( 1 - G21 ) / ( ( x1 + x2 * G21 ) **3 ) \
                                          + G12 * tau12 * ( G12 - 1 ) / ( ( x2 + x1 * G12 ) **3 ) ) + 1 / x1
            dmu2dx1 = np.nan
            dmu1dx2 = np.nan
            dmu2dx2 = np.nan
            dmudxi[0,0], dmudxi[0,1], dmudxi[1,0], dmudxi[1,1] = dmu1dx1, dmu2dx1, dmu1dx2, dmu2dx2
        else:
            pass
        return dmudxi
        
    def _get_d2mudx2a(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        nci : interger
            number of components
        alphaij, tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        d2mudx2i : numpy array, float nXn matrix
            second derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([d2(mu1)d(x1)2, d2(mu2)d(x1)2], [d2(mu1)d(x2)2, d2(mu2)d(x2)2])

        """
        xi = self.x
        nci = self.nc
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        d2mudx2i = np.zeros((nci,nci), dtype=float)
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            tau12, tau21 = tauij[0,1], tauij[1,0]
            G12, G21 = Gij[0,1], Gij[1,0]
            d2mu1dx12 = 2 * ( tau21 * ( G21 / ( x1 + x2 * G21 ) ) **2 + G12 * tau12 / ( ( x2 + x1 * G12) **2 ) ) + \
                        8 * x2 * ( tau21 * G21 **2 * ( 1 - G21 ) / ( ( x1 + x2 * G21 ) **3 ) + \
                                  G12 * tau12 * ( G12 - 1 ) / ( ( x2 + x1 * G12 ) **3) ) + \
                        6 * x2 **2 * ( tau21 * G21 **2 * ( 1 - G21 ) **2 / ( ( x1 + x2 * G21 ) **4 ) + \
                                      G12 * tau12 * ( G12 - 1 ) **2 / ( ( x2 + x1 * G12 ) **4 ) ) - 1 / ( x1 **2 )
            d2mu2dx12 = np.nan
            d2mu1dx22 = np.nan
            d2mu2dx22 = np.nan
            d2mudx2i[0,0], d2mudx2i[0,1], d2mudx2i[1,0], d2mudx2i[1,1] = d2mu1dx12, d2mu2dx12, d2mu1dx22, d2mu2dx22
        else:
            pass
        return d2mudx2i
    
    def _get_hexa(self):
        """
        Parameters
        ----------
        n : interger
            number of component
        xi : numpy array, float nX1
            mole fraction of component i  
        alphaij, tauij, Gij : numpy array, float nXn matrix
            binary interaction parameters.
        dalphadtij, dtaudtij, dGdtij: numpy array, float nXn matrix
            temperature derivative of binary interaction parameters
        hexi : float
            excess total molar enthalpy divide by RT

        """
        xi = self.x
        ti= self.t
        nci = self.nc
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        dtaudtij = self._calc_dtaudt()
        dalphadtij = self._calc_dalphadt()
        dGdtij = - np.exp( - alphaij * tauij ) * ( alphaij * dtaudtij + dalphadtij * tauij )
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            tau12, tau21 = tauij[0,1], tauij[1,0]
            G12, G21 = Gij[0,1], Gij[1,0]
            dtaudt12, dtaudt21 = dtaudtij[0,1], dtaudtij[1,0]
            dGdt12, dGdt21 = dGdtij[0,1], dGdtij[1,0]
            hexi = - ti * x1 * x2 * ( ( x1 * ( dGdt21 * tau21 + G21 * dtaudt21 ) + x2 * G21 **2 * dtaudt21 ) / ( x1 + x2 * G21 ) **2 + \
                                     ( x2 * ( dGdt12 * tau12 + G12 * dtaudt12 ) + x1 * G12 **2 * dtaudt12 ) / ( x2 + x1 * G12 ) **2 )
        return hexi
    
        
def create(kind):
    """
    Create an liquidmodel object.

    Parameters
    ----------
    kind : string
        Type of liquidmodel
    Returns
    -------
    class of liquid model
    """
    kind_list = ['NRTL']
    if kind == kind_list[0]:
        return NRTL()
    else:
        raise ValueError(f'kind must be one of {kind_list}')
